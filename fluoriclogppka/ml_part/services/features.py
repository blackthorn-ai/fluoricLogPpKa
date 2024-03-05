__all__ = [
    "DipoleMoment",
    "MoleculeVolume",
    "MoleculeSASA",
    "MoleculeTPSAF",
    "MoleculeDihedralAngle",
    "DistanceBetweenX1X2",
    "DistanceBetweenR1R2",
    "AngleX1X2R2",
    "AngleX2X1R1",
    "AngleR2X2R1",
    "AngleR1X1R2",
    "MoleculeRingsAmount",
    "AtomsToRingRatio",
    "Chirality"
]

import math

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import rdForceFieldHelpers, rdPartialCharges
from rdkit.Chem import rdFreeSASA, rdchem, rdMolTransforms
from rdkit.Chem.rdchem import RWMol
from rdkit.Geometry import Point3D

from fluoriclogppka.ml_part.constants import Identificator
from fluoriclogppka.ml_part.constants import FUNCTIONAL_GROUP_TO_SMILES
from fluoriclogppka.ml_part.exceptions import InvalidMoleculeTypeError
from fluoriclogppka.ml_part.services.utils import cycles_amount

class OptimizedMolecule:
    """
    Class that represents a geometrically optimized molecule.

    This class provides functionality to obtain optimized molecule, 
    its conformer with lowest energy and atoms indexes for calculating 3D features.

    Attributes:
        smiles (str): String representation of a molecule.
        f_group (str): Fluor functional group name. Defaults to None.
        functional_group_to_smiles (dict): Dict to convert string f_group to smiles. Defaults to None
        identificator (Identificator): The molecule type. Defaults to None.

    Methods:
        find_conf_with_min_energy(): Get the conformer index with minimal energy.
        prepare_molecule(): Creates 3D sanitized molecule with multiple conformers from smiles.
        set_average_atoms_position(): Creates virtual atom in the middle of given atoms.
        change_vector_direction(): Changes the direction of the atoms vector.
        is_atom_in_cycle(): Checks if atom is in cycle.
        find_X1X2R1R2(): Determines atoms indexes to X1, X2, R1, R2.
        optimize_geometry(): Optimizes molecule's conformers and returns with min energy.
    """
    def __init__(self, 
                 smiles: str,
                 f_group: str = None,
                 identificator: Identificator = None):
        """
        Initialize the OptimizedMolecule instance and calculates atoms indexes for 3d features.
        If fluor functional groups or identificator is None X1, X2, R1, R2 cannot be calculated.

        Args:
            smiles (str): String representation of a molecule.
            f_group (str): Fluor functional group name. Defaults to None.
            identificator (Identificator): The molecule type. Defaults to None.
        """
        self.smiles = smiles
        self.f_group = f_group
        self.functional_group_to_smiles = FUNCTIONAL_GROUP_TO_SMILES
        self.identificator = identificator

        self.optimize_geometry()

        if f_group is not None and identificator is not None:
            self.X1, self.X2, self.R1, self.R2 = self.find_X1X2R1R2()

    @staticmethod
    def find_conf_with_min_energy(mol):
        """
        Optimizes all molecules conformers and finds the one with the lowest energy.
        
        Args:
            mol: 3D sanitized molecule with multiple conformers. 

        Returns:
            min_energy_conf_index (int): Conformer index with lowest energy.
            min_energy (float): The lowest conformer energy of the optimized molecule.
            mol: Rdkit optimized molecule.
        """
        optimization_result = rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(mol)
        
        min_energy, min_energy_conf_index = pow(10,5), None
        for index, (status, energy) in enumerate(optimization_result):
            if energy < min_energy and status == 0:
                min_energy_conf_index = index
                min_energy = min(min_energy, energy)

        return min_energy_conf_index, min_energy, mol
    
    @staticmethod
    def prepare_molecule(smiles):
        """
        Create rdkit 3d molecule from SMILES.
        Generate charges and molecule's conformers.
        
        Args:
            smiles (str): String representation of a molecule. 
            
        Returns:
            mol: Rdkit sanitized molecule with generated charges and multiple conformers.
        """
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        rdForceFieldHelpers.MMFFSanitizeMolecule(mol)
        
        num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        amount_of_confs = pow(3, num_rotatable_bonds + 3)
        AllChem.EmbedMultipleConfs(mol, numConfs=amount_of_confs, randomSeed=3407)

        rdPartialCharges.ComputeGasteigerCharges(mol)

        return mol

    @staticmethod
    def set_average_atoms_position(mol,
                                   atoms_idx: list(),
                                   conf_id: int = -1):
        """
        Create virtual atom in the molecule in the middle of atoms with atoms_idx.
        
        Args:
            mol: Rdkit optimized molecule.
            atoms_idx ([int]): Molecules indexes to find their average position.
            conf_id: Conformer id of the molecule. Defaults to -1.
            
        Returns:
            mol: Rdkit molecule with new virtual atom.
            idx (int): index of new virtual atom.
        """
        x, y, z = 0, 0, 0
        for atom_idx in atoms_idx:
            x += mol.GetConformer(conf_id).GetAtomPosition(atom_idx)[0]
            y += mol.GetConformer(conf_id).GetAtomPosition(atom_idx)[1]
            z += mol.GetConformer(conf_id).GetAtomPosition(atom_idx)[2]
        
        x /= len(atoms_idx)
        y /= len(atoms_idx)
        z /= len(atoms_idx)

        new_atom = rdchem.Atom(0)
        editable_molecule = RWMol(mol)
        idx = editable_molecule.AddAtom(new_atom)
        mol = editable_molecule.GetMol()

        conf = mol.GetConformer(conf_id)
        conf.SetAtomPosition(idx, Point3D(x,y,z))

        return mol, idx

    @staticmethod
    def change_vector_direction(mol, 
                                X1:int, R_1:int, 
                                conf_id:int):
        """
        Changes the direction of the vector X1R-1. Only executed when f_group is "secondary amine":
            1. We have to change direction of X1R-1 to (-1) * X1R1
            2. X1 is in the middle between R(-1) and R1, so R1 = 2*X1 - (R-1)
        
        Args:
            mol: Rdkit optimized molecule.
            X1 (int): Atom index that connects functional group(NH2 or COOH) to the cycle of the molecule.
            R_1 (int): Atom index of the center of the functional group(NH2 or COOH).
            conf_id (int): Conformer id of the molecule. Defaults to -1.
            
        Returns:
            mol: Rdkit molecule with new virtual atom.
            idx (int): index of new virtual atom.
        """
        conf = mol.GetConformer(conf_id)

        Rx_1 = conf.GetAtomPosition(R_1)[0]
        Ry_1 = conf.GetAtomPosition(R_1)[1]
        Rz_1 = conf.GetAtomPosition(R_1)[2]

        Xx1 = conf.GetAtomPosition(X1)[0]
        Xy1 = conf.GetAtomPosition(X1)[1]
        Xz1 = conf.GetAtomPosition(X1)[2]
        
        Rx1 = 2 * Xx1 - Rx_1
        Ry1 = 2 * Xy1 - Ry_1
        Rz1 = 2 * Xz1 - Rz_1

        conf.SetAtomPosition(R_1, Point3D(Rx1,Ry1,Rz1))
        return mol, R_1
    
    @staticmethod
    def is_atom_in_cycle(mol, atom_id):
        """
        Checks if atom is in cycle.
        
        Args:
            mol: Rdkit optimized molecule.
            atom_id (int): Atom id to check if its in cycle.
            
        Returns:
            is_atom_in_ring (bool): True if atom is in ring and False if not.
        """
        atom = mol.GetAtomWithIdx(atom_id)

        is_atom_in_ring = atom.IsInRing()

        return is_atom_in_ring

    def find_X1X2R1R2(self):
        """
        Determines which atoms correspond to X1, X2, R1 and R2
            
        Returns:
            X1 (int): Atom id in cycle, that connects NH2 or COOH to the molecule.
            X2 (int): Atom id in cycle, that connects fluorine functional group to the molecule.
            R1 (int): NH2 or COOH functional group center atom id.
            R2 (int): fluorine functional group center atom id.
        """
        f_group_smiles = self.functional_group_to_smiles[self.f_group]

        carboxile_submol = Chem.MolFromSmiles('CC=O')
        nitro_amine_submol = Chem.MolFromSmiles('CN')

        carboxile_matches = self.mol.GetSubstructMatches(carboxile_submol)
        nitro_amine_matches = self.mol.GetSubstructMatches(nitro_amine_submol)

        X1, R1 = None, None
        if self.identificator == Identificator.carboxilic_acid:
            if len(carboxile_matches) == 0:
                raise "Problem with carboxile acid"
            
            X1 = carboxile_matches[0][0]
            R1 = carboxile_matches[0][1]

        print(self.identificator.name)
        if "amine" in self.identificator.name.lower():
            if len(nitro_amine_matches) == 0:
                raise "Problem with amine"
            
            if "primary" in self.identificator.lower():
                X1 = nitro_amine_matches[0][0]
                R1 = nitro_amine_matches[0][1]
            elif "secondary" in self.identificator.lower():
                X1 = nitro_amine_matches[0][1]
                self.mol, R_1 = OptimizedMolecule.set_average_atoms_position(self.mol, [nitro_amine_matches[0][0], nitro_amine_matches[1][0]], self.min_energy_conf_index)
                self.mol, R1 = OptimizedMolecule.change_vector_direction(self.mol, X1, R_1=R_1, conf_id=self.min_energy_conf_index)

        X2, R2 = None, None
        f_group_submol = Chem.MolFromSmiles(f_group_smiles)
        f_group_matches = self.mol.GetSubstructMatches(f_group_submol)
        if self.f_group.upper() in ['CF3', 'CHF2', 'CH2F']:
            X2 = f_group_matches[0][0]
            R2 = f_group_matches[0][1]

        elif self.f_group == 'gem-CF2':
            X2 = f_group_matches[0][0]
            self.mol, R2 = OptimizedMolecule.set_average_atoms_position(self.mol, [f_group_matches[0][1], f_group_matches[0][2]], self.min_energy_conf_index)

        elif self.f_group.upper() == 'CHF':
            if len(f_group_matches) == 1:
                X2 = f_group_matches[0][0]
                R2 = f_group_matches[0][1]
            elif len(f_group_matches) == 2:
                self.mol, X2 = OptimizedMolecule.set_average_atoms_position(self.mol, [f_group_matches[0][0], f_group_matches[1][0]], self.min_energy_conf_index)
                self.mol, R2 = OptimizedMolecule.set_average_atoms_position(self.mol, [f_group_matches[0][1], f_group_matches[1][1]], self.min_energy_conf_index)

        if len(set([X1, X2, R1, R2])) != 4:
            X1, X2, R1, R2 = None, None, None, None
        
        if not OptimizedMolecule.is_atom_in_cycle(mol=self.mol, atom_id=f_group_matches[0][0]):
            # print(f"X1: {X1} or X2: {X2} is not in the cycle, smiles: {self.smiles}")
            X1, X2, R1, R2 = None, None, None, None

        return X1, X2, R1, R2

    def optimize_geometry(self):
        """Executes geometric optimization of the molecule"""
        self.mol = OptimizedMolecule.prepare_molecule(self.smiles)
        self.min_energy_conf_index, self.min_energy, self.mol = OptimizedMolecule.find_conf_with_min_energy(self.mol)


class DipoleMoment(OptimizedMolecule):
    """
    Class for calculating whole dipole moment of the molecule.

    This class inherits from OptimizedMolecule and provides functionality to calculate dipole moment 
    based on atoms positions and charges using optimized molecule with lowest energy conformer.

    Attributes:
        smiles (str): String representation of a molecule.

    Methods:
        description(): Description of the class.
        dependencies(): Dependencies needed for the MoleculeVolume class.
        calculate(): Calculate the dipole moment of the molecule.
    """
    def __init__(self, smiles):
        super().__init__(smiles)

    def description(self):
        """Returns description of the class"""
        return "Whole dipole moment of the optimized molecule with lowest energy conformer"
    
    def dependencies(self):
        """Returns a list of dependencies needed for the DipoleMoment class."""
        return {"mol": OptimizedMolecule()}
    
    def calculate(self):
        """
        Calculate the dipole moment.

        Returns:
            dipole_moment (float): Float value with calculated dipole_moment
        """
        charges = []
        coordinates = []
        x_centroid, y_centroid, z_centroid = 0, 0, 0
        for atom in self.mol.GetAtoms():
            pos = self.mol.GetConformer(self.min_energy_conf_index).GetAtomPosition(atom.GetIdx())
            charge = atom.GetDoubleProp("_GasteigerCharge")

            charges.append(charge)
            coordinates.append([pos[0], pos[1], pos[2]])

            x_centroid += pos[0]
            y_centroid += pos[1]
            z_centroid += pos[2]

        x_centroid /= len(self.mol.GetAtoms())
        y_centroid /= len(self.mol.GetAtoms())
        z_centroid /= len(self.mol.GetAtoms())

        charges_multiply_coordinates = coordinates.copy()
        for charges_multiply_coordinate_index in range(len(charges_multiply_coordinates)):
            charges_multiply_coordinates[charges_multiply_coordinate_index][0] = \
                (charges_multiply_coordinates[charges_multiply_coordinate_index][0] - x_centroid) \
                * charges[charges_multiply_coordinate_index]
            charges_multiply_coordinates[charges_multiply_coordinate_index][1] = \
                (charges_multiply_coordinates[charges_multiply_coordinate_index][1] - y_centroid) \
                * charges[charges_multiply_coordinate_index]
            charges_multiply_coordinates[charges_multiply_coordinate_index][2] = \
                (charges_multiply_coordinates[charges_multiply_coordinate_index][2] - z_centroid) \
                * charges[charges_multiply_coordinate_index]

        dipole_moment_vector = [0, 0, 0]
        for charges_multiply_coordinate_index in range(len(charges_multiply_coordinates)):
            dipole_moment_vector[0] += charges_multiply_coordinates[charges_multiply_coordinate_index][0]
            dipole_moment_vector[1] += charges_multiply_coordinates[charges_multiply_coordinate_index][1]
            dipole_moment_vector[2] += charges_multiply_coordinates[charges_multiply_coordinate_index][2]

        dipole_moment = math.sqrt(pow(dipole_moment_vector[0], 2) 
                                  + pow(dipole_moment_vector[1], 2)
                                  + pow(dipole_moment_vector[2], 2))

        return dipole_moment


class MoleculeVolume(OptimizedMolecule):
    """
    Class for calculating molecule Volume.

    This class inherits from OptimizedMolecule and provides functionality to calculate 
    molecule volume using optimized molecule with lowest energy conformer and rdkit.

    Attributes:
        smiles (str): String representation of a molecule.

    Methods:
        description(): Description of the class.
        dependencies(): Dependencies needed for the MoleculeVolume class.
        calculate(): Calculate the molecule volume.
    """
    def __init__(self, smiles):
        super().__init__(smiles)

    def description(self):
        """Returns description of the class"""
        return "Molecule volume on optimized molecule with lowest energy conformer"
    
    def dependencies(self):
        """Returns a list of dependencies needed for the MoleculeVolume class."""
        return {"mol": OptimizedMolecule()}
    
    def calculate(self):
        """
        Calculate the molecule Volume.

        Returns:
            mol_volume (float): Float value with calculated molecule volume
        """
        mol_volume = AllChem.ComputeMolVolume(mol=self.mol,
                                              confId=self.min_energy_conf_index)
        
        return mol_volume


class MoleculeSASA(OptimizedMolecule):
    """
    Class for calculating molecule solvent accessible surface area.

    This class inherits from OptimizedMolecule and provides functionality to calculate 
    molecule solvent accessible surface area using optimized molecule with lowest energy conformer and rdkit.

    Attributes:
        smiles (str): String representation of a molecule.

    Methods:
        description(): Description of the class.
        dependencies(): Dependencies needed for the MoleculeSASA class.
        calculate(): Calculate the molecule solvent accessible surface area.
    """
    def __init__(self, smiles):
        super().__init__(smiles)

    def description(self):
        """Returns description of the class"""
        return "Molecule free solvent access area on optimized molecule with lowest energy conformer"
    
    def dependencies(self):
        """Returns a list of dependencies needed for the MoleculeSASA class."""
        return {"mol": OptimizedMolecule()}
    
    def calculate(self):
        """
        Calculate the molecule solvent accessible surface area.

        Returns:
            sasa (float): Float value with calculated molecule solvent accessible surface area.
        """
        mol_classify = rdFreeSASA.classifyAtoms(self.mol)
        sasa = rdFreeSASA.CalcSASA(mol=self.mol, 
                                   radii=mol_classify, 
                                   confIdx=self.min_energy_conf_index)
        
        return sasa


class MoleculeTPSAF(OptimizedMolecule):
    """
    Class for calculating molecule topological polar surface area.

    This class inherits from OptimizedMolecule and provides functionality to calculate 
    molecule topological polar surface area with additional fluorine area using optimized molecule 
    with lowest energy conformer and rdkit.

    Attributes:
        smiles (str): String representation of a molecule.

    Methods:
        description(): Description of the class.
        dependencies(): Dependencies needed for the MoleculeTPSAF class.
        calculate(): Calculate the molecule topological polar surface area with additional fluorine area.
    """
    def __init__(self, smiles):
        super().__init__(smiles)

    def description(self):
        """Returns description of the class"""
        return "Topological polar surface area with additional Fluor surface area"
    
    def dependencies(self):
        """Returns a list of dependencies needed for the MoleculeTPSAF class."""
        return {"mol": OptimizedMolecule()}
    
    def calculate(self):
        """
        Calculate the molecule topological polar surface area with additional fluorine area.

        Returns:
            tpsa_f (float): Float value with calculated molecule topological polar surface area 
            with additional fluorine area.
        """
        tpsa = Descriptors.TPSA(self.mol)
        fluor_idxs = [atom.GetIdx() for atom in self.mol.GetAtoms() if atom.GetSymbol().lower() == 'f']
        tpsa_f = tpsa

        radii = rdFreeSASA.classifyAtoms(self.mol)
        rdFreeSASA.CalcSASA(self.mol, radii)
        
        for fluor_idx in fluor_idxs:
            atom_sasa = self.mol.GetAtoms()[fluor_idx].GetProp('SASA')
            tpsa_f += float(atom_sasa)

        return tpsa_f


class MoleculeDihedralAngle(OptimizedMolecule):
    """
    Class for calculating dihedral angle between X1, R1, R2 and X2.

    This class inherits from OptimizedMolecule and provides functionality to calculate 
    dihedral angle using optimized molecule with lowest energy conformer and rdkit.

    Attributes:
        smiles (str): String representation of a molecule.
        f_group (str): Fluor functional group name. Defaults to None.
        identificator (Identificator): The molecule type. Defaults to None.

    Methods:
        description(): Description of the class.
        dependencies(): Dependencies needed for the MoleculeDihedralAngle class.
        calculate(): Calculate the dihedral angle.
    """
    def __init__(self, smiles, f_group, identificator):
        super().__init__(smiles, f_group, identificator)
        self.feature_name = "dihedral_angle"

    def description(self):
        """Returns description of the class"""
        return "Molecule free solvent access area on optimized molecule with lowest energy conformer"
    
    def dependencies(self):
        """Returns a list of dependencies needed for the MoleculeDihedralAngle class."""
        return {"mol": OptimizedMolecule()}
    
    @staticmethod
    def _dihedral_angle(mol, 
                       iAtomId:int, jAtomId:int, kAtomId:int, lAtomId:int, 
                       conf_id:int):
        """
        Calculate dihedral angle in molecule between IJK and JKL flat areas with specified conformer id.
        
        Args:
            mol: Rdkit optimized molecule.
            iAtomId (int): AtomI index in molecule.
            jAtomId (int): AtomJ index in molecule.
            kAtomId (int): AtomK index in molecule.
            lAtomId (int): AtomL index in molecule.
            conf_id: Conformer id of the molecule.
            
        Returns:
            dihedral_angle_value (float): dihedral angle between areas.
        """
        
        conf = mol.GetConformer(conf_id)

        dihedral_angle_value = abs(rdMolTransforms.GetDihedralDeg(conf, iAtomId, jAtomId, kAtomId, lAtomId))

        return dihedral_angle_value
    
    def calculate(self):
        """
        Calculate dihedral angle in molecule between IJK and JKL flat areas.

        Returns:
            dihedral_angle_value (float): dihedral angle between areas R2X2X1 and X2X1R1.
        
        Raises:
            ValueError: If the f_group or identificator is None.
            InvalidMoleculeTypeError: If we cannot calculate dihedral angle for this molecule.
        """
        if self.f_group is None or self.identificator is None:
            raise ValueError("fluor functional group or molecule identificator NotFound")

        if len(set([self.X1, self.X2, self.R1, self.R2])) != 4:
            raise InvalidMoleculeTypeError(self.X1, self.X2, self.R1, self.R2, self.feature_name)

        dihedral_angle_value = MoleculeDihedralAngle._dihedral_angle(self.mol, 
                                                            self.R2, self.X2, self.X1, self.R1, 
                                                            self.min_energy_conf_index) 
        return dihedral_angle_value


class MoleculeDistance(OptimizedMolecule):
    """
    A class for calculating distances between atoms in a molecule.

    This class extends the OptimizedMolecule class to provide methods for calculating
    distances between atoms in a molecule.

    Attributes:
        Inherits attributes from OptimizedMolecule.

    Methods:
        _distance_between_atoms(): Calculates the distance between two atoms.
    """
    def __init__(self, smiles, f_group, identificator):
        """
        Initialize the MoleculeDistance object.

        Args:
            smiles (str): The SMILES string representing the molecule.
            f_group (str): The functional group of the molecule.
            identificator (str): The identifier of the molecule.
        """
        super().__init__(smiles, f_group, identificator)
    
    @staticmethod
    def _distance_between_atoms(iAtom_pos, jAtom_pos):
        """
        Calculate distance between two atoms.
        
        Args:
            iAtom_pos (tuple(int, int, int)): AtomI coordinates.
            jAtom_pos (tuple(int, int, int)): AtomJ coordinates.
            
        Returns:
            distance (float): distance between two points.
        """
        vector = (jAtom_pos[0] - iAtom_pos[0], jAtom_pos[1] - iAtom_pos[1], jAtom_pos[2] - iAtom_pos[2])

        distance = math.sqrt(pow(vector[0], 2) + pow(vector[1], 2) + pow(vector[2], 2))

        return distance


class DistanceBetweenX1X2(MoleculeDistance):
    """
    Class for calculating distance between X1 and X2 atoms in cycles.

    This class inherits from MoleculeDistance and provides functionality to calculate 
    distance using optimized molecule with lowest energy conformer and rdkit.

    Attributes:
        smiles (str): String representation of a molecule.
        f_group (str): Fluor functional group name. Defaults to None.
        identificator (Identificator): The molecule type. Defaults to None.

    Methods:
        description(): Description of the class.
        dependencies(): Dependencies needed for the DistanceBetweenX1X2 class.
        calculate(): Calculate the distance between X1 and X2.
    """
    def __init__(self, smiles, f_group, identificator):
        super().__init__(smiles, f_group, identificator)
        self.feature_name = "distance_between_atoms_in_cycle_and_f_group"

    def description(self):
        """Returns description of the class"""
        return "Distance between atoms, which connect functional groups to the main part of the molecule"
    
    def dependencies(self):
        """Returns a list of dependencies needed for the DistanceBetweenX1X2 class."""
        return {"mol": OptimizedMolecule(), "_distance_between_atoms": MoleculeDistance()}

    def calculate(self):
        """
        Calculate distance in molecule between X1 and X2.

        Returns:
            r_distance (float): dihedral between X1 and X2 atoms.
        
        Raises:
            ValueError: If the f_group or identificator is None.
            InvalidMoleculeTypeError: If we cannot calculate distance for this molecule.
        """
        if self.f_group is None or self.identificator is None:
            raise ValueError("fluor functional group or molecule identificator NotFound")
        
        if len(set([self.X1, self.X2, self.R1, self.R2])) != 4:
            raise InvalidMoleculeTypeError(self.X1, self.X2, self.R1, self.R2, self.feature_name)

        X1_pos = self.mol.GetConformer(self.min_energy_conf_index).GetAtomPosition(self.X1)
        X2_pos = self.mol.GetConformer(self.min_energy_conf_index).GetAtomPosition(self.X2)

        r_distance = DistanceBetweenX1X2._distance_between_atoms(X1_pos, X2_pos)

        return r_distance


class DistanceBetweenR1R2(MoleculeDistance):
    """
    Class for calculating distance between R1 and R2 atoms in functional groups centers.

    This class inherits from MoleculeDistance and provides functionality to calculate 
    distance using optimized molecule with lowest energy conformer and rdkit.

    Attributes:
        smiles (str): String representation of a molecule.
        f_group (str): Fluor functional group name. Defaults to None.
        identificator (Identificator): The molecule type. Defaults to None.

    Methods:
        description(): Description of the class.
        dependencies(): Dependencies needed for the DistanceBetweenR1R2 class.
        calculate(): Calculate the distance between R1 and R2.
    """
    def __init__(self, smiles, f_group, identificator):
        super().__init__(smiles, f_group, identificator)
        self.feature_name = "distance_between_atoms_in_f_group_centers"

    def description(self):
        """Returns description of the class"""
        return "Distance between atoms in the centers of functional groups"
    
    def dependencies(self):
        """Returns a list of dependencies needed for the DistanceBetweenR1R2 class."""
        return {"mol": OptimizedMolecule(), "_distance_between_atoms": MoleculeDistance()}

    def calculate(self):
        """
        Calculate distance in molecule between R1 and R2.

        Returns:
            R_distance (float): distance between R1 and R2 atoms.
        
        Raises:
            ValueError: If the f_group or identificator is None.
            InvalidMoleculeTypeError: If we cannot calculate distance for this molecule.
        """
        if self.f_group is None or self.identificator is None:
            raise ValueError("fluor functional group or molecule identificator NotFound")
        
        if len(set([self.X1, self.X2, self.R1, self.R2])) != 4:
            raise InvalidMoleculeTypeError(self.X1, self.X2, self.R1, self.R2, self.feature_name)

        R1_pos = self.mol.GetConformer(self.min_energy_conf_index).GetAtomPosition(self.R1)
        R2_pos = self.mol.GetConformer(self.min_energy_conf_index).GetAtomPosition(self.R2)

        R_distance = DistanceBetweenR1R2._distance_between_atoms(R1_pos, R2_pos)

        return R_distance


class AngleX1X2R2(OptimizedMolecule):
    """
    Class for calculating flat angle between X1, X2 and R2 atoms.

    This class inherits from OptimizedMolecule and provides functionality to calculate 
    flat angle using optimized molecule with lowest energy conformer and rdkit.

    Attributes:
        smiles (str): String representation of a molecule.
        f_group (str): Fluor functional group name. Defaults to None.
        identificator (Identificator): The molecule type. Defaults to None.

    Methods:
        description(): Description of the class.
        dependencies(): Dependencies needed for the AngleX1X2R2 class.
        _flat_angle(): Calculate flat angle between 3 atoms.
        calculate(): Calculate the angle between X1, X2 and R2 points.
    """
    def __init__(self, smiles, f_group, identificator):
        super().__init__(smiles, f_group, identificator)
        self.feature_name = "angle_X1X2R2"

    def description(self):
        """Returns description of the class"""
        return "Flat angle between X1, X2 and R2"
    
    def dependencies(self):
        """Returns a list of dependencies needed for the AngleX1X2R2 class."""
        return {"mol": OptimizedMolecule()}
    
    @staticmethod
    def _flat_angle(mol, 
                    iAtomId:int, jAtomId:int, kAtomId:int, 
                    conf_id:int):
        """
        Calculate flat angle between I, J and K atoms in molecule.
        
        Args:
            iAtomId (int): AtomI coordinates.
            jAtomId (int): AtomJ coordinates.
            kAtomId (int): AtomK coordinates.
            
        Returns:
            (float): IJK Angle.
        """
        conf = mol.GetConformer(conf_id)

        return rdMolTransforms.GetAngleDeg(conf, iAtomId, jAtomId, kAtomId)


    def calculate(self):
        """
        Calculate angle in molecule between X1, X2 and R2.

        Returns:
            angle_X1X2R2 (float): angle between X1, X2 and R2 atoms.
        
        Raises:
            ValueError: If the f_group or identificator is None.
            InvalidMoleculeTypeError: If we cannot calculate angle for this molecule.
        """
        if self.f_group is None or self.identificator is None:
            raise ValueError("fluor functional group or molecule identificator NotFound")
        
        if len(set([self.X1, self.X2, self.R1, self.R2])) != 4:
            raise InvalidMoleculeTypeError(self.X1, self.X2, self.R1, self.R2, self.feature_name)

        angle_X1X2R2 = AngleX1X2R2._flat_angle(self.mol, self.X1, self.X2, self.R2, self.min_energy_conf_index)

        return angle_X1X2R2


class AngleX2X1R1(OptimizedMolecule):
    """
    Class for calculating flat angle between X2, X1 and R1 atoms.

    This class inherits from OptimizedMolecule and provides functionality to calculate 
    flat angle using optimized molecule with lowest energy conformer and rdkit.

    Attributes:
        smiles (str): String representation of a molecule.
        f_group (str): Fluor functional group name. Defaults to None.
        identificator (Identificator): The molecule type. Defaults to None.

    Methods:
        description(): Description of the class.
        dependencies(): Dependencies needed for the AngleX2X1R1 class.
        _flat_angle(): Calculate flat angle between 3 atoms.
        calculate(): Calculate the angle between X2, X1 and R1 points.
    """
    def __init__(self, smiles, f_group, identificator):
        super().__init__(smiles, f_group, identificator)
        self.feature_name = "angle_X2X1R1"

    def description(self):
        """Returns description of the class"""
        return "Flat angle between X2, X1 and R1"
    
    def dependencies(self):
        """Returns a list of dependencies needed for the AngleX2X1R1 class."""
        return {"mol": OptimizedMolecule()}
    
    @staticmethod
    def _flat_angle(mol, 
                    iAtomId:int, jAtomId:int, kAtomId:int, 
                    conf_id:int):
        """
        Calculate flat angle between I, J and K atoms in molecule.
        
        Args:
            iAtomId (int): AtomI coordinates.
            jAtomId (int): AtomJ coordinates.
            kAtomId (int): AtomK coordinates.
            
        Returns:
            (float): IJK Angle.
        """
        conf = mol.GetConformer(conf_id)

        return rdMolTransforms.GetAngleDeg(conf, iAtomId, jAtomId, kAtomId)


    def calculate(self):
        """
        Calculate angle in molecule between X2, X1 and R1.

        Returns:
            angle_X2X1R1 (float): angle between X2, X1 and R1 atoms.
        
        Raises:
            ValueError: If the f_group or identificator is None.
            InvalidMoleculeTypeError: If we cannot calculate angle for this molecule.
        """
        if self.f_group is None or self.identificator is None:
            raise ValueError("fluor functional group or molecule identificator NotFound")
        
        if len(set([self.X1, self.X2, self.R1, self.R2])) != 4:
            raise InvalidMoleculeTypeError(self.X1, self.X2, self.R1, self.R2, self.feature_name)

        angle_X2X1R1 = AngleX2X1R1._flat_angle(self.mol, self.X2, self.X1, self.R1, self.min_energy_conf_index)

        return angle_X2X1R1


class AngleR2X2R1(OptimizedMolecule):
    """
    Class for calculating flat angle between R2, X2 and R1 atoms.

    This class inherits from OptimizedMolecule and provides functionality to calculate 
    flat angle using optimized molecule with lowest energy conformer and rdkit.

    Attributes:
        smiles (str): String representation of a molecule.
        f_group (str): Fluor functional group name. Defaults to None.
        identificator (Identificator): The molecule type. Defaults to None.

    Methods:
        description(): Description of the class.
        dependencies(): Dependencies needed for the AngleR2X2R1 class.
        _flat_angle(): Calculate flat angle between 3 atoms.
        calculate(): Calculate the angle between R2, X2 and R1 points.
    """
    def __init__(self, smiles, f_group, identificator):
        super().__init__(smiles, f_group, identificator)
        self.feature_name = "angle_R2X2R1"

    def description(self):
        """Returns description of the class"""
        return "Flat angle between R2, X2 and R1"
    
    def dependencies(self):
        """Returns a list of dependencies needed for the AngleR2X2R1 class."""
        return {"mol": OptimizedMolecule()}
    
    @staticmethod
    def _flat_angle(mol, 
                    iAtomId:int, jAtomId:int, kAtomId:int, 
                    conf_id:int):
        """
        Calculate flat angle between I, J and K atoms in molecule.
        
        Args:
            iAtomId (int): AtomI coordinates.
            jAtomId (int): AtomJ coordinates.
            kAtomId (int): AtomK coordinates.
            
        Returns:
            (float): IJK Angle.
        """
        conf = mol.GetConformer(conf_id)

        return rdMolTransforms.GetAngleDeg(conf, iAtomId, jAtomId, kAtomId)


    def calculate(self):
        """
        Calculate angle in molecule between R2, X2 and R1.

        Returns:
            angle_X2X1R1 (float): angle between R2, X2 and R1 atoms.
        
        Raises:
            ValueError: If the f_group or identificator is None.
            InvalidMoleculeTypeError: If we cannot calculate angle for this molecule.
        """
        if self.f_group is None or self.identificator is None:
            raise ValueError("fluor functional group or molecule identificator NotFound")
        
        if len(set([self.X1, self.X2, self.R1, self.R2])) != 4:
            raise InvalidMoleculeTypeError(self.X1, self.X2, self.R1, self.R2, self.feature_name)

        angle_R2X2R1 = AngleR2X2R1._flat_angle(self.mol, self.R2, self.X2, self.R1, self.min_energy_conf_index)

        return angle_R2X2R1


class AngleR1X1R2(OptimizedMolecule):
    """
    Class for calculating flat angle between R1, X1 and R2 atoms.

    This class inherits from OptimizedMolecule and provides functionality to calculate 
    flat angle using optimized molecule with lowest energy conformer and rdkit.

    Attributes:
        smiles (str): String representation of a molecule.
        f_group (str): Fluor functional group name. Defaults to None.
        identificator (Identificator): The molecule type. Defaults to None.

    Methods:
        description(): Description of the class.
        dependencies(): Dependencies needed for the AngleR1X1R2 class.
        _flat_angle(): Calculate flat angle between 3 atoms.
        calculate(): Calculate the angle between R1, X1 and R2 points.
    """
    def __init__(self, smiles, f_group, identificator):
        super().__init__(smiles, f_group, identificator)
        self.feature_name = "angle_R1X1R2"

    def description(self):
        """Returns description of the class"""
        return "Flat angle between R1, X1 and R2"
    
    def dependencies(self):
        """Returns a list of dependencies needed for the AngleR1X1R2 class."""
        return {"mol": OptimizedMolecule()}
    
    @staticmethod
    def _flat_angle(mol, 
                    iAtomId:int, jAtomId:int, kAtomId:int, 
                    conf_id:int):
        """
        Calculate flat angle between I, J and K atoms in molecule.
        
        Args:
            iAtomId (int): AtomI coordinates.
            jAtomId (int): AtomJ coordinates.
            kAtomId (int): AtomK coordinates.
            
        Returns:
            (float): IJK Angle.
        """
        conf = mol.GetConformer(conf_id)

        return rdMolTransforms.GetAngleDeg(conf, iAtomId, jAtomId, kAtomId)


    def calculate(self):
        """
        Calculate angle in molecule between R1, X1 and R2.

        Returns:
            angle_R1X1R2 (float): angle between R1, X1 and R2 atoms.
        
        Raises:
            ValueError: If the f_group or identificator is None.
            InvalidMoleculeTypeError: If we cannot calculate angle for this molecule.
        """
        if len(set([self.X1, self.X2, self.R1, self.R2])) != 4:
            raise InvalidMoleculeTypeError(self.X1, self.X2, self.R1, self.R2, self.feature_name)

        angle_R1X1R2 = AngleR2X2R1._flat_angle(self.mol, self.R1, self.X1, self.R2, self.min_energy_conf_index)

        return angle_R1X1R2


class Molecule2D:
    """
    Class that represents a 3D molecule with single conformer.

    This class provides functionality to convert smiles to rdkit molecule, 

    Attributes:
        smiles (str): String representation of a molecule.

    Methods:
        _prepare_molecule(): Creates 3D molecule with single conformers from smiles.
    """
    def __init__(self,
                 smiles) -> None:
        
        self.smiles = smiles

        self.mol = Molecule2D._prepare_molecule(SMILES)
    
    @staticmethod
    def _prepare_molecule(SMILES):
        """
        Create rdkit 3d molecule from SMILES with conformer.
        
        Args:
            SMILES (str): String representation of a molecule. 
            
        Returns:
            mol: Rdkit 3D molecule with conformer.
        """
        mol = Chem.MolFromSmiles(SMILES)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)

        return mol


class MoleculeRingsAmount(Molecule2D):
    """
    Class for calculating amount of rings in the molecule.

    This class inherits from Molecule2D and provides functionality 
    to calculate amount of rings in the molecule 

    Attributes:
        smiles (str): String representation of a molecule.

    Methods:
        description(): Description of the class.
        dependencies(): Dependencies needed for the MoleculeVolume class.
        _cycles_amount(): Get amount of rings in the molecule.
        calculate(): Get ring number in the molecule.
    """
    def __init__(self, smiles):
        super().__init__(smiles)
        self.feature_name = "mol_num_cycles"

    def description(self):
        """Returns description of the class"""
        return "Molecule's amount of cycles"
    
    def dependencies(self):
        """Returns a list of dependencies needed for the MoleculeRingsAmount class."""
        return {"mol": Molecule2D(), "num_rings": cycles_amount()}

    @staticmethod
    def _cycles_amount(mol):
        """
        Calculate amount of cycles in the molecule.
        
        Args:
            mol: rdkit molecule.
            
        Returns:
            num_rings (int): amount of rings.
        """
        sssr = Chem.GetSSSR(mol)

        num_rings = len(sssr)
        
        return num_rings
    
    def calculate(self):
        """
        Calculate amount of cycles in the molecule.
            
        Returns:
            num_rings (int): amount of rings.
        """
        num_rings = MoleculeRingsAmount._cycles_amount(self.mol)
        
        return num_rings


class AtomsToRingRatio(Molecule2D):
    """
    Class for calculating ratio of amount of atoms in rings to rings amount.

    This class inherits from Molecule2D and provides functionality 
    to calculate amount of atoms in rings / amount of rings. 

    Attributes:
        smiles (str): String representation of a molecule.

    Methods:
        description(): Description of the class.
        dependencies(): Dependencies needed for the MoleculeVolume class.
        _cycles_amount(): Get amount of rings in the molecule.
        _atoms_in_cycles_amount(): Get amount of atoms in cycles in the molecule.
        calculate(): Get ring number in the molecule.
    """
    def __init__(self, smiles):
        super().__init__(smiles)
        self.feature_name = "avg_atoms_in_cycle"

    def description(self):
        """Returns description of the class"""
        return "Molecule's amount of atoms in cycle divide by amount of cycles"
    
    def dependencies(self):
        """Returns a list of dependencies needed for the AtomsToRingRatio class."""
        return {"mol": Molecule2D()}
    
    @staticmethod
    def _cycles_amount(mol):
        """
        Calculate amount of cycles in the molecule.
        
        Args:
            mol: rdkit molecule.
            
        Returns:
            amount_of_cycles (int): amount of rings.
        """
        sssr = Chem.GetSSSR(mol)
        
        amount_of_cycles = len(sssr)
        
        return amount_of_cycles
    
    @staticmethod
    def _atoms_in_cycles_amount(mol):
        """
        Calculate amount of atoms that are in cycle in molecule.
        
        Args:
            mol: rdkit molecule.
            
        Returns:
            atoms_num_in_cycles (int): amount of atoms in cycle in the molecule.
        """
        sssr = Chem.GetSSSR(mol)
        
        atoms_idxs = set()

        for i, ring in enumerate(sssr):
            for atom_index in ring:
                atoms_idxs.add(atom_index)
        
        atoms_num_in_cycles = len(atoms_idxs)

        return atoms_num_in_cycles
    
    def calculate(self):
        """
        Calculate ratio amount of atoms in rings to rings amount.
            
        Returns:
            (float): atoms_num_in_cycles / amount_cycles.
        """
        amount_cycles = AtomsToRingRatio._cycles_amount(self.mol)
        if amount_cycles == 0:
            return 0
        
        atoms_num_in_cycles = AtomsToRingRatio._atoms_in_cycles_amount(self.mol)
        
        return atoms_num_in_cycles / amount_cycles


class Chirality(Molecule2D):
    """
    Class for calculating amount of chiral centers in molecule.

    This class inherits from Molecule2D and provides functionality 
    to calculate amount of chiral centers. 

    Attributes:
        smiles (str): String representation of a molecule.

    Methods:
        description(): Description of the class.
        dependencies(): Dependencies needed for the MoleculeVolume class.
        _amount_of_chiral_centers(): Get amount of chiral centers in the molecule.
        calculate(): Get chirals centers amount..
    """
    def __init__(self, smiles):
        super().__init__(smiles)
        self.feature_name = "chirality"

    def description(self):
        """Returns description of the class"""
        return "Amount of chiral centers in the molecule"
    
    def dependencies(self):
        """Returns a list of dependencies needed for the Chirality class."""
        return {"mol": Molecule2D()}
    
    @staticmethod
    def _amount_of_chiral_centers(mol):
        """
        Calculate amount of chiral centers in molecule.
        
        Args:
            mol: rdkit molecule.
            
        Returns:
            amount_of_chiral_centers (int): amount of chiral centers in the molecule.
        """
        Chem.AssignAtomChiralTagsFromStructure(mol)
        
        chirality_centers = Chem.FindMolChiralCenters(mol)

        amount_of_chiral_centers = len(chirality_centers)
        
        return amount_of_chiral_centers
    
    def calculate(self):
        """
        Calculate amount of chiral centers in molecule.
            
        Returns:
            amount_of_chiral_centers (int): amount of chiral centers in the molecule.
        """
        amount_of_chiral_centers = Chirality._amount_of_chiral_centers(self.mol)
        
        return amount_of_chiral_centers


if __name__ == "__main__":
    SMILES = "FC1(F)CCC(C(O)=O)CC1"
    identificator = Identificator.carboxilic_acid
    f_group = "gem-CF2"
    feature = DistanceBetweenX1X2(smiles=SMILES,
                                  f_group=f_group,
                                  identificator=identificator)
    print(feature.calculate())
