import math
import numpy as np
from collections import deque

from rdkit import Chem
from rdkit.Chem import rdchem, rdMolTransforms, rdForceFieldHelpers, rdPartialCharges, rdFreeSASA
from rdkit.Chem import AllChem, Descriptors
from rdkit.Geometry import Point3D
from rdkit.Chem.rdchem import RWMol

from fluoriclogppka.ml_part.constants import Identificator, Target
from fluoriclogppka.ml_part.constants import ALL_SUBMOLS, FUNCTIONAL_GROUP_TO_SMILES
import fluoriclogppka.ml_part.services.utils as utils
import fluoriclogppka.ml_part.services.utils_pKa as utils_pKa
import fluoriclogppka.ml_part.services.utils_logP as utils_logP

class Molecule3DFeaturesService:
    """
    Class that represents a 3D molecule features.

    This class provides functionality to obtain 3d molecules features. 
    """
    def __init__(self, 
                 smiles: str,
                 target_value: Target,
                 conformers_limit: int = None) -> None:

        self.target_value = target_value
        self.mol_2d = Chem.MolFromSmiles(smiles)
        self.smiles = smiles
        self.mol = Molecule3DFeaturesService.prepare_molecule(smiles=smiles,
                                                              conformers_limit=conformers_limit)
        self.min_energy_conf_index, self.min_energy, self.mol = Molecule3DFeaturesService.find_conf_with_min_energy(self.mol)
        self.mol_optimized = self.mol

        self.f_group = self.calculate_fluoric_group()
        self.f_freedom = self.calculate_f_group_freedom()

        self.identificator = self.calculate_identificator()
        self.f_to_fg = self.calculate_linear_path_f_to_fg()
        
        self.dipole_moment = self.calculate_dipole_moment()
        self.mol_volume = self.calculate_volume()
        self.sasa = self.calculate_sasa()
        self.molecular_weight = self.calculate_molecular_weight()

        self.tpsa_with_fluor = self.calculate_TPSA_with_fluor()

        if self.f_group is not None and self.identificator is not None:
            self.X1, self.X2, self.R1, self.R2 = self.find_X1X2R1R2()
            self.dihedral_angle_value = self.calculate_dihedral_angle()
            
            self.distance_between_atoms_in_cycle = self.calculate_distance_between_atoms_in_cycle()
            self.distance_between_atoms_in_f_group_centers = self.calculate_distance_between_atoms_in_f_group_centers()

            self.flat_angle_between_atoms_in_cycle_1, self.flat_angle_between_atoms_in_cycle_2 = self.calculate_flat_angle_between_atoms_in_cycle()
            self.flat_angle_between_atoms_in_f_group_center_1, self.flat_angle_between_atoms_in_f_group_center_2 = self.calculate_flat_angle_between_atoms_in_f_group_center()

            self.cis_trans = self.calculate_cis_trans()

        self.features_3d_dict = {
            "identificator": self.identificator,
            "dipole_moment": self.dipole_moment,
            "mol_volume": self.mol_volume,
            "mol_weight": self.molecular_weight,
            "f_to_fg": self.f_to_fg,
            "sasa": self.sasa,

            "f_freedom": self.f_freedom,

            "cis/trans": self.cis_trans,
            "dihedral_angle": self.dihedral_angle_value,

            "distance_between_atoms_in_cycle_and_f_group": self.distance_between_atoms_in_cycle,
            "distance_between_atoms_in_f_group_centers": self.distance_between_atoms_in_f_group_centers,

            "angle_X1X2R2": self.flat_angle_between_atoms_in_cycle_1,
            "angle_X2X1R1": self.flat_angle_between_atoms_in_cycle_2,

            "angle_R2X2R1": self.flat_angle_between_atoms_in_f_group_center_1,
            "angle_R1X1R2": self.flat_angle_between_atoms_in_f_group_center_2,

            "cis/trans": self.cis_trans,

            "tpsa+f": self.tpsa_with_fluor
        }

    @staticmethod
    def prepare_molecule(smiles: str,
                         conformers_limit: int = None):
        """
        Create rdkit 3d molecule from SMILES.
        Generate charges and molecule's conformers.
        
        Args:
            smiles (str): String representation of a molecule. 
            conformers_limit (int): Max number of generated conformers for optimization.
            
        Returns:
            mol: Rdkit sanitized molecule with generated charges and multiple conformers.
        """
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        rdForceFieldHelpers.MMFFSanitizeMolecule(mol)
        
        if conformers_limit is not None:
            number_of_confs = conformers_limit
        else:
            num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            number_of_confs = pow(3, num_rotatable_bonds + 3)
        
        AllChem.EmbedMultipleConfs(mol, numConfs=number_of_confs, randomSeed=3407)
        rdPartialCharges.ComputeGasteigerCharges(mol)

        return mol

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
    def _amount_of_hydrogen_in_neighbors(mol, atom_idx):
        """
        Calculate the number of hydrogen atoms in the neighbors of a given atom.

        Args:
            mol (rdchem.Mol): RDKit molecule.
            atom_idx (int): Index of the target atom.

        Returns:
            int: Number of hydrogen atoms in the neighboring atoms.
        """
        atom = mol.GetAtomWithIdx(atom_idx)

        neighbors = atom.GetNeighbors()
        
        amount_of_hydrogen = 0
        
        for neighbor in neighbors:
            if neighbor.GetSymbol() == 'H':
                amount_of_hydrogen += 1

        return amount_of_hydrogen

    @staticmethod
    def _check_gem_CF2(mol):
        """
        Check if the molecule contains a geminal difluoromethylene group (gem-CF2) 
            based on amount of hydrogen near "C" atom. If more that 0, it's not gem-CF2.

        Args:
            mol (rdchem.Mol): RDKit molecule.

        Returns:
            bool: True if gem-CF2 is present, False otherwise.
        """
        f_group_submol = Chem.MolFromSmiles("C(F)(F)")
        f_group_matches = mol.GetSubstructMatches(f_group_submol)

        if len(f_group_matches) == 0:
            return False
        
        c_atom_index = f_group_matches[0][0]

        amount_of_hydrogen = Molecule3DFeaturesService._amount_of_hydrogen_in_neighbors(mol, c_atom_index)

        if amount_of_hydrogen > 0:
            return False
        
        return True
    
    @staticmethod
    def _check_CH2F(mol):
        """
        Check if the molecule contains a CH2F group based on amount of hydrogen 
            near "C" atom. If "C" near fluorine hasn't 2 hydrogen atoms it's not CH2F.

        Args:
            mol (rdchem.Mol): RDKit molecule.

        Returns:
            bool: True if CH2F is present, False otherwise.
        """
        f_group_submol = Chem.MolFromSmiles("CCF")
        f_group_matches = mol.GetSubstructMatches(f_group_submol)

        if len(f_group_matches) == 0:
            return False
        
        c_atom_index = f_group_matches[0][1]
        
        amount_of_hydrogen = amount_of_hydrogen = Molecule3DFeaturesService._amount_of_hydrogen_in_neighbors(mol, c_atom_index)

        if amount_of_hydrogen != 2:
            return False
        
        return True
    
    def calculate_fluoric_group(self):
        """
        Identify the fluorine-containing functional group in the molecule.

        Returns:
            str: Identified functional group or 'non-F' if none found.
        """
        for f_group, f_group_SMILES in FUNCTIONAL_GROUP_TO_SMILES.items():
            f_group_submol = Chem.MolFromSmiles(f_group_SMILES)
            f_group_matches = self.mol.GetSubstructMatches(f_group_submol)

            if len(f_group_matches) > 0:
                if "gem-CF2" == f_group:
                    is_gem_CF2 = Molecule3DFeaturesService._check_gem_CF2(self.mol)
                    if not is_gem_CF2:
                        continue

                if "CH2F" == f_group:
                    is_CH2F = Molecule3DFeaturesService._check_CH2F(self.mol)
                    if not is_CH2F:
                        continue
                return f_group

        return "non-F"

    def calculate_f_group_freedom(self):
        """
        Calculate the freedom of the fluoric functional group.

        Args:
            functional_group (str): The functional group.

        Returns:
            int: The freedom of the functional group.
        """
        functional_group_to_freedom = {
                "CF3": 0, 
                "CH2F": 1, 
                "gem-CF2": 0, 
                "CHF2": 1,
                "CHF": 1,
                "non-F": 1,
            }
        
        return functional_group_to_freedom[self.f_group]

    def calculate_identificator(self) -> Identificator:
        """
        Identify the type of molecule.

        Returns:
            Identificator: Type of functional group.

        Raises:
            TypeError: If the molecule doesn't match any expected functional group.
        """
        if self.target_value == Target.pKa:
            return utils_pKa.calculate_identificator(self.mol)
        elif self.target_value == Target.logP:
            return utils_logP.calculate_identificator(self.mol)

    @staticmethod
    def set_average_atoms_position(mol,
                                   atoms_idx: list,
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
    def dihedral_angle(mol, 
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
            (float): dihedral angle between areas in degrees.
        """
        conf = mol.GetConformer(conf_id)

        return abs(rdMolTransforms.GetDihedralDeg(conf, iAtomId, jAtomId, kAtomId, lAtomId))

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

        return atom.IsInRing()

    def calculate_volume(self):
        """
        Calculate the molecule Volume.

        Returns:
            mol_volume (float): Float value with calculated molecule volume
        """
        mol_volume = AllChem.ComputeMolVolume(mol=self.mol,
                                              confId=self.min_energy_conf_index)
        
        return mol_volume

    def calculate_sasa(self):
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
    
    def find_X1X2R1R2(self):
        """
        Determines which atoms correspond to X1, X2, R1 and R2
            
        Returns:
            X1 (int): Atom id in cycle, that connects NH2 or COOH to the molecule.
            X2 (int): Atom id in cycle, that connects fluorine functional group to the molecule.
            R1 (int): NH2 or COOH functional group center atom id.
            R2 (int): fluorine functional group center atom id.
        """
        f_group_smiles = FUNCTIONAL_GROUP_TO_SMILES[self.f_group]

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

        if "amine" in self.identificator.name.lower():
            if len(nitro_amine_matches) == 0:
                raise "Problem with amine"
            
            if Identificator.primary_amine == self.identificator:
                X1 = nitro_amine_matches[0][0]
                R1 = nitro_amine_matches[0][1]
            elif Identificator.secondary_amine == self.identificator:
                X1 = nitro_amine_matches[0][1]
                self.mol, R_1 = Molecule3DFeaturesService.set_average_atoms_position(self.mol, [nitro_amine_matches[0][0], nitro_amine_matches[1][0]], self.min_energy_conf_index)
                self.mol, R1 = Molecule3DFeaturesService.change_vector_direction(self.mol, X1, R_1=R_1, conf_id=self.min_energy_conf_index)

        X2, R2 = None, None
        f_group_submol = Chem.MolFromSmiles(f_group_smiles)
        f_group_matches = self.mol.GetSubstructMatches(f_group_submol)
        if self.f_group.upper() in ['CF3', 'CHF2', 'CH2F']:
            X2 = f_group_matches[0][0]
            R2 = f_group_matches[0][1]

        elif self.f_group == 'gem-CF2':
            X2 = f_group_matches[0][0]
            self.mol, R2 = Molecule3DFeaturesService.set_average_atoms_position(self.mol, [f_group_matches[0][1], f_group_matches[0][2]], self.min_energy_conf_index)

        elif self.f_group.upper() == 'CHF':
            if len(f_group_matches) == 1:
                X2 = f_group_matches[0][0]
                R2 = f_group_matches[0][1]
            elif len(f_group_matches) == 2:
                self.mol, X2 = Molecule3DFeaturesService.set_average_atoms_position(self.mol, [f_group_matches[0][0], f_group_matches[1][0]], self.min_energy_conf_index)
                self.mol, R2 = Molecule3DFeaturesService.set_average_atoms_position(self.mol, [f_group_matches[0][1], f_group_matches[1][1]], self.min_energy_conf_index)
        else:
            return X1, X2, R1, R2
        
        if len(set([X1, X2, R1, R2])) != 4:
            X1, X2, R1, R2 = None, None, None, None
        
        if not Molecule3DFeaturesService.is_atom_in_cycle(mol=self.mol, atom_id=f_group_matches[0][0]):
            X1, X2, R1, R2 = None, None, None, None

        return X1, X2, R1, R2

    def calculate_dihedral_angle(self):
        """
        Calculate dihedral angle in molecule between X1X2R1 and X2R1R2 flat areas with lowest energy conformer.
        
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
        if len(set([self.X1, self.X2, self.R1, self.R2])) != 4:
            return None

        dihedral_angle_value = Molecule3DFeaturesService.dihedral_angle(self.mol, 
                                                                        self.R2, self.X2, self.X1, self.R1, 
                                                                        self.min_energy_conf_index) 
        return dihedral_angle_value
    
    @staticmethod
    def distance_between_atoms(iAtom_pos, jAtom_pos):
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

    def calculate_distance_between_atoms_in_cycle(self):
        """
        Calculate distance in molecule between X1 and X2.

        Returns:
            r_distance (float): dihedral between X1 and X2 atoms.
        """
        if len(set([self.X1, self.X2, self.R1, self.R2])) != 4:
            return None

        X1_pos = self.mol.GetConformer(self.min_energy_conf_index).GetAtomPosition(self.X1)
        X2_pos = self.mol.GetConformer(self.min_energy_conf_index).GetAtomPosition(self.X2)

        r_distance = Molecule3DFeaturesService.distance_between_atoms(X1_pos, X2_pos)

        return r_distance
    
    def calculate_distance_between_atoms_in_f_group_centers(self):
        """
        Calculate distance in molecule between R1 and R2.

        Returns:
            R_distance (float): distance between R1 and R2 atoms.
        """
        if len(set([self.X1, self.X2, self.R1, self.R2])) != 4:
            return None

        R1_pos = self.mol.GetConformer(self.min_energy_conf_index).GetAtomPosition(self.R1)
        R2_pos = self.mol.GetConformer(self.min_energy_conf_index).GetAtomPosition(self.R2)

        R_distance = Molecule3DFeaturesService.distance_between_atoms(R1_pos, R2_pos)

        return R_distance

    @staticmethod
    def flat_angle(mol, 
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

    def calculate_flat_angle_between_atoms_in_cycle(self):
        """
        Calculate 2 flat angles(in Deg) between 2 atoms in cycle and 1 atom in the center of functional group.

        If angle is not possible to calculate angles are None.

        Returns:
            flat_angle_between_atoms_in_cycle_1 (float): angle between X1, X2 and R2 atoms.
            flat_angle_between_atoms_in_cycle_2 (float): angle between X2, X1 and R1 atoms.
        """
        if len(set([self.X1, self.X2, self.R1, self.R2])) != 4:
            return None, None

        flat_angle_between_atoms_in_cycle_1 = Molecule3DFeaturesService.flat_angle(self.mol, self.X1, self.X2, self.R2, self.min_energy_conf_index)
        flat_angle_between_atoms_in_cycle_2 = Molecule3DFeaturesService.flat_angle(self.mol, self.X2, self.X1, self.R1, self.min_energy_conf_index)

        return flat_angle_between_atoms_in_cycle_1, flat_angle_between_atoms_in_cycle_2

    def calculate_flat_angle_between_atoms_in_f_group_center(self):
        """
        Calculate 2 flat angles(in Deg) between 2 atoms in the center of functional groups and 1 atom in cycle.

        If angle is not possible to calculate angles are None.

        Returns:
            flat_angle_between_atoms_in_f_group_center_1 (float): angle between R2, X2 and R1 atoms.
            flat_angle_between_atoms_in_f_group_center_2 (float): angle between R1, X1 and R2 atoms.
        """
        if len(set([self.X1, self.X2, self.R1, self.R2])) != 4:
            return None, None

        flat_angle_between_atoms_in_f_group_center_1 = Molecule3DFeaturesService.flat_angle(self.mol, self.R2, self.X2, self.R1, self.min_energy_conf_index)
        flat_angle_between_atoms_in_f_group_center_2 = Molecule3DFeaturesService.flat_angle(self.mol, self.R1, self.X1, self.R2, self.min_energy_conf_index)

        return flat_angle_between_atoms_in_f_group_center_1, flat_angle_between_atoms_in_f_group_center_2
    
    @staticmethod
    def _is_on_the_same_side(R1X1X2_angle: float,
                             R1X1R2_angle: float,
                             threshold: int = 5):
        """
        Determine if the functional group and fluorine substituents are on the same side of the molecule.
        For this R1X1X2 and R1X1R2 angles are used.

        Args:
            R1X1X2_angle (float): Angle R1X1X2.
            R1X1R2_angle (float): Angle R1X1R2.
            threshold (int, optional): Threshold value. Defaults to 5.

        Returns:
            bool: True if the functional group and fluorine substituents are on the same side, False otherwise.
        """
        if R1X1X2_angle - threshold > R1X1R2_angle:
            return True
        
        return False
    
    @staticmethod
    def _find_the_furthest_atom_id(mol: rdchem.Mol,
                                   atom_id: int):
        """
        Find the furthest atom ID from a given atom ID in a molecule using BFS.

        Args:
            mol (rdchem.Mol): RDKit molecule.
            atom_id (int): Atom ID.

        Returns:
            tuple: Tuple containing the furthest atom ID and its distance.
        """
        queue = deque([(atom_id, 0)])
    
        visited = set()
        
        while queue:
            current_atom, distance = queue.popleft()
            
            visited.add(current_atom)
            
            neighbors = []
            for atom in mol.GetAtomWithIdx(current_atom).GetNeighbors():
                if atom.GetSymbol().lower() == 'h':
                    continue
                neighbors.append(atom.GetIdx())
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append((neighbor, distance + 1))
        
        return current_atom, distance

    @staticmethod
    def _is_on_the_same_side_(mol: rdchem.Mol,
                              identificator: Identificator,
                              conf_id: int):
        """
        Determine if the functional group and fluorine substituents are on the same side of the molecule.
        For this R1X1X2 and R1X1R2 angles are used.

        Args:
            mol (rdchem.Mol): RDKit molecule.
            identificator (Identificator): Identifier for specific atoms (acid, primary amine, secondary amine).
            conf_id (int): Conformer with the lowest energy.

        Returns:
            bool: True if the functional group and fluorine substituents are on the same side, False otherwise.
        """
        
        ring_submol = Chem.MolFromSmiles("C1=CC=CC=C1")
        ring_matches = mol.GetSubstructMatches(ring_submol)
        atoms_to_skip = ring_matches[0] if len(ring_matches) > 0 else []

        if identificator == Identificator.carboxilic_acid:
            # submol = Chem.MolFromSmiles('CC=O')
            submol = Chem.MolFromSmiles('C=O')
        else:
            submol = Chem.MolFromSmiles('CN')

        matches = mol.GetSubstructMatches(submol)
        atom_oxygen_idx = matches[0][1]
        # atomR1Idx = matches[0][1]
        # atomX1Idx = matches[0][0]
        atomX1Idx, atomR1Idx = utils.find_the_closest_atom_in_ring(mol=mol,
                                                                   atom_id=atom_oxygen_idx,
                                                                   atoms_not_to_visit=atoms_to_skip)

        atomR2Idx, _ = utils.find_the_furthest_atom(mol=mol, 
                                                    atom_id=atomR1Idx,
                                                    atoms_not_to_visit=atoms_to_skip)
        atomX2Neighbors = mol.GetAtomWithIdx(atomR2Idx).GetNeighbors()
        for atomX2Neighbor in atomX2Neighbors:
            if atomX2Neighbor.GetSymbol().lower() != 'h':
                atomX2Idx = atomX2Neighbor.GetIdx()
                break

        angle_R1X1R2 = Molecule3DFeaturesService.flat_angle(mol=mol,
                                                            iAtomId=atomR1Idx,
                                                            jAtomId=atomX1Idx,
                                                            kAtomId=atomR2Idx,
                                                            conf_id=conf_id)
        
        angle_R1X1X2 = Molecule3DFeaturesService.flat_angle(mol=mol,
                                                            iAtomId=atomR1Idx,
                                                            jAtomId=atomX1Idx,
                                                            kAtomId=atomX2Idx,
                                                            conf_id=conf_id)
        
        return Molecule3DFeaturesService._is_on_the_same_side(R1X1X2_angle=angle_R1X1X2,
                                                              R1X1R2_angle=angle_R1X1R2)

    @staticmethod
    def _amount_of_specific_atoms_in_molecule(mol: rdchem.Mol,
                                              atom_symbol: str = 'F'):
        """
        Calculate the number of specific atoms in a molecule.

        Args:
            mol (rdchem.Mol): RDKit molecule.
            atom_symbol (str, optional): Atom symbol to count. Defaults to 'F'.

        Returns:
            int: Number of specific atoms in the molecule.
        """
        amount_of_atoms = 0
        
        for atom in mol.GetAtoms():
            if atom.GetSymbol().lower() == atom_symbol.lower():
                amount_of_atoms += 1

        return amount_of_atoms

    @staticmethod
    def _first_atoms_in_cycle(mol: rdchem.Mol,
                              atoms_id: list,
                              amount_of_atoms: int = 3):
        """
        Find the first atoms in a cycle starting from given atom IDs using BFS.

        Args:
            mol (rdchem.Mol): RDKit molecule.
            atoms_id (list): List of starting atom IDs for BFS.
            amount_of_atoms (int, optional): Number of atoms in the cycle. Defaults to 3.

        Returns:
            list: List of atom IDs representing the first atoms in the cycle from given atom id.
        """
        queue = deque(atoms_id)
    
        visited = set()

        atoms_in_ring = []
        
        while queue:
            current_atom = queue.popleft()

            if mol.GetAtomWithIdx(current_atom).IsInRing() and current_atom not in atoms_in_ring:
                atoms_in_ring.append(current_atom)
                if len(atoms_in_ring) == amount_of_atoms:
                    return atoms_in_ring
            
            visited.add(current_atom)
            
            neighbors = []
            for atom in mol.GetAtomWithIdx(current_atom).GetNeighbors():
                if atom.GetSymbol().lower() == 'h':
                    continue
                neighbors.append(atom.GetIdx())
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append(neighbor)
        
        return atoms_in_ring
    
    @staticmethod
    def _is_on_the_same_side_of_plane(plane_atoms_pos: list,
                                      atom1_pos: tuple,
                                      atom2_pos: tuple):
        """
        Determine if two atoms are on the same side of a plane.

        Args:
            plane_atoms_pos (list): Positions of atoms defining the plane.
            atom1_pos (tuple): Position of the first atom.
            atom2_pos (tuple): Position of the second atom.

        Returns:
            bool: True if atoms are on the same side of the plane, False otherwise.
        """
        atom_a = plane_atoms_pos[0]
        atom_b = plane_atoms_pos[1]
        atom_c = plane_atoms_pos[2]
        
        vector_ab = atom_b - atom_a
        vector_ac = atom_c - atom_a

        normal_vector = np.cross(vector_ab, vector_ac)

        d = -np.dot(normal_vector, atom_a)

        plane_eq_value_f_1 = np.dot(atom1_pos, normal_vector) + d
        plane_eq_value_f_2 = np.dot(atom2_pos, normal_vector) + d

        if np.sign(plane_eq_value_f_1) == np.sign(plane_eq_value_f_2):
            return True
        else:
            return False

    @staticmethod
    def _is_R1_R2_on_the_same_side(mol: rdchem.Mol,
                                   R1: int,
                                   X1: int,
                                   conf_id: int = -1):
        """
        Determine if two atoms are on the same side of a plane defined by a ring.

        Args:
            mol (rdchem.Mol): RDKit molecule.
            R1 (int): Atom R1 ID.
            X1 (int): Atom X1 ID.
            conf_id (int, optional): Conformer id with the lowest energy. Defaults to -1.

        Returns:
            bool: True if atoms R1 and R2 are on the same side of the plane, False otherwise.
        """
        for atom in mol.GetAtoms():
            if atom.GetSymbol().lower() == 'f':
                fluorine_atom = atom.GetIdx()
                break

        first_atom_from_fluorine_in_ring = Molecule3DFeaturesService._first_atoms_in_cycle(mol=mol,
                                                                                           atoms_id=[fluorine_atom],
                                                                                           amount_of_atoms=1)[0]
        
        X2_atom = mol.GetAtomWithIdx(first_atom_from_fluorine_in_ring)
        for neighbor in X2_atom.GetNeighbors():
            if not neighbor.IsInRing() and neighbor.GetSymbol().lower() != 'h':
                R2_atom = neighbor
                break

        X2 = X2_atom.GetIdx()
        R2 = R2_atom.GetIdx()

        three_atoms_in_ring = Molecule3DFeaturesService._first_atoms_in_cycle(mol=mol,
                                                                              atoms_id=[X1],
                                                                              amount_of_atoms=3)

        atoms_from__ring_pos = []
        for atom_in_ring in three_atoms_in_ring:
            atoms_from__ring_pos.append(np.array(mol.GetConformer(conf_id).GetAtomPosition(atom_in_ring)))

        atom_1 = np.array(mol.GetConformer(conf_id).GetAtomPosition(R1))
        atom_2 = np.array(mol.GetConformer(conf_id).GetAtomPosition(R2))

        return Molecule3DFeaturesService._is_on_the_same_side_of_plane(plane_atoms_pos=atoms_from__ring_pos, 
                                                                       atom1_pos=atom_1,
                                                                       atom2_pos=atom_2)

    @staticmethod
    def _is_fluorines_on_the_same_side(mol: rdchem.Mol,
                                       conf_id: int = -1):
        """
        Determine if two fluorine atoms are on the same side of a plane.
        Used for molecules where we have 2 CHF groups.

        Args:
            mol (rdchem.Mol): RDKit molecule.
            conf_id (int, optional): Conformer id with the lowest energy. Defaults to -1.

        Returns:
            bool: True if fluorine atoms are on the same side of the plane, False otherwise.
        """
        fluorines = []

        for atom in mol.GetAtoms():
            if atom.GetSymbol().lower() == 'f':
                fluorines.append(atom.GetIdx())

        fluorine_atom_1 = fluorines[0]
        fluorine_atom_2 = fluorines[1]

        atoms_from_fluorine_in_ring = Molecule3DFeaturesService._first_atoms_in_cycle(mol=mol,
                                                                                      atoms_id=[fluorine_atom_1, fluorine_atom_2],
                                                                                      amount_of_atoms=3)

        atoms_from_fluorine_in_ring_pos = []
        for atom_from_fluorine_in_ring in atoms_from_fluorine_in_ring:
            atoms_from_fluorine_in_ring_pos.append(np.array(mol.GetConformer(conf_id).GetAtomPosition(atom_from_fluorine_in_ring)))

        atom_f_1 = np.array(mol.GetConformer(conf_id).GetAtomPosition(fluorine_atom_1))
        atom_f_2 = np.array(mol.GetConformer(conf_id).GetAtomPosition(fluorine_atom_2))

        return Molecule3DFeaturesService._is_on_the_same_side_of_plane(plane_atoms_pos=atoms_from_fluorine_in_ring_pos, 
                                                                       atom1_pos=atom_f_1,
                                                                       atom2_pos=atom_f_2)
        
    def calculate_cis_trans(self):
        """
        Calculate cis-trans configuration based on specific conditions for rdkit molecule.

        - if there is no chiral centers in the molecule or no "@" in the molecule cis/trans is nan.
        - if We haven't any fluorine groups in our molecule we are using COOH or NH2 and the furthest atom from it.
            For defining cis/trans we are checking two angles from functional group to the furthest atom.
        - If we have 2 CHF groups we are checking if fluorine atoms are the from the same part of molecule or they are
            from the different part of the molecule.
        - In all others cases we are looking to X1R1X2, X1R1R2 angles and their values, 
            if X1R1X2 > X1R1R2 then molecules are on the same part.
        
        Returns:
            'cis' if atoms are on the same side, 'trans' otherwise, nan if there is no chiral centers in the molecule.
        """
        if "@" not in self.smiles:
            return np.nan
        
        if len(set([self.X1, self.X2, self.R1, self.R2])) != 4:
            if self.f_group == "non-F":
                is_on_the_same_side = Molecule3DFeaturesService._is_on_the_same_side_(mol=self.mol,
                                                                                      identificator=self.identificator,
                                                                                      conf_id=self.min_energy_conf_index) 
        else:
            if self.f_group == "CHF" and Molecule3DFeaturesService._amount_of_specific_atoms_in_molecule(mol=self.mol,
                                                                                                         atom_symbol='F') == 2:
                is_on_the_same_side = Molecule3DFeaturesService._is_fluorines_on_the_same_side(mol=self.mol,
                                                                                               conf_id=self.min_energy_conf_index)
            else:
                is_on_the_same_side = Molecule3DFeaturesService._is_on_the_same_side(self.flat_angle_between_atoms_in_cycle_2,
                                                                                     self.flat_angle_between_atoms_in_f_group_center_2)
        if is_on_the_same_side:
            return "cis"
        else:
            return "trans"
        
    def calculate_linear_path_f_to_fg(self):
        """
        Calculate amount of linear paths from fluorine (F) to the functional group (FG) based on the target value.

        If the target value is pKa, calculate the linear path from fluorine to the functional group using the provided SMILES.
        If the target value is logP, calculate the linear path from fluorine to the functional group using the provided SMILES and identificator.

        Returns:
            int: Amount of the linear paths from fluorine to the functional group.

        Raises:
            ValueError: If the molecule identificator is inappropriate.
        """
        if self.target_value == Target.pKa:
            return utils_pKa.calculate_linear_path_f_to_fg(smiles=self.smiles)
        elif self.target_value == Target.logP:
            return utils_logP.calculate_linear_path_f_to_fg(smiles=self.smiles,
                                                            identificator=self.identificator)
        
        raise ValueError("inappropriate molecule identificator")
                       
    def calculate_dipole_moment(self):
        """
        Calculate the dipole moment based on atoms positions and charges using optimized molecule with lowest energy conformer.

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
                (charges_multiply_coordinates[charges_multiply_coordinate_index][0] - x_centroid) * charges[charges_multiply_coordinate_index]
            charges_multiply_coordinates[charges_multiply_coordinate_index][1] = \
                (charges_multiply_coordinates[charges_multiply_coordinate_index][1] - y_centroid) * charges[charges_multiply_coordinate_index]
            charges_multiply_coordinates[charges_multiply_coordinate_index][2] = \
                (charges_multiply_coordinates[charges_multiply_coordinate_index][2] - z_centroid) * charges[charges_multiply_coordinate_index]

        dipole_moment_vector = [0, 0, 0]
        for charges_multiply_coordinate_index in range(len(charges_multiply_coordinates)):
            dipole_moment_vector[0] += charges_multiply_coordinates[charges_multiply_coordinate_index][0]
            dipole_moment_vector[1] += charges_multiply_coordinates[charges_multiply_coordinate_index][1]
            dipole_moment_vector[2] += charges_multiply_coordinates[charges_multiply_coordinate_index][2]

        dipole_moment = math.sqrt(pow(dipole_moment_vector[0], 2) + pow(dipole_moment_vector[1], 2) + pow(dipole_moment_vector[2], 2))

        return dipole_moment
    
    def calculate_molecular_weight(self):
        """
        Calculate the molecular weight based on the target value.

        If the target value is pKa, the molecular weight is calculated using only the SMILES provided.
        If the target value is logP, the molecular weight is calculated using the SMILES and identificator.

        Returns:
            float: The calculated molecular weight, rounded to three decimal places.
        """
        if self.target_value == Target.pKa:
            molecular_weight =  utils_pKa.calculate_molecular_weight(SMILES=self.smiles)
        elif self.target_value == Target.logP:
            molecular_weight = utils_logP.calculate_molecular_weight(SMILES=self.smiles,
                                                         identificator=self.identificator)
        return round(molecular_weight, 3)

    def calculate_TPSA_with_fluor(self):
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
