from rdkit import Chem
from rdkit.Chem import AllChem

class Molecule2DFeaturesService:
    """
    Class that represents a 2D molecule features.

    This class provides functionality to obtain 2d molecules features. 

    Attributes:
        mol: Rdkit molecule with single conformer.
        mol_num_cycles (int): Amount of cycles in the molecule.
        atoms_num_in_cycles (float): Ratio of amount of atoms in rings to rings amount.
        chirality (int): Amount of chiral centers in the molecule.
        features_2d_dict (dict(str, float)): Feature name to feature value.

    Methods:
        prepare_molecule(): Creates rdkit molecule with single conformers from smiles.
        mol_cycles_amount(): Get amount of rings in the molecule.
        atoms_num_in_cycles_divide_by_amount_cycles(): Get ratio of amount of atoms in rings to rings amount.
        get_amount_of_chiral_centers(): Amount of chiral centers in molecule.
    """
    def __init__(self,
                 SMILES):
        """
        Initialize the Molecule2DFeaturesService instance and calculates molecule features,
        such as: amount of cycles, ratio of amount of atoms in rings to rings amount, amount of chiral centers.

        Args:
            SMILES (str): String representation of a molecule.
        """
        self.mol = Molecule2DFeaturesService.prepare_molecule(SMILES)

        self.mol_num_cycles = self.mol_cycles_amount()
        self.atoms_num_in_cycles = self.atoms_num_in_cycles_divide_by_amount_cycles()
        self.chirality = self.get_amount_of_chiral_centers()

        self.features_2d_dict = {
            "mol_num_cycles": self.mol_num_cycles,
            "avg_atoms_in_cycle": self.atoms_num_in_cycles,
            "chirality": self.chirality,
        }


    @staticmethod
    def prepare_molecule(SMILES):
        """
        Create rdkit 3d molecule from SMILES with conformer.
        
        Args:
            SMILES (str): String representation of a molecule. 
            
        Returns:
            mol: Rdkit 3D molecule with single conformer.
        """
        mol = Chem.MolFromSmiles(SMILES)
        mol = Chem.AddHs(mol)

        AllChem.EmbedMolecule(mol, randomSeed=42)

        return mol
    

    def mol_cycles_amount(self):
        """
        Calculate amount of cycles in the molecule.
            
        Returns:
            num_rings (int): amount of rings.
        """
        sssr = Chem.GetSSSR(self.mol)

        num_rings = len(sssr)
        return num_rings

    
    def atoms_num_in_cycles_divide_by_amount_cycles(self):
        """
        Calculate ratio amount of atoms in rings to rings amount.
            
        Returns:
            (float): atoms_num_in_cycles / amount_cycles.
        """
        sssr = Chem.GetSSSR(self.mol)
        amount_cycles = len(sssr)
        
        if amount_cycles == 0:
            return 0

        atoms_idxs = set()

        for i, ring in enumerate(sssr):
            for atom_index in ring:
                atoms_idxs.add(atom_index)
        
        atoms_num_in_cycles = len(atoms_idxs)
        
        return atoms_num_in_cycles / amount_cycles
    

    def get_amount_of_chiral_centers(self):
        """
        Calculate amount of chiral centers in molecule.
            
        Returns:
            amount_of_chiral_centers (int): amount of chiral centers in the molecule.
        """
        mol = Chem.AddHs(self.mol)
        AllChem.EmbedMolecule(mol)
        
        Chem.AssignAtomChiralTagsFromStructure(mol)
        chirality_centers = Chem.FindMolChiralCenters(mol)
        amount_of_chiral_centers = len(chirality_centers)
        
        return amount_of_chiral_centers
