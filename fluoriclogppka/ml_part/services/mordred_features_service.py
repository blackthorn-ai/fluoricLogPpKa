from rdkit import Chem
from mordred import Calculator, descriptors
from rdkit.Chem import AllChem

from fluoriclogppka.ml_part.services.utils import has_numbers

class MordredFeaturesService:
    """
    A service class for obtaining Mordred features from chemical structures.

    This class provides methods to prepare a molecule from a SMILES string and
    obtain Mordred features for the molecule.

    Attributes:
        mol: The molecule object prepared from the SMILES string.
        mordred_features_dict: A dictionary containing Mordred features extracted
            from the molecule.

    Methods:
        __init__(): Initializes the MordredFeaturesService object.
        prepare_molecule(): Prepares a molecule object from a SMILES string.
        obtain_mordred_features(): Obtains Mordred features for the molecule.
    """
    def __init__(self,
                 SMILES):
        """
        Initialize the MordredFeaturesService object.

        Args:
            SMILES (str): The SMILES string representing the molecule.
        """
        self.mol = MordredFeaturesService.prepare_molecule(SMILES)

        self.mordred_features_dict = self.obtain_mordred_features()


    @staticmethod
    def prepare_molecule(SMILES):
        """
        Prepare a molecule object from a SMILES string.

        Args:
            SMILES (str): The SMILES string representing the molecule.

        Returns:
            Mol: The prepared molecule object.
        """
        mol = Chem.MolFromSmiles(SMILES)
        mol = Chem.AddHs(mol)

        AllChem.EmbedMolecule(mol, randomSeed=42)

        return mol


    def obtain_mordred_features(self):
        """
        Obtain Mordred features for the molecule.

        Returns:
            dict: A dictionary containing Mordred features extracted from the molecule.
        """
        calc = Calculator(descriptors, ignore_3D=False)
        df = calc.pandas([self.mol], quiet=True)

        mordred_dict = {}
        for _, row in df.iterrows():
            for key in row.keys():
                if "ring" in key.lower() and has_numbers(key):
                    continue
                if type(row[key]) in [int, float]:
                    mordred_dict[key] = row[key]

        return mordred_dict
