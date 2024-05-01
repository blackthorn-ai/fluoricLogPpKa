import pandas as pd

from dgllife.utils.mol_to_graph import SMILESToBigraph
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from dgllife.utils import AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer

from fluoriclogppka.ml_part.constants import Target

class Featurizer:
    """
    A class for featurizing molecules based on their SMILES representation.

    This class provides methods to featurize molecules for different target properties,
    such as pKa or logP.

    Attributes:
        SMILES (str): The SMILES string representing the molecule.
        target_value (Target): The target property to predict (pKa or logP).

    Methods:
        __init__(): Initializes the Featurizer object.
        prepare_pKa_graph(): Prepares a graph representation of the molecule for pKa prediction.
        prepare_logP_graph(): Prepares a graph representation of the molecule for logP prediction.
    """
    def __init__(self, 
                 SMILES: str,
                 target_value: Target
                 ) -> None:
        """
        Initialize the Featurizer object.

        Args:
            SMILES (str): The SMILES string representing the molecule.
            target_value (Target): The target property to predict (pKa or logP).

        Returns:
            None
        """
        self.SMILES = SMILES
        self.target_value = target_value

        if target_value == Target.pKa:
            self.bg = Featurizer.prepare_pKa_graph(self.SMILES)
        elif target_value == Target.logP:
            self.bg = Featurizer.prepare_logP_graph(self.SMILES)

    @staticmethod
    def prepare_pKa_graph(SMILES):
        """
        Prepares a graph representation of the molecule for pKa prediction.

        Args:
            SMILES (str): The SMILES string representing the molecule.

        Returns:
            graph (DGLGraph): The graph representation of the molecule.
        """
        smiles_to_graph = SMILESToBigraph(node_featurizer=CanonicalAtomFeaturizer(),
                                          edge_featurizer=CanonicalBondFeaturizer())

        return smiles_to_graph(SMILES)

    @staticmethod
    def prepare_logP_graph(SMILES):
        """
        Prepares a graph representation of the molecule for logP prediction.

        Args:
            SMILES (str): The SMILES string representing the molecule.

        Returns:
            graph (DGLGraph): The graph representation of the molecule.
        """
        smiles_to_graph = SMILESToBigraph(add_self_loop=True,
                                          node_featurizer=AttentiveFPAtomFeaturizer(),
                                          edge_featurizer=AttentiveFPBondFeaturizer(self_loop=True))

        return smiles_to_graph(SMILES)
