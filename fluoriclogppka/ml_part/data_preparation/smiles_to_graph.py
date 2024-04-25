import pandas as pd

from dgllife.utils.mol_to_graph import SMILESToBigraph
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer

from fluoriclogppka.ml_part.exceptions import FeatureNotFoundError

from fluoriclogppka.ml_part.constants import Target
from fluoriclogppka.ml_part.constants import LOGP_FEATURES, PKA_FEATURES
from fluoriclogppka.ml_part.constants import CONVERT_FEATURE_TO

from fluoriclogppka.ml_part.services.molecule_3d_features_service import Molecule3DFeaturesService
from fluoriclogppka.ml_part.services.molecule_2d_features_service import Molecule2DFeaturesService
from fluoriclogppka.ml_part.services.mordred_features_service import MordredFeaturesService

class Featurizer:
    """
    A class for preparing data for chemical property prediction.

    This class prepares data from Enamine dataset for predicting chemical properties such as pKa or logP.

    Attributes:
        SMILES (str): The SMILES string representing the molecule.
        target_value (Target): The target property to predict (pKa or logP).
        conformers_limit (int): Max number of generated conformers for optimization.

    Methods:
        extract_all_features(): Extracts all posible features for the molecule from 
            rdkit, mordred and our dataset.
        extract_required_features(): Extracts only the required features for predicting pKa or LogP.
        prepare_features_for_model(): Converts string molecule features to int. 
    """
    def __init__(self, 
                 SMILES: str,
                 target_value: Target
                 ) -> None:
        """
        Initialize the PrepareFluorineData object.

        Args:
            SMILES (str): The SMILES string representing the molecule.
            target_value (Target): The target property to predict (pKa or logP).
            conformers_limit (int): Max number of generated conformers for optimization.
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
        Extracts all posible features for the molecule from rdkit, mordred and our dataset.

        Returns:
            dict: A dictionary containing all extracted features from rdkit, mordred and Enamine dataset.
        """
        smiles_to_graph = SMILESToBigraph(node_featurizer=CanonicalAtomFeaturizer(),
                                          edge_featurizer=CanonicalBondFeaturizer())

        return smiles_to_graph(SMILES)

    @staticmethod
    def prepare_logP_graph(SMILES):
        """
        Extracts all posible features for the molecule from rdkit, mordred and our dataset.

        Returns:
            dict: A dictionary containing all extracted features from rdkit, mordred and Enamine dataset.
        """
        smiles_to_graph = SMILESToBigraph(add_self_loop=True,
                                          node_featurizer=CanonicalAtomFeaturizer(),
                                          edge_featurizer=CanonicalBondFeaturizer(self_loop=True))

        return smiles_to_graph(SMILES)
