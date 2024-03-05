import pandas as pd

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
        row_from_enamine_dataset (pd.Series): A row from the Enamine Dataframe containing molecule information.

    Methods:
        extract_all_features(): Extracts all posible features for the molecule from 
            rdkit, mordred and our dataset.
        extract_required_features(): Extracts only the required features for predicting pKa or LogP.
        prepare_features_for_model(): Converts string molecule features to int. 
    """
    def __init__(self, 
                 SMILES: str,
                 target_value: Target,
                 ) -> None:
        """
        Initialize the PrepareFluorineData object.

        Args:
            SMILES (str): The SMILES string representing the molecule.
            target_value (Target): The target property to predict (pKa or logP).
            row_from_enamine_dataset (pd.Series): A row from the Enamine dataset containing molecule information.
        """
        self.SMILES = SMILES
        self.target_value = target_value

        if target_value == Target.pKa:
            self.required_features = PKA_FEATURES
        elif target_value == Target.logP:
            self.required_features = LOGP_FEATURES

        self.all_features_dict = self.extract_all_features()

        self.required_features_for_predict = self.extract_required_features()

        self.features_for_predict = self.prepare_features_for_model()

    def extract_all_features(self):
        """
        Extracts all posible features for the molecule from rdkit, mordred and our dataset.

        Returns:
            dict: A dictionary containing all extracted features from rdkit, mordred and Enamine dataset.
        """
        all_features = {}

        mordredFeaturesService = MordredFeaturesService(self.SMILES)
        all_features.update(mordredFeaturesService.mordred_features_dict)

        moleculeFeatures2dService = Molecule2DFeaturesService(self.SMILES)
        all_features.update(moleculeFeatures2dService.features_2d_dict)

        moleculeFeatures3dService = Molecule3DFeaturesService(smiles=self.SMILES,
                                                              target_value=self.target_value)
        all_features.update(moleculeFeatures3dService.features_3d_dict)

        return all_features

    def extract_required_features(self):
        """
        Extracts only the required features for predicting target value(pKa or LogP).

        Returns:
            dict: A dictionary containing the required features and their values for prediction.
        """
        for feature_name in self.required_features:
            if feature_name not in self.all_features_dict:
                raise FeatureNotFoundError(feature_name)
            
        required_features_dict = {feature: self.all_features_dict[feature] for feature in self.required_features} 
        return required_features_dict
    
    def prepare_features_for_model(self):
        """
        Prepare features for model prediction.

        This method prepares features for model prediction based on the required features
        for prediction and their corresponding int values. It applies any necessary conversions
        to the feature values before returning the prepared features.

        Returns:
            dict: A dictionary containing the prepared features for model prediction.
        """
        features_for_predict = self.required_features_for_predict.copy()
        
        for feature_name, value in features_for_predict.items():
            if feature_name in CONVERT_FEATURE_TO:
                features_for_predict[feature_name] = CONVERT_FEATURE_TO[feature_name][value]

        return features_for_predict
