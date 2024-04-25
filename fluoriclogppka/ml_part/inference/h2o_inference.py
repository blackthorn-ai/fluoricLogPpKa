import os
import pandas as pd

from fluoriclogppka.ml_part.constants import Target, Identificator
from fluoriclogppka.ml_part.constants import LOGP_MODEL_PATH, PKA_AMINE_MODEL_PATH, PKA_ACID_MODEL_PATH

from fluoriclogppka.ml_part.data_preparation.smiles_to_features import Featurizer
from fluoriclogppka.ml_part.services.h2o_service import H2OService

class H2OInference:
    """
    A class for making predictions using pre-trained models based on input molecule information.

    This class prepares data for prediction and utilizes an H2OService to make predictions
    using the appropriate pre-trained model.

    Attributes:
        SMILES (str): The SMILES string representing the molecule.
        target_value (Target): The target property to predict (pKa or logP).
        row_from_enamine_dataset: A row from the Enamine dataset containing molecule information.
        model_path (str): The path to the pre-trained model.

    Methods:
        best_model_path(): Determines the best model path based on the target value and molecule type.
        predict(): Makes predictions using the loaded model.
    """
    def __init__(self, 
                 SMILES: str,
                 target_value: Target = Target.pKa,
                 model_path: str = None,
                 is_fast_mode: bool = False
                 ) -> None:
        """
        Initialize the Inference object.

        Args:
            SMILES (str): The SMILES string representing the molecule.
            target_value (Target): The target property to predict (pKa or logP).
            row_from_enamine_dataset: A row from the Enamine dataset containing molecule information.
            model_path (str, optional): The path to the pre-trained model. Defaults to None.
            is_fast_mode (bool): Specifies whether to limit the number of conformers to speed up prediction.
        """
        conformers_limit = None
        if is_fast_mode:
            conformers_limit = 50

        dataPrep = Featurizer(SMILES=SMILES,
                              target_value=target_value,
                              conformers_limit=conformers_limit)
        self.features_for_predict = dataPrep.features_for_predict

        identificator = dataPrep.all_features_dict['identificator']

        self.model_path = model_path
        if model_path is None:
            self.model_path = H2OInference.best_model_path(target_value=target_value,
                                                        identificator=identificator)
        
        self._h2oService = H2OService(self.model_path)
        self.model = self._h2oService.model
    
    @staticmethod
    def best_model_path(target_value: Target, identificator: Identificator):
        """
        Determines the best model path based on the target value and molecule type.

        Args:
            target_value (Target): The target property to predict (pKa or logP).
            identificator (Identificator): The type of the molecule.

        Returns:
            str: The path to the pre-trained model.
        """
        if target_value == Target.logP:
            model_path = LOGP_MODEL_PATH
        elif target_value == Target.pKa and "amine" in identificator.name.lower():
            model_path = PKA_AMINE_MODEL_PATH
        elif target_value == Target.pKa and "acid" in identificator.name.lower():
            model_path = PKA_ACID_MODEL_PATH
        
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), model_path)
        
        return model_path

    def predict(self):
        """
        Make predictions using the loaded model.

        Returns:
            predicted_value (float): Predicted pKa or logP target value.
        """
        predicted_value = self._h2oService.predict(self.features_for_predict)
        
        return predicted_value


if __name__ == "__main__":
    SMILES = "F[C@H]1C[C@H](F)CN(C1)C(=O)C1=CC=CC=C1"
    
    inference = H2OInference(SMILES=SMILES,
                             target_value=Target.logP)
    
    predicted_logP = inference.predict()
    print(predicted_logP)
