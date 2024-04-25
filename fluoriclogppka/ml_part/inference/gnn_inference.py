import os
import pandas as pd

from fluoriclogppka.ml_part.constants import Target, Identificator
from fluoriclogppka.ml_part.constants import GNN_PKA_ACID_MODEL_PATH, GNN_PKA_AMINE_MODEL_PATH, GNN_LOGP_MODEL_PATH

from fluoriclogppka.ml_part.services.gnn_service import GNNService
from fluoriclogppka.ml_part.data_preparation.smiles_to_graph import Featurizer

from fluoriclogppka.ml_part.utils.molecule_features import obtain_identificator

class GNNInference:
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
                 model_path: str = None
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

        dataPrep = Featurizer(SMILES=SMILES,
                              target_value=target_value)
        self.bg = dataPrep.bg
        
        identificator = None
        if target_value == Target.pKa:
            identificator = obtain_identificator(SMILES=SMILES,
                                                 target_value=target_value)

        self.model_path = model_path
        if model_path is None:
            self.model_path = GNNInference.best_model_path(target_value=target_value,
                                                        identificator=identificator)
        
        self._gnnService = GNNService(self.model_path)
    
    @staticmethod
    def best_model_path(target_value: Target, identificator: Identificator = None):
        """
        Determines the best model path based on the target value and molecule type.

        Args:
            target_value (Target): The target property to predict (pKa or logP).
            identificator (Identificator): The type of the molecule.

        Returns:
            str: The path to the pre-trained model.
        """
        if target_value == Target.logP:
            model_path = GNN_LOGP_MODEL_PATH
        elif target_value == Target.pKa and identificator is not None and "amine" in identificator.name.lower():
            model_path = GNN_PKA_AMINE_MODEL_PATH
        elif target_value == Target.pKa and identificator is not None and "acid" in identificator.name.lower():
            model_path = GNN_PKA_ACID_MODEL_PATH
        
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), model_path)
        
        return model_path

    def predict(self):
        """
        Make predictions using the loaded model.

        Returns:
            predicted_value (float): Predicted pKa or logP target value.
        """
        predicted_value = self._gnnService.predict(self.bg)
        
        return predicted_value


if __name__ == "__main__":
    SMILES = "CCC(F)(F)CC(O)=O"
    
    inference = GNNInference(SMILES=SMILES,
                          target_value=Target.pKa)
    
    predicted_pKa = inference.predict()
    print(predicted_pKa)
