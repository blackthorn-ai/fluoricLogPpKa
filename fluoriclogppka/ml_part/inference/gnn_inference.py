import os
import pandas as pd

from fluoriclogppka.ml_part.constants import Target
from fluoriclogppka.ml_part.constants import GNN_PKA_MODEL_PATH, GNN_LOGP_MODEL_PATH

from fluoriclogppka.ml_part.services.gnn_service import GNNService
from fluoriclogppka.ml_part.data_preparation.smiles_to_graph import Featurizer

class GNNInference:
    """
    A class for making predictions using Graph Neural Network (GNN) models.

    This class allows for making predictions using GNN models to predict pKa or logP values
    based on molecular structures.

    Attributes:
        SMILES (str): The SMILES string representing the molecule.
        target_value (Target): The target property to predict (pKa or logP).
        model_path (str): The path to the pre-trained model file.

    Methods:
        __init__(): Initializes the GNNInference object.
        best_model_path(): Determines the best model path based on the target value and molecule type.
        predict(): Makes predictions using the loaded GNN model.
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
            model_path (str, optional): The path to the pre-trained model. Defaults to None.
        """

        dataPrep = Featurizer(SMILES=SMILES,
                              target_value=target_value)
        self.bg = dataPrep.bg

        self.model_path = model_path
        if model_path is None:
            self.model_path = GNNInference.best_model_path(target_value=target_value)
        
        self._gnnService = GNNService(self.model_path)
    
    @staticmethod
    def best_model_path(target_value: Target):
        """
        Determines the best model path based on the target value and molecule type.

        Args:
            target_value (Target): The target property to predict (pKa or logP).

        Returns:
            str: The path to the pre-trained model.
        """
        if target_value == Target.logP:
            model_path = GNN_LOGP_MODEL_PATH
        elif target_value == Target.pKa:
            model_path = GNN_PKA_MODEL_PATH
        
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
