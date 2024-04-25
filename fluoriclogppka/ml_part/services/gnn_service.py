import pandas as pd

from fluoriclogppka.ml_part.constants import Target
from fluoriclogppka.ml_part.constants import LOGP_MODEL_PATH, PKA_AMINE_MODEL_PATH, PKA_ACID_MODEL_PATH
from fluoriclogppka.ml_part.utils.gnn_models import PKaAcidicModel, PKaBasicModel, LogPModel

class GNNService:
    """
    A class for making predictions using Graph Neural Network (GNN) models.

    This class provides methods to load and use pre-trained GNN models for predicting
    various molecular properties such as pKa and logP.

    Attributes:
        model_path (str): The path to the pre-trained GNN model file.

    Methods:
        __init__(): Initializes the GNNService object.
        _model_init(): Initializes the specified GNN model based on the provided model path.
        predict(): Makes predictions using the loaded GNN model.
    """
    def __init__(self,
                 model_path: str):
        """
        Initialize the GNNService object.

        Args:
            model_path (str): The path to the pre-trained GNN model file.

        Returns:
            None
        """
        self.model_path = model_path
        
        self.model = GNNService._model_init(self.model_path)
    
    @staticmethod
    def _model_init(model_path):
        """
        Initializes the specified GNN model based on the provided model path.

        Args:
            model_path (str): The path to the pre-trained GNN model file.

        Returns:
            model: The initialized GNN model.
        """
        if 'acid' in model_path.lower():
            model = PKaAcidicModel(model_path=model_path)
        elif 'amine' in model_path.lower():
            model = PKaBasicModel(model_path=model_path)
        elif 'logp' in model_path.lower():
            model = LogPModel(model_path=model_path)
        else:
            raise ValueError("Model name has an invalid name.")

        return model
    
    def predict(self,
                bg):
        """
        Make predictions using the loaded GNN model.

        Args:
            bg (DGLGraph): The graph representation of the molecule.

        Returns:
            prediction: The predicted molecular property.
        """
        self.model.eval()

        prediction = self.model.predict(bg=bg)

        return prediction
