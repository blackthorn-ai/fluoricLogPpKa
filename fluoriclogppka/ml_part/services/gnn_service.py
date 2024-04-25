import pandas as pd

from fluoriclogppka.ml_part.constants import Target
from fluoriclogppka.ml_part.constants import LOGP_MODEL_PATH, PKA_AMINE_MODEL_PATH, PKA_ACID_MODEL_PATH
from fluoriclogppka.ml_part.utils.gnn_models import PKaAcidicModel, PKaBasicModel, LogPModel

class GNNService:
    """
    A service class for working with H2O models.

    This class provides methods for initializing an H2O model, converting data in 
    specific h2o format for prediction, and making predictions using the model.

    Attributes:
        model_path (str): The path to the pre-trained H2O model.

    Methods:
        _model_init(): Initializes the H2O model from the specified path.
        _prepare_h2o_data(): Prepares H2O data frame from a dictionary of features.
        predict(): Makes predictions using the loaded model.
    """
    def __init__(self,
                 model_path: str):
        """
        Initialize the H2OService object.

        Args:
            model_path (str): The path to the pre-trained H2O model.
        """
        self.model_path = model_path
        
        self.model = GNNService._model_init(self.model_path)
    
    @staticmethod
    def _model_init(model_path):
        """
        Initialize the H2O model from the specified path.

        Args:
            model_path (str): The path to the pre-trained H2O model.

        Returns:
            h2o_model: The initialized H2O model.
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
        Make predictions using the loaded model.

        Args:
            bg (Dgllife graph object): A dictionary containing features.

        Returns:
            prediction (float): Predicted target value using dgllife gnn model.
        """
        self.model.eval()

        prediction = self.model.predict(bg=bg)

        return prediction
