import pandas as pd

import warnings
import h2o
from h2o.exceptions import H2ODependencyWarning

warnings.filterwarnings("ignore", category=H2ODependencyWarning)

from fluoriclogppka.ml_part.constants import Target
from fluoriclogppka.ml_part.constants import LOGP_MODEL_PATH, PKA_AMINE_MODEL_PATH, PKA_ACID_MODEL_PATH

class H2OService:
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
        
        self.model = H2OService._model_init(self.model_path)
    
    @staticmethod
    def _model_init(model_path):
        """
        Initialize the H2O model from the specified path.

        Args:
            model_path (str): The path to the pre-trained H2O model.

        Returns:
            h2o_model: The initialized H2O model.
        """
        h2o.init(verbose=False)
        h2o.no_progress()

        h2o_model = h2o.load_model(model_path)

        return h2o_model
    
    @staticmethod
    def _prepare_h2o_data(features_dict: dict):
        """
        Prepare H2O data frame from a dictionary of features.

        Args:
            features_dict (dict): A dictionary containing features.

        Returns:
            h2o_frame: The prepared H2O data frame.
        """
        features_df = pd.DataFrame(features_dict, index=[0])
        
        h2o_frame = h2o.H2OFrame(features_df)

        return h2o_frame
    
    def predict(self,
                features_dict):
        """
        Make predictions using the loaded model.

        Args:
            features_dict (dict): A dictionary containing features.

        Returns:
            predictions (float): Predicted target value using h2o model.
        """
        h2o_frame = H2OService._prepare_h2o_data(features_dict)

        predictions_h2o_frame = self.model.predict(h2o_frame)

        predictions = predictions_h2o_frame.as_data_frame()['predict'][0]

        return predictions
