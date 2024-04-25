from fluoriclogppka.ml_part.constants import Target, ModelType
from fluoriclogppka.ml_part.inference.gnn_inference import GNNInference
from fluoriclogppka.ml_part.inference.h2o_inference import H2OInference

class Inference:
    """
    A class for making predictions using different inference models.

    This class allows for making predictions using either a GNN (Graph Neural Network) model
    or an H2O model. It provides a unified interface to make predictions regardless of the
    underlying model type.

    Attributes:
        SMILES (str): The SMILES string representing the molecule.
        target_value (Target): The target property to predict (default is pKa).
        model_path (str): The path to the pre-trained model file.
        model_type (ModelType): The type of the inference model (default is H2O).
        is_fast_mode (bool): A flag indicating whether to use a fast mode for prediction.

    Methods:
        __init__(): Initializes the Inference object.
        predict(): Makes predictions using the selected inference model.
    """
    def __init__(self, 
                 SMILES: str,
                 target_value: Target = Target.pKa,
                 model_path: str = None,
                 model_type: ModelType = ModelType.h2o,
                 is_fast_mode: bool = False
                 ) -> None:
        """
        Initialize the Inference object.

        Args:
            SMILES (str): The SMILES string representing the molecule.
            target_value (Target): The target property to predict (default is pKa).
            model_path (str, optional): The path to the pre-trained model file.
            model_type (ModelType, optional): The type of the inference model (default is H2O).
            is_fast_mode (bool, optional): A flag indicating whether to use a fast mode for prediction.

        Returns:
            None
        """
        if model_type == ModelType.gnn:
            self.inference = GNNInference(SMILES=SMILES,
                                          model_path=model_path,
                                          target_value=target_value)
        elif model_type == ModelType.h2o:
            self.inference = H2OInference(SMILES=SMILES,
                                          model_path=model_path,
                                          target_value=target_value,
                                          is_fast_mode=is_fast_mode)
            
    def predict(self):
        """
        Make predictions using the selected inference model.

        Returns:
            predicted_value (float): The predicted value.
        """
        predicted_value = self.inference.predict()

        return predicted_value

if __name__ == "__main__":
    SMILES = "F[C@H]1C[C@H](F)CN(C1)C(=O)C1=CC=CC=C1"
    
    inference = Inference(SMILES=SMILES,
                          target_value=Target.logP,
                          model_type=ModelType.gnn)
    
    predicted_logP = inference.predict()
    print(predicted_logP)
