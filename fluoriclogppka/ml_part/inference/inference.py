from fluoriclogppka.ml_part.constants import Target, ModelType
from fluoriclogppka.ml_part.inference.gnn_inference import GNNInference
from fluoriclogppka.ml_part.inference.h2o_inference import H2OInference

class Inference:
    def __init__(self, 
                 SMILES: str,
                 target_value: Target = Target.pKa,
                 model_path: str = None,
                 model_type: ModelType = ModelType.h2o,
                 is_fast_model: bool = False
                 ) -> None:
        
        if model_type == ModelType.gnn:
            self.inference = GNNInference(SMILES=SMILES,
                                          model_path=model_path,
                                          target_value=target_value)
        elif model_type == ModelType.h2o:
            self.inference = H2OInference(SMILES=SMILES,
                                          model_path=model_path,
                                          target_value=target_value,
                                          is_fast_mode=is_fast_model)
            
    def predict(self):
        predicted_value = self.inference.predict()

        return predicted_value

if __name__ == "__main__":
    SMILES = "F[C@H]1C[C@H](F)CN(C1)C(=O)C1=CC=CC=C1"
    
    inference = Inference(SMILES=SMILES,
                          target_value=Target.logP,
                          model_type=ModelType.gnn)
    
    predicted_logP = inference.predict()
    print(predicted_logP)
