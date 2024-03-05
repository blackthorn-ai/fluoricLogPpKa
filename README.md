# FluoricLogPpKa
Tool for predicting logP and pKa values.
## Features

- Obtain a lot of features from the molecule using custom methods, rdkit and mordred.
- Predict pKa and logP values using H2O models.

## Installation

FluoricLogPpKa requires installed Java to run.

```sh
pip install fluoriclogppka
```
## How to use

```
import fluoriclogppka

if __name__ == "__main__":

    SMILES = "F[C@H]1C[C@H](F)CN(C1)C(=O)C1=CC=CC=C1"

    inference = fluoriclogppka.Inference(SMILES=SMILES,
                                        target_value=fluoriclogppka.Target.logP)
        
    inference.predict()
```
