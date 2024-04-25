import os
from enum import Enum

import numpy as np

class Target(Enum):
    pKa = 'pKa'
    logP = 'logP'

class ModelType(Enum):
    gnn = 'gnn'
    h2o = 'h2o'

class Identificator(Enum):
    carboxilic_acid = 'carboxilic_acid'
    primary_amine = 'primary_amine'
    secondary_amine = 'secondary_amine'

LOGP_FEATURES = ['f_freedom', 'PPSA5', 'mol_num_cycles', 'nFRing', 'nF', 'identificator',
                 'mol_weight', 'dipole_moment', 'nHRing', 'nO', 'PBF', 'nC', 'nARing',
                 'cis/trans', 'PNSA5', 'FPSA3', 'mol_volume', 'RPCS', 'GeomShapeIndex',
                 'WPSA5', 'TASA', 'f_to_fg', 'avg_atoms_in_cycle', 'nFHRing',
                 'chirality']

PKA_FEATURES = ['RPCS', 'PBF', 'mol_weight', 'dipole_moment', 'PPSA5',
                'avg_atoms_in_cycle', 'nHRing', 'cis/trans', 'FPSA3', 'nF', 'chirality',
                'sasa', 'PNSA5', 'GeomShapeIndex', 'TASA', 'mol_num_cycles',
                'f_freedom', 'nFRing', 'identificator', 'nO', 'nARing', 'nC', 'nFHRing',
                'f_to_fg']

MODELS_PATH = os.path.join('fluoriclogppka', 'ml_part', 'models_weights')

# H2O
H2O_MODELS_PATH = os.path.join(MODELS_PATH, 'h2o_models')

LOGP_MODEL_PATH = os.path.join(H2O_MODELS_PATH, 'logP', 'all_molecules(without_angle_feature)_without_outliers', 'StackedEnsemble_BestOfFamily_3_AutoML_2_20240208_214951')

PKA_AMINE_MODEL_PATH = os.path.join(H2O_MODELS_PATH, 'pKa', 'amine_molecules(without_angle_feature)_without_outliers', 'StackedEnsemble_BestOfFamily_5_AutoML_3_20240213_92029')
PKA_ACID_MODEL_PATH = os.path.join(H2O_MODELS_PATH, 'pKa', 'acid_molecules(without_angle_feature)_without_outliers', 'DeepLearning_grid_2_AutoML_4_20240213_102321_model_43')

# GNN
GNN_MODELS_PATH = os.path.join(MODELS_PATH, 'gnn_models')

GNN_LOGP_MODEL_PATH = os.path.join(GNN_MODELS_PATH, 'logP', 'GCNPredictor_logP_best_loss.pth')

GNN_PKA_AMINE_MODEL_PATH = os.path.join(GNN_MODELS_PATH, 'pKa', 'amine_molecules(without_angle_feature)_without_outliers', 'basic_best_loss_glamorous-totem-39.pkl')
GNN_PKA_ACID_MODEL_PATH = os.path.join(GNN_MODELS_PATH, 'pKa', 'acid_molecules(without_angle_feature)_without_outliers', 'acid_best_loss_blooming-deluge-52.pkl')

FUNCTIONAL_GROUP_TO_SMILES = {
            "CF3": "CC(F)(F)F", 
            "CH2F": "CCF", 
            "gem-CF2": "C(F)(F)", 
            "CHF2": "CC(F)(F)",
            "CHF": "CF",
            "non-F": ""
        }

CONVERT_FEATURE_TO = {
    "identificator": {
        Identificator.carboxilic_acid: 0,
        Identificator.primary_amine: 1,
        Identificator.secondary_amine: 2,
    },
    "cis/trans": {
        np.nan: 0,
        "": 0,
        "cis": 1,
        "trans": 2
    }
}

ALL_SUBMOLS = {
    "CF3CH2N": ["FC(CN)(F)F"],
    "CF3(CH2)2N": ["FC(CCN)(F)F"],
    "CF3(CH2)3N": ["FC(CCCN)(F)F"],
    "CF3(CH2)4N": ["FC(CCCCN)(F)F"],
    "CF3(CH2)5N": ["FC(CCCCCN)(F)F"],
    "CHF2CH2N": ["FC(CN)([H])F"],
    "CHF2(CH2)2N": ["FC(CCN)([H])F"],
    "CHF2(CH2)3N": ["FC(CCCN)([H])F"],
    "CHF2(CH2)4N": ["FC(CCCCN)([H])F"],
    "CHF2(CH2)5N": ["FC(CCCCCN)([H])F"],
    "CH2FCH2N": ["FC(CN)([H])[H]"],
    "CH2F(CH2)2N": ["FC(CCN)([H])[H]"],
    "CH2F(CH2)3N": ["FC(CCCN)([H])[H]"],
    "CH2F(CH2)4N": ["FC(CCCCN)([H])[H]"],
    "CH2F(CH2)5N": ["FC(CCCCCN)([H])[H]"],
    "CCF2CN": ["FC(CN)(F)C"],
    "CCF2CCN": ["FC(CCN)(F)C", "FC(F)(C1)C1N"],
    "CCF2CCCN": ["FC(CCCN)(F)C", "FC(F)(CC1)C1N"],
    "CCF2CCCCN": ["FC(CCCCN)(F)C", "FC(F)(CCC1)C1N"],
    "CCF2CCCCCN": ["FC(CCCCCN)(F)C", "FC(F)(CCCC1)C1N"],
    "CCFHCN": ["FC(CN)([H])C"],
    "CCFHCCN": ["FC(CCN)([H])C"],
    "CCFHCCCN": ["FC(CCCN)([H])C"],
    "CCFHCCCCN": ["FC(CCCCN)([H])C"],
    "CCFHCCCCCN": ["FC(CCCCCN)([H])C"],
    "CF3CH2COOH": ["FC(CC(O)=O)(F)F"],
    "CF3(CH2)2COOH": ["FC(CCC(O)=O)(F)F"],
    "CF3(CH2)3COOH": ["FC(CCCC(O)=O)(F)F"],
    "CF3(CH2)4COOH": ["FC(CCCCC(O)=O)(F)F"],
    "CF3(CH2)5COOH": ["FC(CCCCCC(O)=O)(F)F"],
    "CHF2CH2COOH": ["FC(CC(O)=O)([H])F"],
    "CHF2(CH2)2COOH": ["FC(CCC(O)=O)([H])F"],
    "CHF2(CH2)3COOH": ["FC(CCCC(O)=O)([H])F"],
    "CHF2(CH2)4COOH": ["FC(CCCCC(O)=O)([H])F"],
    "CHF2(CH2)5COOH": ["FC(CCCCCC(O)=O)([H])F"],
    "CH2FCH2COOH": ["FC(CC(O)=O)([H])[H]"],
    "CH2F(CH2)2COOH": ["FC(CCC(O)=O)([H])[H]"],
    "CH2F(CH2)3COOH": ["FC(CCCC(O)=O)([H])[H]"],
    "CH2F(CH2)4COOH": ["FC(CCCCC(O)=O)([H])[H]"],
    "CH2F(CH2)5COOH": ["FC(CCCCCC(O)=O)([H])[H]"],
    "CCF2CCOOH": ["FC(CC(O)=O)(F)C"],
    "CCF2CCCOOH": ["FC(CCC(O)=O)(F)C", "FC1(F)CC1C(O)=O"],
    "CCF2CCCCOOH": ["FC(CCCC(O)=O)(F)C", "FC1(F)CCC1C(O)=O"],
    "CCF2CCCCCOOH": ["FC(CCCCC(O)=O)(F)C", "FC1(F)CCCC1C(O)=O"],
    "CCF2CCCCCCOOH": ["FC(CCCCCC(O)=O)(F)C", "FC1(F)CCCCC1C(O)=O"],
    "CCFHCCOOH": ["FC(CC(O)=O)([H])C"],
    "CCFHCCCOOH": ["FC(CCC(O)=O)([H])C"],
    "CCFHCCCCOOH": ["FC(CCCC(O)=O)([H])C"],
    "CCFHCCCCCOOH": ["FC(CCCCC(O)=O)([H])C"],
    "CCFHCCCCCCOOH": ["FC(CCCCCC(O)=O)([H])C"]
}
