from rdkit import Chem

from fluoriclogppka.ml_part.constants import Identificator, Target

import fluoriclogppka.ml_part.services.utils_pKa as utils_pKa
import fluoriclogppka.ml_part.services.utils_logP as utils_logP

def obtain_identificator(SMILES: str,
                         target_value: Target):
    mol = Chem.MolFromSmiles(SMILES)
    mol = Chem.AddHs(mol)
    
    if target_value == Target.pKa:
        identificator = utils_pKa.calculate_identificator(mol)
    elif target_value == Target.logP:
        identificator = utils_logP.calculate_identificator(mol)

    return identificator