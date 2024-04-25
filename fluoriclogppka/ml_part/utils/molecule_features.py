from rdkit import Chem

from fluoriclogppka.ml_part.constants import Target

import fluoriclogppka.ml_part.services.utils_pKa as utils_pKa
import fluoriclogppka.ml_part.services.utils_logP as utils_logP

def obtain_identificator(SMILES: str,
                         target_value: Target):
    """
    Obtains the identificator of a molecule based on its SMILES representation and target value.

    Args:
        SMILES (str): The SMILES string representing the molecule.
        target_value (Target): The target property to predict (pKa or logP).

    Returns:
        identificator (Identificator): The identificator of the molecule.
    """
    mol = Chem.MolFromSmiles(SMILES)
    mol = Chem.AddHs(mol)
    
    if target_value == Target.pKa:
        identificator = utils_pKa.calculate_identificator(mol)
    elif target_value == Target.logP:
        identificator = utils_logP.calculate_identificator(mol)

    return identificator