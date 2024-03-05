from rdkit import Chem
from rdkit.Chem import Descriptors

from fluoriclogppka.ml_part.constants import Identificator
from fluoriclogppka.ml_part.constants import ALL_SUBMOLS

def amount_of_hydrogen_in_neighbors(mol, atom_idx):
    """
    Calculate the number of hydrogen atoms in the neighbors of a given atom.

    Args:
        mol (rdchem.Mol): RDKit molecule.
        atom_idx (int): Index of the target atom.

    Returns:
        int: Number of hydrogen atoms in the neighboring atoms.
    """
    atom = mol.GetAtomWithIdx(atom_idx)

    neighbors = atom.GetNeighbors()
    
    amount_of_hydrogen = 0
    
    for neighbor in neighbors:
        if neighbor.GetSymbol() == 'H':
            amount_of_hydrogen += 1

    return amount_of_hydrogen

def calculate_identificator(mol) -> Identificator:
    """
    Identify the type of functional group present in the molecule - COOH, NH or NH2.

    Returns:
        Identificator: Type of functional group.

    Raises:
        TypeError: If the molecule doesn't match any expected functional group.
    """
    carboxile_submol = Chem.MolFromSmiles('CC=O')
    nitro_amine_submol = Chem.MolFromSmiles('CN')

    carboxile_matches = mol.GetSubstructMatches(carboxile_submol)
    nitro_amine_matches = mol.GetSubstructMatches(nitro_amine_submol)

    if len(carboxile_matches) > 0:
        return Identificator.carboxilic_acid
    
    n_atom_index = nitro_amine_matches[0][1]
    amount_of_hydrogen = amount_of_hydrogen_in_neighbors(mol, n_atom_index)

    if amount_of_hydrogen == 2:
        return Identificator.primary_amine
    elif amount_of_hydrogen == 1:
        return Identificator.secondary_amine
    
    raise TypeError("Wrong molecule")

def calculate_linear_path_f_to_fg(smiles):
    """
    Calculate amount of linear paths from fluorine (F) to the functional group (FG) based on the provided SMILES.

    Args:
        smiles (str): SMILES representation of the molecule.

    Returns:
        int: The total number of matches found between substructures and the molecule.
    """    
    mol = Chem.MolFromSmiles(smiles, sanitize=True) 
    mol = Chem.AddHs(mol) 
    
    total_matches = 0
    for _, sub_SMILES_array in ALL_SUBMOLS.items():
        for sub_SMILES in sub_SMILES_array:
            
            submol = Chem.MolFromSmiles(sub_SMILES, sanitize=False)
            matches = mol.GetSubstructMatches(submol)

            if len(matches) > 0:
                total_matches += len(matches)
    
    return total_matches

def calculate_molecular_weight(SMILES):
    """
    Calculate the molecular weight of the molecule based on the provided SMILES.

    Args:
        SMILES (str): SMILES representation of the molecule.

    Returns:
        float: The calculated molecular weight.
    """
    mol = Chem.MolFromSmiles(SMILES)
    mol = Chem.AddHs(mol)
    
    molecular_weight = Descriptors.MolWt(mol)
    
    return molecular_weight
