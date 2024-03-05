from rdkit import Chem
from rdkit.Chem import Descriptors

from fluoriclogppka.ml_part.constants import Identificator
from fluoriclogppka.ml_part.constants import ALL_SUBMOLS
from fluoriclogppka.ml_part.services.utils import find_the_furthest_atom, find_all_atoms_from

def calculate_identificator(mol) -> Identificator:
    """
    Identify the type of functional group present in the molecule - COOH, NH or NH2.

    Returns:
        Identificator: Type of functional group.

    Raises:
        TypeError: If the molecule doesn't match any expected functional group.
    """
    ring_submol = Chem.MolFromSmiles("C1=CC=CC=C1")
    COOH_submol = Chem.MolFromSmiles("C(=O)")
    NH_submol = Chem.MolFromSmiles("N")
    secondary_amine_submol = Chem.MolFromSmiles("CN(C)C")

    ring_matches = mol.GetSubstructMatches(ring_submol)
    COOH_matches = mol.GetSubstructMatches(COOH_submol)
    NH_matches = mol.GetSubstructMatches(NH_submol)
    secondary_amine_matches = mol.GetSubstructMatches(secondary_amine_submol)

    furthest_atom_from_COOH, furthest_COOH_distance = find_the_furthest_atom(mol=mol,
                                                                             atom_id=COOH_matches[0][0],
                                                                             atoms_not_to_visit=ring_matches[0])
    
    furthest_atom_from_NH, furthest_NH_distance = find_the_furthest_atom(mol=mol,
                                                                         atom_id=NH_matches[0][0],
                                                                         atoms_not_to_visit=ring_matches[0])
    
    if furthest_NH_distance > furthest_COOH_distance:
        return Identificator.carboxilic_acid
    
    if len(secondary_amine_matches) > 0:
        return Identificator.secondary_amine
    else:
        return Identificator.primary_amine
    
def calculate_linear_path_f_to_fg(smiles: str,
                                  identificator: Identificator):
    """
    Calculate the amount of linear paths from fluorine (F) to the functional group (FG) based on the provided SMILES 
    and molecule identificator.

    This function constructs a molecule from the given SMILES representation and identifies the linear path 
    from fluorine to the specified functional group. It iterates over all substructures associated with the 
    functional group and counts the total number of matches found in the molecule, excluding certain substructures 
    based on the provided identificator.

    Args:
        smiles (str): SMILES representation of the molecule.
        identificator (Identificator): Identificator specifying the type of functional group.

    Returns:
        int: The total number of matches found between substructures and the molecule.
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=True) 
    mol = Chem.AddHs(mol) 
    
    total_matches = 0
    for _, sub_SMILES_array in ALL_SUBMOLS.items():
        for sub_SMILES in sub_SMILES_array:
            
            if identificator == Identificator.carboxilic_acid and "n" in sub_SMILES.lower():
                continue
            elif "amine" in identificator.name.lower() and "(O)=O" in sub_SMILES:
                continue

            sub_SMILES = sub_SMILES.replace("(O)=", "(N)=")
            
            submol = Chem.MolFromSmiles(sub_SMILES, sanitize=False)
            matches = mol.GetSubstructMatches(submol)

            total_matches += len(matches)
    
    return total_matches

def calculate_molecular_weight(SMILES: str,
                               identificator: Identificator):
    """
    Calculate the molecular weight of the molecule based on the provided SMILES and molecule identificator.

    This function calculates the molecular weight of the molecule represented by the given SMILES. 
    It first constructs the molecule from the SMILES, then identifies specific substructures relevant to 
    the provided identificator (such as carbonyl groups for carboxylic acids or amine groups for amines). 
    After removing unnecessary atoms, it sanitizes the molecule, replaces certain substructures, adds hydrogens, 
    and finally calculates the molecular weight.

    Args:
        SMILES (str): SMILES representation of the molecule.
        identificator (Identificator): Identificator specifying the type of functional group.

    Returns:
        float: The calculated molecular weight.
    
    Raises:
        TypeError: If the type of molecule is inappropriate for the given identificator.
    """
    mol = Chem.MolFromSmiles(SMILES)

    CO_submol = Chem.MolFromSmiles("C=O")
    NH_submol = Chem.MolFromSmiles("N")

    CO_matches = mol.GetSubstructMatches(CO_submol)
    NH_matches = mol.GetSubstructMatches(NH_submol)

    if len(NH_matches) == 0 or len(CO_matches) == 0:
        raise TypeError("Inappropriate type of the molecule")

    if identificator == Identificator.carboxilic_acid:
        atoms_to_remove = find_all_atoms_from(mol=mol, 
                                            from_atom_id=NH_matches[0][0], 
                                            atoms_not_to_visit=CO_matches[0])
    else:
        atoms_to_remove = find_all_atoms_from(mol=mol, 
                                            from_atom_id=CO_matches[0][0], 
                                            atoms_not_to_visit=NH_matches[0])
        
    rwmol = Chem.RWMol(mol) 
    for atom_idx in reversed(list(atoms_to_remove)):
        rwmol.RemoveAtom(atom_idx)
    mol = rwmol.GetMol()

    Chem.SanitizeMol(mol)

    mol = Chem.ReplaceSubstructs(mol, 
                                Chem.MolFromSmiles('C=O'), 
                                Chem.MolFromSmiles('C(=O)O'),
                                replaceAll=True)[0]
    mol = Chem.AddHs(mol)

    return Descriptors.MolWt(mol)
