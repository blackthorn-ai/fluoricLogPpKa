from collections import deque

from rdkit import Chem
from rdkit.Chem import rdchem

def has_numbers(inputString: str):
    """
    Checks if there is a number in the string.

    Args:
        - inputString (str): Text, that needs to be checked. Example: "acb1"
    
    Returns:
        - (bool): True if there is a number in the function, False if not.
    """
    return any(char.isdigit() for char in inputString)

def cycles_amount(mol):
    """
        Calculate amount of cycles in the molecule.
        
        Args:
            - mol: rdkit molecule.
            
        Returns:
            - amount_of_cycles (int): amount of rings in the molecule.
        """
    sssr = Chem.GetSSSR(mol)
    num_rings = len(sssr)
    return num_rings

def find_the_furthest_atom(mol: rdchem.Mol, 
                           atom_id: int, 
                           atoms_not_to_visit: list = []):
    """
    Find the furthest atom from a given atom ID in a molecule, excluding specified atoms.

    This function performs a BFS starting from the provided atom ID to find the furthest 
    atom in the molecule. It iterates through the neighbors of each atom, excluding hydrogen atoms and 
    those specified in the 'atoms_not_to_visit' list, and tracks the distance from the starting atom. 
    The search continues until no more unvisited neighbors remain, and the furthest atom and its distance 
    from the starting atom are returned.

    Args:
        mol (rdchem.Mol): RDKit molecule.
        atom_id (int): Index of the starting atom.
        atoms_not_to_visit (list, optional): List of atom indices to exclude from the search. Defaults to [].

    Returns:
        tuple: Tuple containing the index of the furthest atom and its distance from the starting atom.
    """
    queue = deque([(atom_id, 0)])

    visited = set()
    
    while queue:
        current_atom, distance = queue.popleft()
        
        visited.add(current_atom)
        
        neighbors = []
        for atom in mol.GetAtomWithIdx(current_atom).GetNeighbors():
            if atom.GetSymbol().lower() == 'h':
                continue
            if atom.GetIdx() in atoms_not_to_visit:
                continue
            neighbors.append(atom.GetIdx())
        
        for neighbor in neighbors:
            if neighbor not in visited:
                queue.append((neighbor, distance + 1))
    
    return current_atom, distance

def find_the_closest_atom_in_ring(mol: rdchem.Mol, 
                                  atom_id: int, 
                                  atoms_not_to_visit: list = []):
    """
    Find the closest atom within a ring from a given atom ID in a molecule, excluding specified atoms.

    This function performs BFS starting from the provided atom ID to find the closest 
    atom within a ring in the molecule. It iterates through the neighbors of each atom, excluding hydrogen 
    atoms and those specified in the 'atoms_not_to_visit' list, until a ring atom is encountered. 
    The search stops at the first encountered ring atom, and its index along with the index of the previous 
    atom in the search path are returned.

    Args:
        mol (rdchem.Mol): RDKit molecule.
        atom_id (int): Index of the starting atom.
        atoms_not_to_visit (list, optional): List of atom indices to exclude from the search. Defaults to [].

    Returns:
        tuple: Tuple containing the index of the closest ring atom and the index of the previous atom in the search path.
    """
    queue = deque([(atom_id, -1)])

    visited = set()
    
    while queue:
        current_atom, previous_atom = queue.popleft()
        if mol.GetAtomWithIdx(current_atom).IsInRing():
            break
        visited.add(current_atom)
        
        neighbors = []
        for atom in mol.GetAtomWithIdx(current_atom).GetNeighbors():
            if atom.GetSymbol().lower() == 'h':
                continue
            if atom.GetIdx() in atoms_not_to_visit:
                continue
            neighbors.append(atom.GetIdx())
        
        for neighbor in neighbors:
            if neighbor not in visited:
                queue.append((neighbor, current_atom))
    
    return current_atom, previous_atom

def find_all_atoms_from(mol: rdchem.Mol, 
                        from_atom_id: int, 
                        atoms_not_to_visit: list = []):
    """
    Find all atoms reachable from a given starting atom ID in a molecule, excluding specified atoms.

    This function performs BFS starting from the provided starting atom ID to find 
    all reachable atoms in the molecule. It iterates through the neighbors of each atom, excluding hydrogen 
    atoms and those specified in the 'atoms_not_to_visit' list, and adds unvisited neighbors to the queue 
    for further exploration. The search continues until all reachable atoms are visited, and a set containing 
    the indices of all visited atoms is returned.

    Args:
        mol (rdchem.Mol): RDKit molecule.
        from_atom_id (int): Index of the starting atom.
        atoms_not_to_visit (list, optional): List of atom indices to exclude from the search. Defaults to [].

    Returns:
        set: Set containing the indices of all atoms reachable from the starting atom.
    """
    queue = deque([(from_atom_id, -1)])

    visited = set()
    
    while queue:
        current_atom, previous_atom = queue.popleft()

        visited.add(current_atom)
        
        neighbors = []
        for atom in mol.GetAtomWithIdx(current_atom).GetNeighbors():
            if atom.GetSymbol().lower() == 'h':
                continue
            if atom.GetIdx() in atoms_not_to_visit:
                continue
            neighbors.append(atom.GetIdx())
        
        for neighbor in neighbors:
            if neighbor not in visited:
                queue.append((neighbor, current_atom))
    
    return visited
