"""
Functions to handle parsing VASP outputs to get all necessary data
from IBRION=5 calculations.
"""

import re
from ase.io import read
import numpy as np
import re
from ase.io import read
import numpy as np

from ase.data import atomic_numbers, covalent_radii
from itertools import combinations, product
import ase


def parse_finite_difference_files(path:str):
    """
    Read the OUTCAR file to get the energy, hessian, atoms, and atoms_list
    
    Args:
        path (str): Path to the directory containing the OUTCAR file

    Returns:
        (bool): Whether the calculation completed successfully
        (np.ndarray): Hessian matrix
        (list): List of atoms
        (ase.Atoms): the atoms object of the unperturbed system
    """
    try:
        with open(path + "/OUTCAR", "r") as f:
            outcar = f.read()
    except FileNotFoundError:
        return False, None, None, None

    completed = re.search("General timing and accounting informations for this job", outcar) is not None

    if not completed:
        return False, None, None, None

    # Get the Hessian
    idx = 0
    hessian = []
    with open(path + "/OUTCAR", "r") as f:
        lines = f.readlines()
    for line in lines:
        if line == " SECOND DERIVATIVES (NOT SYMMETRIZED)\n":
            break
        idx += 1
    complete = False        
    while not complete:
        line = lines[idx + 3]
        vals = line.split()[1:]
        if len(vals) == 0:
            complete = True
            break
        hessian.append(vals)
        idx += 1
    hessian = np.array(hessian, dtype=float)

    # Get the atoms
    atoms_list = read(path + "/OUTCAR", ":")

    return completed, hessian, atoms_list, atoms_list[0]

def is_adsorbate_adsorbed_and_unreacted(adsorbate_slab_config: ase.Atoms):
    """
    Check to see if the adsorbate is adsorbed on the surface.

    Args:
        adsorbate_slab_config (ase.Atoms): the combined adsorbate and slab configuration
            with adsorbate atoms tagged as 2s and surface atoms tagged as 1s.

    Returns:
        (bool): True if the adsorbate is adsorbed, False otherwise.
    """
    adsorbate_idxs = [idx for idx, tag in enumerate(adsorbate_slab_config.get_tags()) if tag == 2]
    surface_idxs = [idx for idx, tag in enumerate(adsorbate_slab_config.get_tags()) if tag != 2]
    elements = adsorbate_slab_config.get_chemical_symbols()
    all_combos = list(product(adsorbate_idxs, surface_idxs))
    adsorbed_and_connected = []
    for combo in all_combos:
        total_distance = adsorbate_slab_config.get_distance(
            combo[1], combo[0], mic=True
        )
        r1 = covalent_radii[atomic_numbers[elements[combo[0]]]]
        r2 = covalent_radii[atomic_numbers[elements[combo[1]]]]
        distance_ratio = total_distance / (r1 + r2)
        if distance_ratio < 1.25:
            adsorbed_and_connected.append(check_surface_atom_connectivity(combo[1], adsorbate_slab_config, [])[0])
    if len(adsorbed_and_connected) == 0:
        return False
    return all(adsorbed_and_connected)

def is_edge_list_respected(frame_initial: ase.Atoms, frame_final: list):
    """
    Check to see that the expected adsorbate-adsorbate edges are found and no additional
    edges exist between the adsorbate atoms.

    Args:
        frame (ase.Atoms): the atoms object for which edges will be checked.
            This must comply with ocp tagging conventions.
        edge_list (list[tuples]): The expected edges
    """
    adsorbate_initial = frame_initial[[idx for idx, tag in enumerate(frame_initial.get_tags()) if tag == 2]]
    adsorbate_final = frame_final[[idx for idx, tag in enumerate(frame_final.get_tags()) if tag == 2]]
    elements = adsorbate_initial.get_chemical_symbols()
    all_combos = list(combinations(range(len(adsorbate_initial)), 2))

    edge_list = []
    for combo in all_combos:
        total_distance = adsorbate_initial.get_distance(combo[0], combo[1], mic=True)
        r1 = covalent_radii[atomic_numbers[elements[combo[0]]]]
        r2 = covalent_radii[atomic_numbers[elements[combo[1]]]]
        distance_ratio = total_distance / (r1 + r2)
        if distance_ratio < 1.25:
            edge_list.append(combo)

    for combo in all_combos:
        total_distance = adsorbate_final.get_distance(combo[0], combo[1], mic=True)
        r1 = covalent_radii[atomic_numbers[elements[combo[0]]]]
        r2 = covalent_radii[atomic_numbers[elements[combo[1]]]]
        distance_ratio = total_distance / (r1 + r2)
        if combo in edge_list:
            if distance_ratio > 1.25:
                return False
        elif distance_ratio < 1.25:
            return False
    return True

def check_surface_atom_connectivity(surface_idx, adsorbate_slab_config, idxs_checked):

    """
    Check to see if the surface atom connected to the adsorbate atoms is connected the the
    slab atoms.

    Args:
        combo (tuple): the indices of the adsorbate and surface atoms.
        adsorbate_slab_config (ase.Atoms): the combined adsorbate and slab configuration
            with adsorbate atoms tagged as 2s, surface atoms tagged as 1s and subsurface
            atoms tagged 0.

    Returns:
        (bool): True if the surface atom connected to the adsorbate atom is coordinated
            with the subsurface atoms, False otherwise.
    """
    idxs_checked.append(surface_idx)
    slab_idxs = [idx for idx, tag in enumerate(adsorbate_slab_config.get_tags()) if tag != 2]
    tags = adsorbate_slab_config.get_tags()
    syms = adsorbate_slab_config.get_chemical_symbols()
    r1 = covalent_radii[atomic_numbers[syms[surface_idx]]]
    neighbors = []
    for idxy in slab_idxs:
        r2 = covalent_radii[atomic_numbers[syms[idxy]]]
        if adsorbate_slab_config.get_distance(surface_idx, idxy, mic=True) < 1.25 * (r1 + r2) and idxy not in idxs_checked:
            neighbors.append(idxy)
    if len(neighbors) == 0:
        return False, idxs_checked
    
    neighbor_bools = [tags[idx] == 0 for idx in neighbors]
    if any(neighbor_bools):
        return True, idxs_checked
    else:
        for idx in neighbors:
            if idx not in idxs_checked:
                return check_surface_atom_connectivity(idx, adsorbate_slab_config, idxs_checked)