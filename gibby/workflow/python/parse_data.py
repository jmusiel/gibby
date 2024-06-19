"""
Create primary input file to all workflows
"""

from ase.io import read
from gibby.utils.process_data import (
    parse_finite_difference_files,
    is_edge_list_respected,
    is_adsorbate_adsorbed_and_unreacted,
)
import pandas as pd
import pickle
from tqdm import tqdm
from glob import glob
import argparse


def set_tags(row):
    """
    Set the tags of the atoms object in the row.
    """
    row["atoms"].set_tags(row.tags)


def get_initial_frame(row, path):
    """
    Parse the traj of the initial frame to add to the dataframe.

    Args:
        row (pd.Series): the row of the dataframe
        path (str): the path to the directory containing the traj file
    """
    initial = read(f"{path}/{row.random_id}.traj", "0")
    initial.set_tags(row.tags)
    return initial


def check_edges(row):
    """
    Wrapper for the edge list check.
    """
    return is_edge_list_respected(row.initial_atoms, row.atoms)


def no_anomaly(row):
    return row.correct_edges and row.unreacted_and_adsorbed


def get_tags(randid):
    return adslab_tags[randid]


def get_adsorbate_info(smile):
    if smile in smile_lookup:
        return smile_lookup[smile]
    else:
        return None, None


def get_meta(idx, meta):
    meta_ = meta[idx]
    if type(meta_) == tuple:
        return meta_[0], meta_[1]
    else:
        return "transition_state", idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--traj_path", type=str, help="Path to the oc20 trajectory files"
    )
    parser.add_argument("--data_path", type=str, help="Path to the VASP output files")
    parser.add_argument("--metadata_path", type=str, help="Path to the metadata file")
    parser.add_argument(
        "--tag_mapping_path", type=str, help="Path to the tag mapping file"
    )
    parser.add_argument("--output_path", type=str, help="Path to the output file")
    parser.add_argument("--adsorbate_path", type=str, help="Path to the adsorbate file")
    args = parser.parse_args()

    # Load metadata
    metadata = pd.read_csv(args.metadata_path)
    with open(args.tag_mapping_path, "rb") as f:
        adslab_tags = pickle.load(f)
    with open(args.adsorbate_path, "rb") as f:
        adsorbates = pickle.load(f)
    smile_lookup = {}
    for adsorbate in adsorbates:
        smile_lookup[adsorbates[adsorbate][1]] = (
            len(adsorbates[adsorbate][0]),
            adsorbates[adsorbate][4],
        )

    entries = []
    for folder in tqdm(glob(f"{args.data_path}/**")):
        idx = folder.split("/")[-1]
        completed, hessian, atoms_list, atoms = parse_finite_difference_files(folder)
        entry = {
            "mapping_idx": int(idx),
            "completed": completed,
            "hessian": hessian,
            "atoms": atoms,
            "atoms_list": atoms_list,
        }
        entries.append(entry)

    df = pd.DataFrame(entries)

    # Merge the metadata
    df = df.merge(metadata, left_on="mapping_idx", right_on="mapping_idx")

    # Proces data to look for anomalies
    df["tags"] = df.random_id.apply(get_tags)
    df["adsorbate_len"], df["adsorbate_edges"] = zip(
        *df.adsorbate.apply(get_adsorbate_info)
    )
    df["initial_atoms"] = df.apply(get_initial_frame, axis=1)
    df["correct_edges"] = df.apply(check_edges, axis=1)
    df["unreacted_and_adsorbed"] = df.atoms.apply(is_adsorbate_adsorbed_and_unreacted)
    df["no_anomaly"] = df.apply(no_anomaly, axis=1)

    df.to_pickle(args.output_path)
