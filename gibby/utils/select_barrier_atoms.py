from ase.calculators.singlepoint import SinglePointCalculator
from gibby.utils.ocp_calc_wrapper import OCPCalcWrapper, get_config_override
from gibby.utils.ase_utils import get_hessian
import os
from tqdm import tqdm
from ase.io import read
import pandas as pd
from collections import defaultdict
import pickle

import numpy as np
from gibby.utils.ase_utils import get_fmax
import argparse
import pprint
pp = pprint.PrettyPrinter(indent=4)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--neb_path",
        type=str,
        nargs="+",
        default=[  # ml trajectories
            "/home/jovyan/shared-scratch/Brook/neb_stuff/checkpoint/brookwander/neb_results/val_sps_w_dft_rx_init_final/ml_val_w_dft_sp_disoc_202310/eq2_153M_ec4_allmd",
            "/home/jovyan/shared-scratch/Brook/neb_stuff/checkpoint/brookwander/neb_results/val_sps_w_dft_rx_init_final/ml_val_w_dft_sp_transfer_202310/eq2_153M_ec4_allmd",
            "/home/jovyan/shared-scratch/Brook/neb_stuff/checkpoint/brookwander/neb_results/val_sps_w_dft_rx_init_final/ml_val_w_dft_sp_desorp_20231026/eq2_153M_ec4_allmd",
        ],
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/jovyan/shared-scratch/joe/personal_checkpoints/ocp_public_3_3_2023/EqV2/eq2_153M_ec4_allmd.pt"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/jovyan/working/repos/gibby/gibby/workflow/python/barrier_atoms_pickle",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="ml_neb",
    )
    parser.add_argument(
        "--get_hessians",
        type=str,
        default="y",
        help="y/n, default y",
    )
    parser.add_argument(
        "--hessian_delta",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--cpu",
        type=str,
        default="n",
    )
    parser.add_argument(
        "--debug",
        type=int,
        default=None,
        help="debugging flag, default:None, if set to an integer, how many to debug",
    )
    parser.add_argument(
        "--tags_file",
        type=str,
        default="/home/jovyan/shared-scratch/Brook/all_ml_rx_inputs/cattsunami_tag_lookup.pkl",
    )
    return parser

def main(config):
    pp.pprint(config)

    neb_files_list = []
    if isinstance(config["neb_path"], str):
        neb_files_list.append(config["neb_path"])
    else:
        for path in config["neb_path"]:
            if path.endswith(".traj"):
                neb_files_list.append(path)
            else:
                neb_files_list += [
                    os.path.join(path, file)
                    for file in os.listdir(path)
                    if file.endswith(".traj")
                ]

    if config["debug"] is not None:
        neb_files_list = neb_files_list[: config["debug"]]

    if config["tags_file"] is not None:
        with open(config["tags_file"], "rb") as f:
            tags_lookup = pickle.load(f)

    config_override = get_config_override(
        config["checkpoint"], scale_file_path="",
    )
    calc = OCPCalcWrapper(
        checkpoint_path=config["checkpoint"],
        cpu=config["cpu"] == "y",
        config_overrides=config_override,
    )

    results_data = defaultdict(list)

    for j, filepath in tqdm(enumerate(neb_files_list), total=len(neb_files_list)):
        neb_traj = read(filepath, index="-10:")
        (
            ts_atoms,
            neb_list,
            max_neb_energy,
            max_neb_forces,
            max_neb_fmax,
            index,
            barrier,
        ) = select_barrier_atoms(neb_traj, calc=calc)

        name = filepath.split("/")[-1].split(".")[0]
        results_data["name"].append(name)
        results_data["neb_list"].append(neb_list)
        results_data["NEB_barrier_ML_energy"].append(max_neb_energy)
        results_data["NEB_barrier_ML_forces"].append(max_neb_forces)
        results_data["NEB_barrier_ML_fmax"].append(max_neb_fmax)
        results_data["atoms"].append(ts_atoms)
        results_data["index"].append(index)
        results_data["barrier"].append(barrier)

        if config["get_hessians"] == "y":
            tags = None
            if config["tags_file"] is not None:
                tags = tags_lookup[name.replace("-k_now","")]

            hessian_atoms = ts_atoms.copy()
            hessian_atoms.calc = calc
            hessian = get_hessian(hessian_atoms, hessian_delta=config["hessian_delta"], tags=tags)
            results_data["hessian"].append(hessian)

    if config["output_dir"] is not None:
        os.makedirs(config["output_dir"], exist_ok=True)
        df = pd.DataFrame(results_data)
        df.to_pickle(os.path.join(config["output_dir"], f"{config['output_name']}.pkl"))


def select_barrier_atoms(neb_traj, calc=None):
    """
    Select the atoms with the highest energy in the final relaxed version of the NEB trajectory
    args: 
        neb_traj: list of atoms objects, the final 10 relaxed atoms after the NEB relaxation, should include one highest barrier atom

    returns:
        ts_atoms: the atoms object with the highest energy in the final relaxed NEB trajectory
        neb_list: list of atoms objects, the final 10 relaxed atoms after the NEB relaxation
    """
    ts_atoms = None
    neb_list = []
    max_neb_energy = -np.inf
    max_neb_forces = None
    max_neb_fmax = None
    index = None
    ts_opt_energy = None
    ts_opt_forces = None
    ts_opt_fmax = None
    barrier = False
    for i, atoms in enumerate(neb_traj):
        neb_list.append(atoms)  # add to neb_list the DFT singlepoint results
        atoms_copy = atoms.copy()
        if calc is not None:
            atoms_copy.calc = calc
        else:
            atoms_copy.calc = atoms.get_calculator()
        neb_energy = atoms_copy.get_potential_energy()
        neb_forces = atoms_copy.get_forces()
        sp_calc = SinglePointCalculator(
            atoms=atoms_copy, energy=float(neb_energy), forces=neb_forces
        )
        sp_calc.implemented_properties = ["energy", "forces"]
        atoms_copy.calc = sp_calc
        # print(
        #     f"NEB atoms {i} {atoms_copy.symbols} DFT calc: {neb_energy:.4f}, fmax: {get_fmax(neb_forces):.4f}"
        # )
        if neb_energy > max_neb_energy or ts_atoms is None:
            if not i == 0 and not i == len(neb_traj) - 1:
                barrier = True
            max_neb_energy = neb_energy
            max_neb_forces = neb_forces
            max_neb_fmax = get_fmax(max_neb_forces)
            ts_atoms = atoms_copy
            index = i
    # print(
    #     f"ML calc max energy: {max_neb_energy} index {index}, fmax: {max_neb_fmax}, barrier: {barrier}"
    # )


    return (
        ts_atoms,
        neb_list,
        max_neb_energy,
        max_neb_forces,
        max_neb_fmax,
        index,
        barrier,
    )

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    config = vars(args)
    main(config)
    print("done")