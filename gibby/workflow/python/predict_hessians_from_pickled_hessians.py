import argparse
import pprint

pp = pprint.PrettyPrinter(indent=4)

import pickle
import numpy as np
import os
from jlaunch.hessian_smoothness.utils.make_lmdb_class import MakeLmdb
import pandas as pd
from tqdm import tqdm
import json
from gibby.utils.ocp_calc_wrapper import OCPCalcWrapper, get_config_override
from ase.optimize import BFGS
from ase.vibrations import Vibrations
import sys
import shutil
import glob
from gibby.utils.ase_utils import get_hessian, get_analytical_hessian


import warnings

warnings.filterwarnings("ignore", message="__floordiv__ is deprecated")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pickle_file",
        type=str,
        nargs="+",
        default=[
            "/home/jovyan/shared-scratch/Brook/gibbs_proj/brooks_analysis/val_dataset_ediff6.pkl",
            "/home/jovyan/shared-scratch/Brook/gibbs_proj/brooks_analysis/ts_dataset_ediff6.pkl",
        ],
    )
    parser.add_argument(
        "--tag_mapping",
        type=str,
        default="/home/jovyan/shared-scratch/adeesh/splits_02_07/mappings/adslab_tags_full.pkl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/jovyan/working/repos/launch/jlaunch/hessian_smoothness/predict_hessians_ase/hessians_dataframes_predict_output",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        nargs="+",
        default=[
            "/home/jovyan/working/repos/launch/jlaunch/hessian_smoothness/finetuning/train_outputs/eqv2_finetuning_test_val/checkpoints/2024-06-24-20-05-20-lr5e6-20epochs/checkpoint_1.pt",  # 5e-6 running for 20 epochs, checkpointing after each
            "/home/jovyan/working/repos/launch/jlaunch/hessian_smoothness/finetuning/train_outputs/eqv2_finetuning_test_val/checkpoints/2024-06-24-20-05-20-lr5e6-20epochs/checkpoint_2.pt",  # 5e-6 running for 20 epochs, checkpointing after each
            "/home/jovyan/working/repos/launch/jlaunch/hessian_smoothness/finetuning/train_outputs/eqv2_finetuning_test_val/checkpoints/2024-06-24-20-05-20-lr5e6-20epochs/checkpoint_3.pt",  # 5e-6 running for 20 epochs, checkpointing after each
            "/home/jovyan/working/repos/launch/jlaunch/hessian_smoothness/finetuning/train_outputs/eqv2_finetuning_test_val/checkpoints/2024-06-24-20-05-20-lr5e6-20epochs/checkpoint_4.pt",  # 5e-6 running for 20 epochs, checkpointing after each
            "/home/jovyan/working/repos/launch/jlaunch/hessian_smoothness/finetuning/train_outputs/eqv2_finetuning_test_val/checkpoints/2024-06-24-20-05-20-lr5e6-20epochs/checkpoint_5.pt",  # 5e-6 running for 20 epochs, checkpointing after each
            # "/home/jovyan/working/repos/launch/jlaunch/hessian_smoothness/finetuning/train_outputs/eqv2_finetuning_test_val/checkpoints/2024-06-19-16-34-08-lr1e5/checkpoint.pt", # stopped at epoch 1
            # "/home/jovyan/working/repos/launch/jlaunch/hessian_smoothness/finetuning/train_outputs/eqv2_finetuning_test_val/checkpoints/2024-06-19-16-38-24-lr1e6/checkpoint.pt", # stopped at epoch 1
            # "/home/jovyan/working/repos/launch/jlaunch/hessian_smoothness/finetuning/train_outputs/eqv2_finetuning_test_val/checkpoints/2024-06-19-16-38-24-lr1e7/checkpoint.pt", # stopped at epoch 1
            # "/home/jovyan/working/repos/launch/jlaunch/hessian_smoothness/finetuning/train_outputs/eqv2_finetuning_test_val/checkpoints/2024-06-19-16-38-24-lr5e6/checkpoint.pt", # stopped at epoch 1
            # "/home/jovyan/working/repos/launch/jlaunch/hessian_smoothness/finetuning/train_outputs/eqv2_finetuning_test_val/checkpoints/2024-06-13-19-20-32-lr1e6/checkpoint.pt", # stopped at epoch 5
            # "/home/jovyan/working/repos/launch/jlaunch/hessian_smoothness/finetuning/train_outputs/eqv2_finetuning_test_val/checkpoints/2024-06-13-19-20-32-lr1e7/checkpoint.pt", # stopped at epoch 5
            # "/home/jovyan/working/repos/launch/jlaunch/hessian_smoothness/finetuning/train_outputs/eqv2_finetuning_test_val/checkpoints/2024-06-13-19-20-32-lr5e6/checkpoint.pt", # stopped at epoch 5
            # "/home/jovyan/working/repos/launch/jlaunch/hessian_smoothness/finetuning/train_outputs/eqv2_finetuning_test_val/checkpoints/2024-06-13-19-20-32-lr5e7/checkpoint.pt", # stopped at epoch 5
            # "/home/jovyan/working/repos/launch/jlaunch/hessian_smoothness/finetuning/train_outputs/eqv2_finetuning_test/checkpoints/2024-06-06-21-20-00/best_checkpoint.pt", # this one got labeled as best_checkpoint
            # "/home/jovyan/shared-scratch/joe/personal_checkpoints/ocp_public_3_3_2023/EqV2/eq2_153M_ec4_allmd.pt",
            # "/home/jovyan/working/data/checkpoints/schnet/schnet_all_large.pt",
            # "/home/jovyan/shared-scratch/joe/personal_checkpoints/ocp_public_3_3_2023/EqV2/eq2_31M_ec4_allmd.pt",
            # "/home/jovyan/shared-scratch/joe/personal_checkpoints/gemnet_dt_gradient_forces_perspective_paper/gemnet_dt_gradf_best_checkpoint.pt", # gemnet dt gradient forces
            # "/home/jovyan/shared-scratch/joe/personal_checkpoints/gemnet_dt_all_download_2022_09_21/gemnet_t_direct_h512_all.pt", # gemnet dt all (default for comparison)
            # "/home/jovyan/shared-scratch/joe/personal_checkpoints/ocp_public_3_3_2023/gemnet_oc/gemnet_oc_large_s2ef_all_md.pt",
        ],
    )
    parser.add_argument(
        "--scale_file",
        type=str,
        # default=None,
        default="/home/jovyan/working/ocp/configs/s2ef/all/gemnet/scaling_factors/gemnet-dT.json",
    )
    parser.add_argument(
        "--cpu",
        type=str,
        default="n",
    )
    parser.add_argument(
        "--output_pickle_name_mod",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--checkpoint_model_class_override",
        type=str,
        default=None,
        help="optional, used to change the model class used to load the checkpoint to a custom one",
    )
    parser.add_argument(
        "--use_analytical_hessian",
        type=str,
        default="n",
    )
    parser.add_argument(
        "--hessian_delta",
        type=float,
        default=0.01,
    )
    return parser


def main(config):
    pp.pprint(config)

    with open(config["tag_mapping"], "rb") as f:
        tag_mapping = pickle.load(f)

    for pickle_file in config["pickle_file"]:
        pickle_name = pickle_file.split("/")[-1].split(".")[0]
        output_dir = config["output_dir"]
        output_dir = os.path.join(output_dir, pickle_name)
        os.makedirs(output_dir, exist_ok=True)
        pickle_search_files = glob.glob(
            os.path.join(output_dir, f"{pickle_name}_rev*.pkl")
        )
        if len(pickle_search_files) > 0:
            rev = 0
            for name in pickle_search_files:
                rev = max(rev, int(name.split("_rev")[1].split(".")[0]))
            current_rev_pickle_name = f"{pickle_name}_rev{rev}"
            pickle_file = os.path.join(output_dir, current_rev_pickle_name + ".pkl")
            pickle_name = current_rev_pickle_name
        if "_rev" in pickle_name:
            new_pickle_name = f"{pickle_name.split('_rev')[0]}_rev{int(pickle_name.split('_rev')[1])+1}"
        else:
            new_pickle_name = f"{pickle_name}_rev1"

        if config["output_pickle_name_mod"] is not None:
            new_pickle_name = new_pickle_name + config["output_pickle_name_mod"]
            new_pickle_name = new_pickle_name.replace("rev", "modded")

        with open(pickle_file, "rb") as f:
            df = pickle.load(f)

        for checkpoint_path in config["checkpoint_path"]:
            config_override = get_config_override(
                checkpoint_path=checkpoint_path,
                scale_file_path=config["scale_file"],
                checkpoint_model_class_override=config[
                    "checkpoint_model_class_override"
                ],
            )

            calc = OCPCalcWrapper(
                checkpoint_path=checkpoint_path,
                cpu=config["cpu"] == "y",
                config_overrides=config_override,
            )

            calc_name = checkpoint_path.split("/")[-1].split(".")[0]
            if calc_name == "best_checkpoint" or calc_name == "checkpoint":
                calc_name = checkpoint_path.split("/")[-2]
            if config["checkpoint_model_class_override"] is not None:
                calc_name = f"{config['checkpoint_model_class_override']}_{calc_name}"
            series_list = []
            for index, row in tqdm(
                df.iterrows(),
                total=len(df),
                desc=f"Iterating over {pickle_name}.pkl, for model: {calc_name} (analytical? {config['use_analytical_hessian'] == 'y'})",
            ):
                new_data = series_list.append(
                    add_hessian_columns(
                        row,
                        calc,
                        calc_name,
                        analytical=config["use_analytical_hessian"] == "y",
                        hessian_delta=config["hessian_delta"],
                    )
                )
            new_data = pd.concat(series_list, axis=1).T
            cols_to_drop = df.columns.intersection(new_data.columns)
            df = df.drop(columns=cols_to_drop)
            df = df.join(new_data)

        df.to_pickle(os.path.join(output_dir, new_pickle_name + ".pkl"))


def add_hessian_columns(row, calc, calc_name, analytical=False, hessian_delta=0.01):
    return_list = []
    column_names = []
    keys = [
        "unrelaxed_hessian",
        # 'relaxed_hessian',
    ]
    if row["success"]:
        for key in keys:
            atoms = row["atoms"].copy()
            atoms.calc = calc
            if key == "relaxed_hessian":
                with suppress_stdout():
                    BFGS(atoms).run(fmax=0.03, steps=1000)
                column_names.append(f"{calc_name}_relaxed")
            else:
                column_names.append(f"{calc_name}")
            if analytical:  # get analytical hessian
                hessian = get_analytical_hessian(atoms)
            else:  # get numerical hessian
                hessian = get_hessian(atoms, hessian_delta=hessian_delta)
            return_list.append(hessian)
    else:
        for key in keys:
            if key == "relaxed_hessian":
                column_names.append(f"{calc_name}_relaxed")
            else:
                column_names.append(f"{calc_name}")
            return_list.append(None)

    return pd.Series(return_list, index=column_names)


class suppress_stdout:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    config = vars(args)
    main(config)
    print("done")
