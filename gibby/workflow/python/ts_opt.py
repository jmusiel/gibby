import argparse
import pprint

pp = pprint.PrettyPrinter(indent=4)

import os
from ase.io import read
from ase.calculators.emt import EMT
from gibby.utils.ocp_calc_wrapper import OCPCalcWrapper, get_config_override
from gibby.utils.ase_utils import get_fmax, get_hessian

from sella import Sella, Constraints


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--neb_path",
        type=str,
        default="/home/jovyan/shared-scratch/Brook/neb_stuff/dft_trajs_for_release/dissociations/dissociation_ood_22_8962_11_111-3_neb1.0.traj",
        # default="/home/jovyan/shared-scratch/Brook/neb_stuff/dft_trajs_for_release/dissociations",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/jovyan/shared-scratch/joe/personal_checkpoints/ocp_public_3_3_2023/EqV2/eq2_153M_ec4_allmd.pt"
        # default=None,
    )
    parser.add_argument(
        "--cpu",
        type=str,
        default="n",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sella_results",
    )
    parser.add_argument(
        "--scale_file",
        type=str,
        default="/home/jovyan/working/ocp/configs/s2ef/all/gemnet/scaling_factors/gemnet-dT.json",
    )
    return parser


def main(config):
    pp.pprint(config)

    # os.makedirs(config['output_dir'], exist_ok=True)

    config_override = get_config_override(
        config["checkpoint"], scale_file_path=config["scale_file"]
    )

    if config["checkpoint"] is not None:
        calc = OCPCalcWrapper(
            checkpoint_path=config["checkpoint"],
            cpu=config["cpu"] == "y",
            config_overrides=config_override,
        )
    else:
        calc = EMT()

    neb_files_list = []
    if config["neb_path"].endswith(".traj"):
        neb_files_list = [config["neb_path"]]
    else:
        neb_files_list = [
            os.path.join(config["neb_path"], file)
            for file in os.listdir(config["neb_path"])
            if file.endswith(".traj")
        ]
        print(os.listdir(config["neb_path"]))

    for filepath in neb_files_list:
        neb_traj = read(filepath, index="-10:")
        max_energy = neb_traj[1].get_potential_energy()
        ts_atoms = neb_traj[1]
        index = 0
        for i, atoms in enumerate(neb_traj[1:-1]):
            atoms.calc = calc
            print(
                f"atoms {i} {atoms.symbols} calc: {atoms.get_potential_energy():.4f}, fmax: {get_fmax(atoms.get_forces()):.4f}"
            )
            if atoms.get_potential_energy() > max_energy:
                max_energy = atoms.get_potential_energy()
                ts_atoms = atoms
                index = i
        print(
            f"max energy {max_energy} index {index}, fmax: {get_fmax(ts_atoms.get_forces())}"
        )

        cons = Constraints(ts_atoms)
        for atom in ts_atoms:
            if atom.index in ts_atoms.constraints[0].index:
                cons.fix_translation(atom.index)

        dyn = Sella(
            ts_atoms,
            constraints=cons,
            trajectory=None,
            hessian_function=None,
            eta=5e-5,
            delta0=1.3e-4,
        )
        dyn.run(1e-3, 1000)
        print(
            f"max energy {ts_atoms.get_potential_energy():.4f}, fmax: {get_fmax(ts_atoms.get_forces()):.4f}"
        )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    config = vars(args)
    main(config)
    print("done")
