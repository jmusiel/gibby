import argparse
import pprint

pp = pprint.PrettyPrinter(indent=4)

import os
from ase.io import read
from ase.calculators.emt import EMT
from gibby.utils.ocp_calc_wrapper import OCPCalcWrapper, get_config_override
from gibby.utils.ase_utils import get_fmax, get_hessian
from ase.calculators.singlepoint import SinglePointCalculator
import wandb
import pandas as pd
from tqdm import tqdm
import numpy as np
from datetime import datetime

from sella import Sella, Constraints


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--neb_path",
        type=str,
        nargs="+", 
        # default=[
        #     "/home/jovyan/shared-scratch/Brook/neb_stuff/dft_trajs_for_release/dissociations/dissociation_ood_22_8962_11_111-3_neb1.0.traj",
        #     "/home/jovyan/shared-scratch/Brook/neb_stuff/dft_trajs_for_release/dissociations/dissociation_id_611_9766_9_111-1_neb1.0.traj",
        #     "/home/jovyan/shared-scratch/Brook/neb_stuff/dft_trajs_for_release/dissociations/dissociation_ood_255_857_36_000-0_neb1.0.traj",
        #     "/home/jovyan/shared-scratch/Brook/neb_stuff/dft_trajs_for_release/dissociations/dissociation_ood_473_6681_28_111-4_neb1.0.traj",
        #     "/home/jovyan/shared-scratch/Brook/neb_stuff/dft_trajs_for_release/dissociations/dissociation_ood_486_5397_13_011-1_neb1.0.traj",
        # ],
        # default=[
        #     "/home/jovyan/shared-scratch/Brook/neb_stuff/dft_trajs_for_release/dissociations",
        #     "/home/jovyan/shared-scratch/Brook/neb_stuff/dft_trajs_for_release/desorptions",
        #     "/home/jovyan/shared-scratch/Brook/neb_stuff/dft_trajs_for_release/transfers",
        # ],
        default=[ # ml trajectories
            "/home/jovyan/shared-scratch/Brook/neb_stuff/checkpoint/brookwander/neb_results/val_sps_w_dft_rx_init_final/ml_val_w_dft_sp_disoc_202310/eq2_153M_ec4_allmd",
            "/home/jovyan/shared-scratch/Brook/neb_stuff/checkpoint/brookwander/neb_results/val_sps_w_dft_rx_init_final/ml_val_w_dft_sp_transfer_202310/eq2_153M_ec4_allmd",
            "/home/jovyan/shared-scratch/Brook/neb_stuff/checkpoint/brookwander/neb_results/val_sps_w_dft_rx_init_final/ml_val_w_dft_sp_desorp_20231026/eq2_153M_ec4_allmd",
        ],
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
        default=None,
    )
    parser.add_argument(
        "--output_name",
        type=str, 
        default="ts_opt_df",
    )
    parser.add_argument(
        "--scale_file",
        type=str,
        default="/home/jovyan/working/ocp/configs/s2ef/all/gemnet/scaling_factors/gemnet-dT.json",
    )
    parser.add_argument(
        "--wandb",
        type=str, 
        default="n",
        help="whether to use wandb for logging (only in sweep mode), options:y/n, default: n"
    )
    parser.add_argument(
        "--nsteps",
        type=int, 
        default=500,
    )

    # sella hyperparameters
    parser.add_argument(
        "--method",
        type=str, 
        default="prfo",
        help="sella optimization algorithm, default for saddles is 'prfo', options are: ['qn', 'prfo', 'rfo']"
    )
    parser.add_argument(
        "--eta",
        type=float, 
        default=0.0007,
        help="Finite difference step size"
    )
    parser.add_argument(
        "--gamma",
        type=float, 
        default=0.1,
        help="Convergence criterion for iterative diagonalization"
    )
    parser.add_argument(
        "--delta0",
        type=float, 
        default=0.048,
        help="Initial trust radius"
    )
    parser.add_argument(
        "--rho_inc",
        type=float, 
        default=1.035,
        help="Threshold for increasing trust radius"
    )
    parser.add_argument(
        "--rho_dec",
        type=float, 
        default=5.0,
        help="Threshold for decreasing trust radius"
    )
    parser.add_argument(
        "--sigma_inc",
        type=float, 
        default=1.15,
        help="Trust radius increase factor"
    )
    parser.add_argument(
        "--sigma_dec",
        type=float,
        default=0.65,
        help="Trust radius decrease factor"
    )
    return parser


def main(config):
    pp.pprint(config)

    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"timestamp string (wandb group): {timestamp_str}")

    results_data = {
        "atoms": [],
        "name": [],
        "neb_list": [],
        "converged": [],
        "NEB_barrier_ML_energy": [],
        "NEB_barrier_ML_forces": [],
        "NEB_barrier_ML_fmax": [],
        "TS_opt_ML_max_energy": [],
        "TS_opt_ML_forces": [],
        "TS_opt_ML_fmax": [],
    }

    neb_files_list = []
    if isinstance(config['neb_path'], str):
        neb_files_list.append(config['neb_path'])
    else:
        for path in config['neb_path']:
            if path.endswith('.traj'):
                neb_files_list.append(path)
            else:
                neb_files_list += [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.traj')]

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

    for filepath in tqdm(neb_files_list):
        if config["wandb"] == 'y':
            wandb.init(
                config=config,
                name=filepath.split('/')[-1],
                project=f"ts_opt_sella",
                group=timestamp_str,
                notes=f"{config['output_name']}\ntimestamp group:{timestamp_str}",
            )

        neb_list = []
        neb_traj = read(filepath, index='-10:')
        max_neb_energy = -np.inf
        for i, atoms in enumerate(neb_traj[1:-1]):
            neb_list.append(atoms) # add to neb_list the DFT singlepoint results
            atoms_copy = atoms.copy()
            atoms_copy.calc = calc
            neb_energy = atoms_copy.get_potential_energy()
            neb_forces = atoms_copy.get_forces()
            sp_calc = SinglePointCalculator(atoms=atoms_copy, energy=float(neb_energy), forces=neb_forces)
            sp_calc.implemented_properties = ["energy", "forces"]
            atoms_copy.calc = sp_calc
            print(f"NEB atoms {i} {atoms_copy.symbols} DFT calc: {neb_energy:.4f}, fmax: {get_fmax(neb_forces):.4f}")
            if neb_energy > max_neb_energy:
                max_neb_energy = neb_energy
                max_neb_forces = neb_forces
                max_neb_fmax = get_fmax(max_neb_forces)
                ts_atoms = atoms_copy
                index = i
        print(f"ML calc max energy: {max_neb_energy} index {index}, fmax: {get_fmax(ts_atoms.get_forces())}")
        ts_atoms.calc = calc

        cons = Constraints(ts_atoms)
        for atom in ts_atoms:
            if atom.index in ts_atoms.constraints[0].index:
                cons.fix_translation(atom.index)

        dyn = Sella(
            ts_atoms,
            constraints=cons,
            trajectory=None,
            hessian_function=None,
            eig=True, # saddlepoint setting: True, relaxation: False
            order=1, # saddlepoint setting: 1, relaxation: 0
            method=config['method'], # saddplepoint setting: 'prfo'
            eta=config['eta'],        # Finite difference step size
            gamma=config['gamma'],       # Convergence criterion for iterative diagonalization
            delta0=config['delta0'],   # Initial trust radius
            rho_inc=config['rho_inc'],   # Threshold for increasing trust radius
            rho_dec=config['rho_dec'],     # Threshold for decreasing trust radius
            sigma_inc=config['sigma_inc'],  # Trust radius increase factor
            sigma_dec=config['sigma_dec'],  # Trust radius decrease factor
        )
        if config['wandb'] == 'y':
            dyn.attach(wandb_callback_func, interval=1, dyn=dyn, initial_energy=ts_atoms.get_potential_energy())
        try:
            dyn.run(0.01, config['nsteps'])
        except Exception as e:
            print(e)
        print(f"max energy {ts_atoms.get_potential_energy():.4f}, fmax: {get_fmax(ts_atoms.get_forces()):.4f}")

        ts_opt_energy = ts_atoms.get_potential_energy()
        ts_opt_forces = ts_atoms.get_forces()
        sp_calc = SinglePointCalculator(atoms=ts_atoms, energy=float(ts_opt_energy), forces=ts_opt_forces)
        sp_calc.implemented_properties = ["energy", "forces"]
        ts_atoms.calc = sp_calc

        converged = False
        if get_fmax(ts_opt_forces) <= 0.01:
            converged = True

        results_data['atoms'].append(ts_atoms)
        results_data['name'].append(filepath.split('/')[-1])
        results_data['neb_list'].append(neb_list)
        results_data['converged'].append(converged)
        results_data['NEB_barrier_ML_energy'].append(max_neb_energy)
        results_data['NEB_barrier_ML_forces'].append(max_neb_energy)
        results_data['NEB_barrier_ML_fmax'].append(max_neb_fmax)
        results_data['TS_opt_ML_max_energy'].append(ts_opt_energy)
        results_data['TS_opt_ML_forces'].append(ts_opt_forces)
        results_data['TS_opt_ML_fmax'].append(get_fmax(ts_opt_forces))

        if config["wandb"] == 'y':
            wandb.finish()

    print(f"fraction converged: {sum(results_data['converged'])/len(results_data['converged'])}")

    if config['output_dir'] is not None:
        os.makedirs(config['output_dir'], exist_ok=True)
        df = pd.DataFrame(results_data)
        df.to_pickle(os.path.join(config['output_dir'], f"{config['output_name']}.pkl"))

def wandb_callback_func(dyn, initial_energy):
    log_dict = {
        "fmax": get_fmax(dyn.atoms.get_forces()),
        "energy": dyn.atoms.get_potential_energy(),
        "i": dyn.nsteps,
        "rho": dyn.rho,
        "relative_energy": initial_energy - dyn.atoms.get_potential_energy(),
    }
    wandb.log(log_dict)
    


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    config = vars(args)
    main(config)
    print("done")
