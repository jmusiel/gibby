import wandb
import numpy as np

from gibby.workflow.python.ts_opt import main as ts_opt_main
from gibby.workflow.python.ts_opt import get_parser as ts_opt_get_parser


import argparse
import pprint
pp = pprint.PrettyPrinter(indent=4)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep_name",
        type=str, 
        default="ts_opt_sweep",
    )
    parser.add_argument(
        "--sweep_mode",
        type=str, 
        default="init",
        help="init or agent"
    )
    parser.add_argument(
        "--sweep_id",
        type=str, 
        default=None
    )
    return parser

def main(config):
    pp.pprint(config)

    if config["sweep_mode"] == "init":
        sweep_config = {
            "name": config['sweep_name'],
            "method": "random",
            "metric":{
                "goal": "minimize",
                "name": "fmax"
            },
            "parameters": {
                "neb_path":{"values":[
                    "/home/jovyan/shared-scratch/Brook/neb_stuff/dft_trajs_for_release/dissociations/dissociation_ood_22_8962_11_111-3_neb1.0.traj",
                    "/home/jovyan/shared-scratch/Brook/neb_stuff/dft_trajs_for_release/dissociations/dissociation_id_611_9766_9_111-1_neb1.0.traj",
                    "/home/jovyan/shared-scratch/Brook/neb_stuff/dft_trajs_for_release/dissociations/dissociation_ood_255_857_36_000-0_neb1.0.traj",
                    "/home/jovyan/shared-scratch/Brook/neb_stuff/dft_trajs_for_release/dissociations/dissociation_ood_473_6681_28_111-4_neb1.0.traj",
                    "/home/jovyan/shared-scratch/Brook/neb_stuff/dft_trajs_for_release/dissociations/dissociation_ood_486_5397_13_011-1_neb1.0.traj",
                ]},
                "method":{"values": ["prfo"]},
                "eta":{"max":1e-3 , "min": 1e-5, "distribution": "log_uniform_values"},
                "gamma":{"max": 5e-1, "min": 1e-2, "distribution": "log_uniform_values"},
                "delta0":{"max": 5e-1, "min": 1e-2, "distribution": "log_uniform_values"},
                "rho_inc":{"max": 2, "min": 0.5, "distribution": "log_uniform_values"},
                "rho_dec":{"max": 7, "min": 0.5, "distribution": "log_uniform_values"},
                "sigma_inc":{"max": 2, "min": 0.65, "distribution": "log_uniform_values"},
                "sigma_dec":{"max": 1, "min": 0.1, "distribution": "log_uniform_values"},
            },
        }
        sweep_id = wandb.sweep(sweep_config, project=config['sweep_name'])

    elif config["sweep_mode"] == "agent":
        wandb.agent(config["sweep_id"], function=run_sweep, project=config['sweep_name'])


def run_sweep():
    run = wandb.init()

    ts_opt_config = vars(ts_opt_get_parser().parse_args([]))

    ts_opt_config.update(wandb.config)

    ts_opt_config["wandb"] = "y"

    ts_opt_main(ts_opt_config)

    wandb.finish()

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    config = vars(args)
    main(config)
    print("done")