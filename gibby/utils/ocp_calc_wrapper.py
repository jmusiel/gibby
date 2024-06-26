from ocpmodels.common.relaxation.ase_utils import OCPCalculator, logging
from ocpmodels.trainers.base_trainer import (
    os,
    errno,
    load_state_dict,
    load_scales_compat,
)
import types
import torch
import numpy as np

# this wrapper works for modifying the OCPCalculator in commit c52aeeacb3854c8d7841ab3953a9cfef284a301f


class OCPCalcWrapper(OCPCalculator):
    def __init__(
        self,
        checkpoint_path: str,
        config_overrides: dict = {},
        cpu: bool = True,
    ):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        config = checkpoint["config"]
        if not isinstance(config["dataset"], list):
            config["dataset"] = [config["dataset"]]
        config = recursive_update(config, config_overrides)

        super().__init__(config_yml=config, checkpoint_path=checkpoint_path, cpu=cpu)

    # overwrite load checkpoint so that trainer load checkpoint can be overwritten to allow inference from checkpoints which still include "optimizer"
    # (such as checkpoints which were saved for continued finetuning, but are being used for inference only i.e. "checkpoint.pt" instead of "best_checkpoint.pt")
    def load_checkpoint(
        self,
        checkpoint_path: str,
        checkpoint={},
    ) -> None:
        """
        Load existing trained model

        Args:
            checkpoint_path: string
                Path to trained model
        """
        self.trainer.load_checkpoint = types.MethodType(
            base_trainer_override_load_checkpoint, self.trainer
        )
        try:
            self.trainer.load_checkpoint(checkpoint_path, checkpoint)
        except NotImplementedError:
            logging.warning("Unable to load checkpoint!")

    def calculate(self, atoms, properties, system_changes) -> None:
        super().calculate(atoms, properties, system_changes)
        self.results["forces"] = self.results["forces"].astype(np.float32)


# base_trainer overwrite for load_checkpoint
def base_trainer_override_load_checkpoint(
    self, checkpoint_path: str, checkpoint={}
) -> None:
    if not checkpoint:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                errno.ENOENT, "Checkpoint file not found", checkpoint_path
            )
        else:
            logging.info(f"Loading checkpoint from: {checkpoint_path}")
            map_location = torch.device("cpu") if self.cpu else self.device
            checkpoint = torch.load(checkpoint_path, map_location=map_location)

    self.epoch = checkpoint.get("epoch", 0)
    self.step = checkpoint.get("step", 0)
    self.best_val_metric = checkpoint.get("best_val_metric", None)
    self.primary_metric = checkpoint.get("primary_metric", None)

    # Match the "module." count in the keys of model and checkpoint state_dict
    # DataParallel model has 1 "module.",  DistributedDataParallel has 2 "module."
    # Not using either of the above two would have no "module."

    ckpt_key_count = next(iter(checkpoint["state_dict"])).count("module")
    mod_key_count = next(iter(self.model.state_dict())).count("module")
    key_count_diff = mod_key_count - ckpt_key_count

    if key_count_diff > 0:
        new_dict = {
            key_count_diff * "module." + k: v
            for k, v in checkpoint["state_dict"].items()
        }
    elif key_count_diff < 0:
        new_dict = {
            k[len("module.") * abs(key_count_diff) :]: v
            for k, v in checkpoint["state_dict"].items()
        }
    else:
        new_dict = checkpoint["state_dict"]

    strict = self.config["task"].get("strict_load", True)
    load_state_dict(self.model, new_dict, strict=strict)

    # if "optimizer" in checkpoint:
    #     self.optimizer.load_state_dict(checkpoint["optimizer"])
    # if "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
    #     self.scheduler.scheduler.load_state_dict(checkpoint["scheduler"])
    # if "ema" in checkpoint and checkpoint["ema"] is not None:
    #     self.ema.load_state_dict(checkpoint["ema"])
    # else:
    #     self.ema = None
    self.ema = None

    scale_dict = checkpoint.get("scale_dict", None)
    if scale_dict:
        logging.info(
            "Overwriting scaling factors with those loaded from checkpoint. "
            "If you're generating predictions with a pretrained checkpoint, this is the correct behavior. "
            "To disable this, delete `scale_dict` from the checkpoint. "
        )
        load_scales_compat(self._unwrapped_model, scale_dict)

    for key in checkpoint["normalizers"]:
        if key in self.normalizers:
            self.normalizers[key].load_state_dict(checkpoint["normalizers"][key])
        if self.scaler and checkpoint["amp"]:
            self.scaler.load_state_dict(checkpoint["amp"])


def recursive_update(dict0, dict1):
    for k, v in dict1.items():
        if isinstance(v, dict):
            dict0[k] = recursive_update(dict0.get(k, {}), v)
        else:
            dict0[k] = v
    return dict0


def get_config_override(checkpoint_path: str, scale_file_path: str):
    config_override = {}
    if "gemnet_dt" in checkpoint_path:
        config_override = {
            "model_attributes": {
                "scale_file": scale_file_path,
            },
        }
    elif "schnet" in checkpoint_path:
        config_override = {
            "task": {
                "strict_load": False,
            },
        }
    else:
        config_override = {}

    config_override = recursive_update(config_override, {"optim": {"scheduler": None}})

    return config_override
