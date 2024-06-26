import numpy as np
import shutil
import os
import sys
from ase.vibrations import Vibrations


def get_fmax(forces: np.ndarray):
    return np.sqrt((forces**2).sum(axis=1).max())


class suppress_stdout:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def get_hessian(atoms, run_dir=None):
    if run_dir is not None:
        cwd = os.getcwd()
        os.chdir(run_dir)
    shutil.rmtree("vib", ignore_errors=True)
    vib = Vibrations(
        atoms,
        indices=[i for i in range(len(atoms)) if not i in atoms.constraints[0].index],
    )
    with suppress_stdout():
        vib.run()
        vib.summary(log=sys.stdout)
    shutil.rmtree("vib", ignore_errors=True)
    hessian = vib.H
    if run_dir is not None:
        os.chdir(cwd)
    return hessian
