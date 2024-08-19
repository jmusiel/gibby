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


def get_hessian(atoms, run_dir=None, hessian_delta=0.01, tags=None):
    if run_dir is not None:
        cwd = os.getcwd()
        os.chdir(run_dir)
    shutil.rmtree("vib", ignore_errors=True)
    if tags is None:
        indices = [i for i in range(len(atoms)) if not i in atoms.constraints[0].index]
    else:
        indices = [i for i in range(len(atoms)) if tags[i] == 2]
    vib = Vibrations(
        atoms,
        indices=indices,
        delta=hessian_delta,
    )
    with suppress_stdout():
        vib.run()
        vib.summary(log=sys.stdout)
    shutil.rmtree("vib", ignore_errors=True)
    hessian = vib.H
    if run_dir is not None:
        os.chdir(cwd)
    return hessian


def get_analytical_hessian(atoms):
    """
    Get the analytical hessian from the OCPCalWrapper and an OCP model which saves the hessian during forward.
    """
    free_indices = [i for i in range(len(atoms)) if not i in atoms.constraints[0].index]
    full_hessian = atoms.calc.extract_hessian(atoms)
    hessian = full_hessian.reshape((len(atoms), 3, len(atoms), 3))
    hessian = hessian[free_indices, :, :, :]
    hessian = hessian[:, :, free_indices, :]
    hessian = hessian.reshape((len(free_indices) * 3, len(free_indices) * 3))

    # # alternative approach to get the hessian, gets the same thing
    # hessian_free_indices = np.array([[ind*3, ind*3+1, ind*3+2] for ind in free_indices]).flatten()
    # hessian_alt = full_hessian.reshape((len(atoms)* 3, len(atoms)* 3))
    # hessian_alt = hessian_alt[hessian_free_indices,:]
    # hessian_alt = hessian_alt[:,hessian_free_indices]

    return hessian
