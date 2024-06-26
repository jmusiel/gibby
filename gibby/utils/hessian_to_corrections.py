import ase
import ase.vibrations
import numpy as np
from ase.thermochemistry import HarmonicThermo
from collections import defaultdict
from tqdm import tqdm
import pandas as pd


def hessian_to_corrections(
    dataframe,
    hessian_column_key: str,
    success_column_key: str = "success",
    atoms_column_key: str = "atoms",
    st_conversion=1.239842e-4,
    temperature=300,
    hessian_from_vasp=False,
):
    """
    input a dataframe containing hessians in a column, given by hessian_column_key
    also boolean values for successful convergence at success_column_key, letting this function know to use these hessians
    atoms_column_key is should correspond to a column carrying the atoms object that the hessian calculation was performed on
    st_conversion is the conversion factor from cm^-1 to eV
    temperature is the temperature in K
    hessian_from_vasp is a boolean that should be set to True if the hessians are from VASP, which will multiply the hessians by -1, since VASP outputs negative hessians for backwards compatibility reasons

    returns a dataframe with eigenvalues, frequencies, and corrections
    """

    results_dict = defaultdict(list)
    for index, row in tqdm(
        dataframe.iterrows(),
        total=len(dataframe),
        desc=f"Processing hessians for {hessian_column_key}",
    ):
        if row[success_column_key]:
            hessian = row[hessian_column_key]
            if hessian_from_vasp:
                hessian = -hessian

            atoms = row[atoms_column_key]
            free_indices = [
                i for i in range(len(atoms)) if not i in atoms.constraints[0].index
            ]

            vibdata = ase.vibrations.VibrationsData.from_2d(
                atoms, hessian_2d=hessian, indices=free_indices
            )
            freq = vibdata.get_frequencies()
            real_freq = np.real(freq)
            thermo = HarmonicThermo(
                [f * st_conversion for f in real_freq if not f == 0]
            )
            ts = (
                thermo.get_entropy(temperature=temperature, verbose=False) * temperature
            )
            zpe = thermo.get_ZPE_correction()
            deltah = thermo._vibrational_energy_contribution(temperature=temperature)
            eigvalues, eigvectors = np.linalg.eig(hessian)

            results_dict["random_id"].append(row["random_id"])
            results_dict["ts"].append(ts)
            results_dict["zpe"].append(zpe)
            results_dict["deltah"].append(deltah)
            results_dict["total"].append(zpe + deltah - ts)
            results_dict["freq"].append(freq)
            results_dict["real_freq"].append(real_freq)
            results_dict["eigenvalues"].append([i for i in eigvalues])
            results_dict["eigenvectors"].append([i for i in eigvectors])

    return pd.DataFrame(results_dict)