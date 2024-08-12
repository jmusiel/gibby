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
    st_conversion=ase.units.invcm, # 1.239842e-4
    temperature=300,
    hessian_from_vasp=False,
    linear_scaling=None,
    drop_anomalies=False,
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
        if drop_anomalies and not row["no_anomaly"]:
            continue
        if row[success_column_key]:
            hessian = row[hessian_column_key]
            if hessian_from_vasp:
                hessian = -hessian

            if linear_scaling is not None:
                I = np.eye(len(hessian))
                m = linear_scaling["m"]
                b = linear_scaling["b"]
                hessian = hessian * m + b

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

            if "random_id" in row:
                results_dict["random_id"].append(row["random_id"])
            results_dict["ts"].append(ts)
            results_dict["zpe"].append(zpe)
            results_dict["deltah"].append(deltah)
            results_dict["total"].append(zpe + deltah - ts)
            results_dict["freq"].append(freq)
            results_dict["real_freq"].append(real_freq)
            results_dict["eigenvalues"].append([i for i in eigvalues])
            results_dict["eigenvectors"].append([i for i in eigvectors])
            results_dict["atoms"].append(atoms)
            results_dict["hessian"].append(hessian)

    return pd.DataFrame(results_dict)

def get_mean_corrections(
    dataframe,
    hessian_column_key: str,
    success_column_key: str = "success",
    atoms_column_key: str = "atoms",
    st_conversion=ase.units.invcm, # 1.239842e-4
    temperature=300,
    hessian_from_vasp=False,
    linear_scaling=None,
    drop_anomalies=False,
    ):

    correction_df = hessian_to_corrections(
        dataframe=dataframe,
        hessian_column_key = hessian_column_key,
        success_column_key = success_column_key,
        atoms_column_key = atoms_column_key,
        st_conversion = st_conversion,
        temperature = temperature,
        hessian_from_vasp = hessian_from_vasp,
        linear_scaling = linear_scaling,
        drop_anomalies = drop_anomalies,
    )

    cols = dataframe.columns
    cols_to_keep = ['mapping_idx', success_column_key, 'frequencies', 'E', atoms_column_key, 'random_id', 'no_anomaly', 'mpid', 'miller', 'shift', 'top', 'adsorbate', 'site', 'formula', 'stoichiometry', 'distribution']
    cols_remove = [col for col in cols if col not in cols_to_keep]
    df_meta = dataframe.drop(columns = cols_remove)
    correction_df = correction_df.merge(df_meta, left_on = "random_id", right_on = "random_id")

    mean_vals = correction_df.groupby("adsorbate").agg({"total": "mean", "zpe": "mean", "deltah": "mean", "ts": "mean"}).reset_index().to_dict(orient="records")

    lookup_mean = {}
    for entry in mean_vals:
        lookup_mean[entry["adsorbate"]] = {
            "total": entry["total"], 
            "zpe": entry["zpe"], 
            "deltah": entry["deltah"], 
            "ts": entry["ts"]
        }

    df_output = dataframe[dataframe["no_anomaly"]].copy()
    df_output = df_output[df_output[success_column_key]]

    df_output["total"] = correction_df.apply(lambda row: lookup_mean[row.adsorbate]["total"], axis=1)
    df_output["zpe"] = correction_df.apply(lambda row: lookup_mean[row.adsorbate]["zpe"], axis=1)
    df_output["deltah"] = correction_df.apply(lambda row: lookup_mean[row.adsorbate]["deltah"], axis=1)
    df_output["ts"] = correction_df.apply(lambda row: lookup_mean[row.adsorbate]["ts"], axis=1)

    return df_output
