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
    drop_anomalies=True,
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


def get_mae_over_mean(
    df:pd.DataFrame,
    vasp_key: str,
    ml_key: str,
    success_column_key: str = "success",
    no_anomaly_column_str: str = "no_anomaly",
    atoms_column_key: str = "atoms",
    st_conversion=ase.units.invcm, # 1.239842e-4
    temperature=300,
    ):
    """
    Get the mean absolute error when assuming the mean value per adsorbate for
    ML predictions.

    Args:
        df: pd.DataFrame
            The dataframe with the metadata, VASP values, and ML predictions.
        vasp_key: str
            The column key for the VASP values.
        ml_key: str
            The column key for the ML predictions.
        success_column_key: str
            The column key for the success column.
        no_anomaly_column_str: str
            The column key for the anomaly column. Should be True if there are no anomalies.
        atoms_column_key: str
            The column key for the atoms column.
        st_conversion: float
            The conversion factor to convert cm^-1 to eV.
        temperature: float
            The temperature in Kelvin at which the Gibbs corrections should be assessed.

    Returns:
        float: The mean absolute error when assuming the mean value per adsorbate for
        ML predictions.
        float: The mean absolute error when assuming the mean value per adsorbate for
        the VASP values.
        pd.DataFrame: The dataframe with the mean values per adsorbate.
    """
    # Get corrections
    df_ml = hessian_to_corrections(df, hessian_column_key = ml_key, success_column_key = success_column_key, atoms_column_key = atoms_column_key, st_conversion = st_conversion, temperature = temperature)
    df_vasp = hessian_to_corrections(df, hessian_column_key = vasp_key, success_column_key = success_column_key, atoms_column_key = atoms_column_key, st_conversion = st_conversion, temperature = temperature, hessian_from_vasp = True)

    # Clean up meatadata
    cols = df.columns
    cols_to_keep = ['mapping_idx', success_column_key, 'frequencies', 'E', atoms_column_key, 'random_id', no_anomaly_column_str, 'mpid', 'miller', 'shift', 'top', 'adsorbate', 'site', 'formula', 'stoichiometry', 'distribution']
    cols_remove = [col for col in cols if col not in cols_to_keep]
    df_meta = df.drop(columns = cols_remove)

    # Merge metadata and the ml and vasp corrections
    df_both = df_ml.merge(df_vasp, left_on = "random_id", right_on = "random_id", suffixes=('_ml', '_vasp'))
    df_both = df_both.merge(df_meta, left_on = "random_id", right_on = "random_id")
    df_both["ml_abs_error"] = abs(df_both.total_vasp - df_both.total_ml)

    # Get the mean value per adsorbate
    df_both_no_anom = df_both[df_both[no_anomaly_column_str]].copy()

    mean_vals_vasp = df_both_no_anom.groupby("adsorbate").agg({"total_vasp": "mean", "zpe_vasp": "mean", "deltah_vasp": "mean", "ts_vasp": "mean"}).reset_index().to_dict(orient="records")
    lookup_mean_vasp = {}
    for entry in mean_vals_vasp:
        lookup_mean_vasp[entry["adsorbate"]] = (entry["total_vasp"], entry["zpe_vasp"], entry["deltah_vasp"], entry["ts_vasp"])

    mean_vals_ml = df_both_no_anom.groupby("adsorbate").agg({"total_ml": "mean", "zpe_ml": "mean", "deltah_ml": "mean", "ts_ml": "mean"}).reset_index().to_dict(orient="records")
    lookup_mean_ml = {}
    for entry in mean_vals_ml:
        lookup_mean_ml[entry["adsorbate"]] = (entry["total_ml"], entry["zpe_ml"], entry["deltah_ml"], entry["ts_ml"])

    # Get the mean absolute error when assuming the mean value per adsorbate
    df_both_no_anom["vasp_abs_error_over_mean"] = df_both_no_anom.apply(lambda row: abs(lookup_mean_vasp[row.adsorbate][0] - row.total_vasp), axis=1)
    df_both_no_anom["ml_abs_error_over_mean"] = df_both_no_anom.apply(lambda row: abs(lookup_mean_ml[row.adsorbate][0] - row.total_ml), axis=1)
    # Note you can add each component if you want.

    return df_both_no_anom["ml_abs_error_over_mean"].mean(), df_both_no_anom["vasp_abs_error_over_mean"].mean(), df_both_no_anom

# REMOVE ME later:
# for testing
if __name__ == "__main__":
    # load the dataframe and convert to corrections dataframes

    from gibby.utils.hessian_to_corrections import hessian_to_corrections, get_mae_over_mean, get_mean_corrections
    import pickle

    with open('/home/jovyan/shared-scratch/joe/for_brook/hessian_pickles/rerun_val_rev3.pkl', 'rb') as f:
        val_df = pickle.load(f)
        
    drop_an = True

    val_dict = {
        "DFT": hessian_to_corrections(val_df, hessian_column_key="hessian", hessian_from_vasp=True, drop_anomalies=drop_an),
        "ML":{
            "fine tuned EQ2": hessian_to_corrections(val_df, hessian_column_key="checkpoint_12", hessian_from_vasp=False, drop_anomalies=drop_an),
            "EQ2 153M": hessian_to_corrections(val_df, hessian_column_key="eq2_153M_ec4_allmd", hessian_from_vasp=False, drop_anomalies=drop_an),
        },
        "mean_ads": {
            "Mean per ads. EQ2": get_mean_corrections(val_df, hessian_column_key="eq2_153M_ec4_allmd", hessian_from_vasp=False),
            "Mean per ads. VASP": get_mean_corrections(val_df, hessian_column_key="hessian", hessian_from_vasp=True),
        },
        "name": "Local minimum",
    }

    # check the mean absolute error over mean from get_mae_over_mean
    val_mae_over_mean = get_mae_over_mean(df=val_df, vasp_key="hessian", ml_key="eq2_153M_ec4_allmd")
    print(f"brook's ml: {val_mae_over_mean[0]} brook's vasp: {val_mae_over_mean[1]}")


    # make the table of results
    import pandas as pd
    from collections import defaultdict
    import json
    from tabulate import tabulate
    import numpy as np

    # iterate over the dictionary of corrections from different sources
    # calculate the MAEs by taking the mean of the absolute value of the difference between the DFT and ML correction values
    table_dict = defaultdict(list)
    for key, values in val_dict["ML"].items():
        dft_values = val_dict["DFT"]

        table_dict["Method"].append(key)
        table_dict["ZPE"].append(np.mean(np.abs(values["zpe"] - dft_values["zpe"])))
        table_dict["TS"].append(np.mean(np.abs(values["ts"] - dft_values["ts"])))
        table_dict["C$\mathrm{_p}$"].append(np.mean(np.abs(values["deltah"] - dft_values["deltah"])))
        table_dict["Total Gibbs"].append(np.mean(np.abs(values["total"] - dft_values["total"])))  

    for key, values in val_dict["mean_ads"].items():
        dft_values = val_dict["DFT"]

        table_dict["Method"].append(key)
        table_dict["ZPE"].append(np.mean(np.abs(values["zpe"] - dft_values["zpe"])))
        table_dict["TS"].append(np.mean(np.abs(values["ts"] - dft_values["ts"])))
        table_dict["C$\mathrm{_p}$"].append(np.mean(np.abs(values["deltah"] - dft_values["deltah"])))
        table_dict["Total Gibbs"].append(np.mean(np.abs(values["total"] - dft_values["total"])))  

    df = pd.DataFrame(table_dict)

    # print the table
    table_string = tabulate(
        df,
        headers="keys",
        tablefmt="latex_booktabs",
        floatfmt=".3f",
        showindex=False,
    )
    table_string = table_string.replace("\\^{}", "^")
    table_string = table_string.replace("\\$", "$")
    table_string = table_string.replace("\\_", "_")
    table_string = table_string.replace("\\{", "{")
    table_string = table_string.replace("\\}", "}")
    table_string = table_string.replace("\\textbackslash{}", "\\")

    # add subheader of eV units
    second_row_list = table_string.split("\n")[2].split("&")
    for i in range(len(second_row_list)):
        second_row_list[i] = " "*len(second_row_list[i])
    subheader = "$[eV]$"
    second_row_list[1] = second_row_list[1][:-(len(subheader))] + subheader
    second_row_list[2] = second_row_list[2][:-(len(subheader))] + subheader
    second_row_list[3] = second_row_list[3][:-(len(subheader))] + subheader
    second_row_list[4] = second_row_list[4][:-(len(subheader))-2] + subheader
    second_row = "&".join(second_row_list) + "\\\\"
    table_string = table_string.replace("\\midrule",second_row + "\n\\midrule")

    table_string = table_string.replace(
        "\\toprule", 
        "" + \
            "\\toprule \n" + \
            "\\multicolumn{5}{c}{\\textbf{Local Minimum Correction MAE}} \\\\ \n" + \
            "\\midrule"
    ) 

    print()
    print(table_string)
    print()