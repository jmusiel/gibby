import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from gibby.utils.ase_utils import get_fmax
from sklearn.metrics import mean_absolute_error
import plotly.express as px

def get_sella_info(name):
    reaction_class = name.split("_")[0]
    id_ = name.split("d_")[1].split("-k_")[0]
    return reaction_class, id_

def get_fmax_wrapper(forces):
    if type(forces) != np.ndarray:
        return np.nan
    else:
        return get_fmax(forces)
    
def get_cattsunami_results(df_m):
    n_converged = df_m[(df_m.converged_x) & (~df_m.failed_sp)].shape[0]
    n_calculated = df_m[(~df_m.failed_sp)].shape[0]
    
    n_success = len(df_m[((df_m.sp_residual <= 0.1) & (df_m.all_converged)& (df_m.both_barriered))|((df_m.barrierless_converged))])
    
    fmax_conv = df_m[(df_m.all_converged)].ML_neb_fmax.tolist()
    
    e_all = np.abs(df_m[(df_m.all_converged)].residuals_neb.dropna().values)
    return n_converged/n_calculated, n_success/n_converged, n_success/n_calculated, np.mean(fmax_conv), np.std(fmax_conv), np.mean(e_all), np.std(e_all)

def get_sella_fallback_results(df2):
    # Converged
    n_converged_sella = df2[df2.TS_opt_ML_fmax  <= 0.01].shape[0]
    n_converged_fallback = df2[(df2.TS_opt_ML_fmax  > 0.01) & (df2.converged_x)].shape[0]
    n_converged = n_converged_sella + n_converged_fallback
    
    # Calculated
    n_calculated = len(df2) - len(df2[(df2.TS_opt_ML_fmax > 0.01) & (df2.failed_sp)])
    
    # Success
    ns = len(df2[((abs(df2.residuals_sella) <= 0.1) & (df2.both_barriered) & (df2.TS_opt_ML_fmax <= 0.01))| ((df2.both_barrierless) & (df2.TS_opt_ML_fmax <= 0.01))])
    nf = len(df2[((df2.sp_residual <= 0.1) & (df2.both_barriered) & (df2.TS_opt_ML_fmax > 0.01))| ((df2.barrierless_converged)& (df2.TS_opt_ML_fmax > 0.01))])
    n_success = ns + nf
    
    # fmax
    fmax_sella_conv = df2[(df2.TS_opt_ML_fmax <= 0.01)].sella_opt_fmax.dropna().tolist()
    fmax_fallback_conv = df2[(df2.TS_opt_ML_fmax >= 0.01) & (df2.all_converged)].ML_neb_fmax.tolist()
    fmax_all = fmax_sella_conv + fmax_fallback_conv
    
    # E
    e_sella_conv = df2[(df2.TS_opt_ML_fmax <= 0.01)].residuals_sella.dropna().tolist()
    e_fallback_conv = df2[(df2.TS_opt_ML_fmax >= 0.01) & (df2.all_converged)].residuals_neb.dropna().tolist()
    e_all = np.abs(np.array(e_sella_conv + e_fallback_conv))
    
    
    return n_converged/n_calculated, n_success/n_converged, n_success/n_calculated, np.mean(fmax_all), np.std(fmax_all), np.mean(e_all), np.std(e_all)

def get_sella_success_biased(df):
    n_converged= df[((df.TS_opt_ML_fmax  <= 0.01) & (df.converged_x) & (~df.failed_sp)) | df.barrierless_converged].shape[0]
    n_calculated = df[(~df.failed_sp)].shape[0]
    
    # success
    n_success = len(df[(((abs(df.residuals_sella) <= 0.1) & (df.TS_opt_ML_fmax <= 0.01) & (df.converged_x)) & (~df.failed_sp)) | df.barrierless_converged])
    
    #fmax
    fmax_all = df[(df.TS_opt_ML_fmax <= 0.01) & (df.converged_x)].sella_opt_fmax.dropna().tolist()
    
    # E
    e_all = np.abs(df[(df.TS_opt_ML_fmax <= 0.01) & (df.converged_x)].residuals_sella.dropna().values)
    
    return n_converged/n_calculated, n_success/n_converged, n_success/n_calculated, np.mean(fmax_all), np.std(fmax_all), np.mean(e_all), np.std(e_all)

if __name__ == "__main__":
    
    # Preprocess the data so it may be contained in one df where each row is a unique transition state + slab system
    df_desorp = pd.read_pickle("/home/jovyan/ocpneb/notebooks/parse_validation/desorp_df.pkl")
    df_disoc = pd.read_pickle("/home/jovyan/ocpneb/notebooks/parse_validation/disoc_df.pkl")
    df_transfer = pd.read_pickle("/home/jovyan/ocpneb/notebooks/parse_validation/transfer_df.pkl")
    
    df_sella = pd.read_pickle("/home/jovyan/shared-scratch/Brook/gibbs_proj/sella_sps/sella_sp_ml_ts_pretrained_df.pkl")
    df_sella["reaction_class"], df_sella["calculation_id"] = zip(*df_sella.name.apply(get_sella_info))
    
    df_sella_desorp = df_sella[df_sella.reaction_class == "desorption"]
    df_sella_disoc = df_sella[df_sella.reaction_class == "dissociation"]
    df_sella_transfer = df_sella[df_sella.reaction_class == "transfer"]
    
    df_desorp_m = df_sella_desorp.merge(df_desorp, left_on = "calculation_id", right_on = "neb_id")
    df_disoc_m = df_sella_disoc.merge(df_disoc, left_on = "calculation_id", right_on = "neb_id")
    df_transfer_m = df_sella_transfer.merge(df_transfer, left_on = "calculation_id", right_on = "neb_id")
    
    df_m = pd.concat([df_desorp_m, df_disoc_m, df_transfer_m])
    df_m = df_m[df_m.residual < 200].copy() # There is one dft neb with E_TS(raw) = 0 -- this must be an error so we will ignore
    
    # Apply minor data calculations
    df_m["residuals_neb"] = df_m.E_TS_SP - df_m.E_raw_TS
    df_m["residuals_sella"] = df_m.SP_energy_DFT - df_m.E_raw_TS
    df_desorp_m["residuals_neb"] = df_desorp_m.E_TS_SP - df_desorp_m.E_raw_TS
    df_desorp_m["residuals_sella"] = df_desorp_m.SP_energy_DFT - df_desorp_m.E_raw_TS
    df_disoc_m["residuals_neb"] = df_disoc_m.E_TS_SP - df_disoc_m.E_raw_TS
    df_disoc_m["residuals_sella"] = df_disoc_m.SP_energy_DFT - df_disoc_m.E_raw_TS
    df_disoc_m = df_disoc_m[df_disoc_m.residual < 200].copy() # There is one dft neb with E_TS(raw) = 0 -- this must be an error so we will ignore
    df_transfer_m["residuals_neb"] = df_transfer_m.E_TS_SP - df_transfer_m.E_raw_TS
    df_transfer_m["residuals_sella"] = df_transfer_m.SP_energy_DFT - df_transfer_m.E_raw_TS
    
    df_m["ML_neb_fmax"] = df_m.F_TS_SP.apply(get_fmax_wrapper)
    df_m["sella_opt_fmax"] = df_m.DFT_forces.apply(get_fmax_wrapper)
    df_desorp_m["ML_neb_fmax"] = df_desorp_m.F_TS_SP.apply(get_fmax_wrapper)
    df_desorp_m["sella_opt_fmax"] = df_desorp_m.DFT_forces.apply(get_fmax_wrapper)
    df_disoc_m["ML_neb_fmax"] = df_disoc_m.F_TS_SP.apply(get_fmax_wrapper)
    df_disoc_m["sella_opt_fmax"] = df_disoc_m.DFT_forces.apply(get_fmax_wrapper)
    df_transfer_m["ML_neb_fmax"] = df_transfer_m.F_TS_SP.apply(get_fmax_wrapper)
    df_transfer_m["sella_opt_fmax"] = df_transfer_m.DFT_forces.apply(get_fmax_wrapper)

    # Find convergence and success for CatTSunami baseline and sella
    p_converged_baseline, p_success_baseline, p_success_overall_baseline, mean_fmax_baseline, std_fmax_baseline, mae_e_baseline, std_e_baseline = get_cattsunami_results(df_m)
    p_converged_baseline_desorp, p_success_baseline_desorp, p_success_overall_baseline_desorp, mean_fmax_baseline_desorp, std_fmax_baseline_desorp, mae_e_baseline_desorp, std_e_baseline_desorp = get_cattsunami_results(df_desorp_m)
    p_converged_baseline_disoc, p_success_baseline_disoc, p_success_overall_baseline_disoc, mean_fmax_baseline_disoc, std_fmax_baseline_disoc, mae_e_baseline_disoc, std_e_baseline_disoc = get_cattsunami_results(df_disoc_m)
    p_converged_baseline_transfer, p_success_baseline_transfer, p_success_overall_baseline_transfer, mean_fmax_baseline_transfer, std_fmax_baseline_transfer, mae_e_baseline_transfer, std_e_baseline_transfer = get_cattsunami_results(df_transfer_m)
    
    p_converged_sella, p_success_sella, p_success_sella_overall, mean_fmax_sella, std_fmax_sella, mae_e_sella, std_e_sella = get_sella_fallback_results(df_m)
    p_converged_sella_desorp, p_success_sella_desorp, p_success_sella_overall_desorp, mean_fmax_sella_desorp, std_fmax_sella_desorp, mae_e_sella_desorp, std_e_sella_desorp = get_sella_fallback_results(df_desorp_m)
    p_converged_sella_disoc, p_success_sella_disoc, p_success_sella_overall_disoc, mean_fmax_sella_disoc, std_fmax_sella_disoc, mae_e_sella_disoc, std_e_sella_disoc = get_sella_fallback_results(df_disoc_m)
    p_converged_sella_transfer, p_success_sella_transfer, p_success_sella_overall_transfer, mean_fmax_sella_transfer, std_fmax_sella_transfer, mae_e_sella_transfer, std_e_sella_transfer = get_sella_fallback_results(df_transfer_m)
    
    p_converged_sella2, p_success_sella2, p_success_sella2_overall, mean_fmax_sella2, std_fmax_sella2, mae_e_sella2, std_e_sella2 = get_sella_success_biased(df_m)
    p_converged_sella2_desorp, p_success_sella2_desorp, p_success_sella2_overall_desorp, mean_fmax_sella2_desorp, std_fmax_sella2_desorp, mae_e_sella2_desorp, std_e_sella2_desorp = get_sella_success_biased(df_desorp_m)
    p_converged_sella2_disoc, p_success_sella2_disoc, p_success_sella2_overall_disoc, mean_fmax_sella2_disoc, std_fmax_sella2_disoc, mae_e_sella2_disoc, std_e_sella2_disoc = get_sella_success_biased(df_disoc_m)
    p_converged_sella2_transfer, p_success_sella2_transfer, p_success_sella2_overall_transfer, mean_fmax_sella2_transfer, std_fmax_sella2_transfer, mae_e_sella2_transfer, std_e_sella2_transfer = get_sella_success_biased(df_transfer_m)
    
    
    print(f"\nCatTSunami baseline:\n% Converged = {p_converged_baseline*100:1.2f}\n% Success = {p_success_baseline*100:1.2f}")
    print(f"% Success overall = {p_success_overall_baseline*100:1.2f}")
    print(f"MAE energy [eV]: {mae_e_baseline:1.3f} ({std_e_baseline:1.3f})")
    print(f"Mean fmax [eV/Ang]: {mean_fmax_baseline:1.3f} ({std_fmax_baseline:1.3f})\n")

    print(f"CatTSunami desorption:\n% Converged = {p_converged_baseline_desorp*100:1.2f}\n% Success = {p_success_baseline_desorp*100:1.2f}")
    print(f"% Success overall = {p_success_overall_baseline_desorp*100:1.2f}")
    print(f"MAE energy [eV]: {mae_e_baseline_desorp:1.3f} ({std_e_baseline_desorp:1.3f})")
    print(f"Mean fmax [eV/Ang]: {mean_fmax_baseline_desorp:1.3f} ({std_fmax_baseline_desorp:1.3f})\n")

    print(f"CatTSunami dissociation:\n% Converged = {p_converged_baseline_disoc*100:1.2f}\n% Success = {p_success_baseline_disoc*100:1.2f}")
    print(f"% Success overall = {p_success_overall_baseline_disoc*100:1.2f}")
    print(f"MAE energy [eV]: {mae_e_baseline_disoc:1.3f} ({std_e_baseline_disoc:1.3f})")
    print(f"Mean fmax [eV/Ang]: {mean_fmax_baseline_disoc:1.3f} ({std_fmax_baseline_disoc:1.3f})\n")

    print(f"CatTSunami transfer:\n% Converged = {p_converged_baseline_transfer*100:1.2f}\n% Success = {p_success_baseline_transfer*100:1.2f}")
    print(f"% Success overall = {p_success_overall_baseline_transfer*100:1.2f}")
    print(f"MAE energy [eV]: {mae_e_baseline_transfer:1.3f} ({std_e_baseline_transfer:1.3f})")
    print(f"Mean fmax [eV/Ang]: {mean_fmax_baseline_transfer:1.3f} ({std_fmax_baseline_transfer:1.3f})\n")
    
    print(f"Sella convergence biased:\n% Converged = {p_converged_sella*100:1.2f}\n% Success = {p_success_sella*100:1.2f}")
    print(f"% Success overall = {p_success_sella_overall*100:1.2f}")
    print(f"MAE energy [eV]: {mae_e_sella:1.3f} ({std_e_sella:1.3f})")
    print(f"Mean fmax [eV/Ang]: {mean_fmax_sella:1.3f} ({std_fmax_sella:1.3f})\n")

    print(f"Sella convergence biased desorption:\n% Converged = {p_converged_sella_desorp*100:1.2f}\n% Success = {p_success_sella_desorp*100:1.2f}")
    print(f"% Success overall = {p_success_sella_overall_desorp*100:1.2f}")
    print(f"MAE energy [eV]: {mae_e_sella_desorp:1.3f} ({std_e_sella_desorp:1.3f})")
    print(f"Mean fmax [eV/Ang]: {mean_fmax_sella_desorp:1.3f} ({std_fmax_sella_desorp:1.3f})\n")

    print(f"Sella convergence biased dissociation:\n% Converged = {p_converged_sella_disoc*100:1.2f}\n% Success = {p_success_sella_disoc*100:1.2f}")
    print(f"% Success overall = {p_success_sella_overall_disoc*100:1.2f}")
    print(f"MAE energy [eV]: {mae_e_sella_disoc:1.3f} ({std_e_sella_disoc:1.3f})")
    print(f"Mean fmax [eV/Ang]: {mean_fmax_sella_disoc:1.3f} ({std_fmax_sella_disoc:1.3f})\n")

    print(f"Sella convergence biased transfer:\n% Converged = {p_converged_sella_transfer*100:1.2f}\n% Success = {p_success_sella_transfer*100:1.2f}")
    print(f"% Success overall = {p_success_sella_overall_transfer*100:1.2f}")
    print(f"MAE energy [eV]: {mae_e_sella_transfer:1.3f} ({std_e_sella_transfer:1.3f})")
    print(f"Mean fmax [eV/Ang]: {mean_fmax_sella_transfer:1.3f} ({std_fmax_sella_transfer:1.3f})\n")
    
    print(f"Sella success biased:\n% Converged = {p_converged_sella2*100:1.2f}\n% Success = {p_success_sella2*100:1.2f}")
    print(f"% Success overall = {p_success_sella2_overall*100:1.2f}")
    print(f"MAE energy [eV]: {mae_e_sella2:1.3f} ({std_e_sella2:1.3f})")
    print(f"Mean fmax [eV/Ang]: {mean_fmax_sella2:1.3f} ({std_fmax_sella2:1.3f})\n\n")

    print(f"Sella success biased desorption:\n% Converged = {p_converged_sella2_desorp*100:1.2f}\n% Success = {p_success_sella2_desorp*100:1.2f}")
    print(f"% Success overall = {p_success_sella2_overall_desorp*100:1.2f}")
    print(f"MAE energy [eV]: {mae_e_sella2_desorp:1.3f} ({std_e_sella2_desorp:1.3f})")
    print(f"Mean fmax [eV/Ang]: {mean_fmax_sella2_desorp:1.3f} ({std_fmax_sella2_desorp:1.3f})\n\n")

    print(f"Sella success biased dissociation:\n% Converged = {p_converged_sella2_disoc*100:1.2f}\n% Success = {p_success_sella2_disoc*100:1.2f}")
    print(f"% Success overall = {p_success_sella2_overall_disoc*100:1.2f}")
    print(f"MAE energy [eV]: {mae_e_sella2_disoc:1.3f} ({std_e_sella2_disoc:1.3f})")
    print(f"Mean fmax [eV/Ang]: {mean_fmax_sella2_disoc:1.3f} ({std_fmax_sella2_disoc:1.3f})\n\n")

    print(f"Sella success biased transfer:\n% Converged = {p_converged_sella2_transfer*100:1.2f}\n% Success = {p_success_sella2_transfer*100:1.2f}")
    print(f"% Success overall = {p_success_sella2_overall_transfer*100:1.2f}")
    print(f"MAE energy [eV]: {mae_e_sella2_transfer:1.3f} ({std_e_sella2_transfer:1.3f})")
    print(f"Mean fmax [eV/Ang]: {mean_fmax_sella2_transfer:1.3f} ({std_fmax_sella2_transfer:1.3f})\n\n")
    
    
    cts_base_all = {"Reaction": "all", "Method": "CatTSunami", "Mean fmax [eV/Ang]": f"{mean_fmax_baseline:1.3f} ({std_fmax_baseline:1.3f})",
               "Energy MAE [eV]": f"{mae_e_baseline:1.3f} ({std_e_baseline:1.3f})",
               "Convergence [%]": f"{p_converged_baseline*100:1.2f}", "Success when converged [%]": f"{p_success_baseline*100:1.2f}", "Success overall [%]": f"{p_success_overall_baseline*100:1.2f}"}
    
    sella_c_all = {"Reaction": "all", "Method": "Sella refined (any converged)", "Mean fmax [eV/Ang]": f"{mean_fmax_sella:1.3f} ({std_fmax_sella:1.3f})",
               "Energy MAE [eV]": f"{mae_e_sella:1.3f} ({std_e_sella:1.3f})",
               "Convergence [%]": f"{p_converged_sella*100:1.2f}", "Success when converged [%]": f"{p_success_sella*100:1.2f}", "Success overall [%]": f"{p_success_sella_overall*100:1.2f}"}
    
    sella_s_all = {"Reaction": "all", "Method": "Sella refined (both converged)", "Mean fmax [eV/Ang]": f"{mean_fmax_sella2:1.3f} ({std_fmax_sella2:1.3f})",
               "Energy MAE [eV]": f"{mae_e_sella2:1.3f} ({std_e_sella2:1.3f})",
               "Convergence [%]": f"{p_converged_sella2*100:1.2f}", "Success when converged [%]": f"{p_success_sella2*100:1.2f}", "Success overall [%]": f"{p_success_sella2_overall*100:1.2f}"}

    cts_base_desorp = {"Reaction": "desorption", "Method": "CatTSunami", "Mean fmax [eV/Ang]": f"{mean_fmax_baseline_desorp:1.3f} ({std_fmax_baseline_desorp:1.3f})",
                "Energy MAE [eV]": f"{mae_e_baseline_desorp:1.3f} ({std_e_baseline_desorp:1.3f})",
                "Convergence [%]": f"{p_converged_baseline_desorp*100:1.2f}", "Success when converged [%]": f"{p_success_baseline_desorp*100:1.2f}", "Success overall [%]": f"{p_success_overall_baseline_desorp*100:1.2f}"}

    sella_c_desorp = {"Reaction": "desorption", "Method": "Sella refined (any converged)", "Mean fmax [eV/Ang]": f"{mean_fmax_sella_desorp:1.3f} ({std_fmax_sella_desorp:1.3f})",
                "Energy MAE [eV]": f"{mae_e_sella_desorp:1.3f} ({std_e_sella_desorp:1.3f})",
                "Convergence [%]": f"{p_converged_sella_desorp*100:1.2f}", "Success when converged [%]": f"{p_success_sella_desorp*100:1.2f}", "Success overall [%]": f"{p_success_sella_overall_desorp*100:1.2f}"}

    sella_s_desorp = {"Reaction": "desorption", "Method": "Sella refined (both converged)", "Mean fmax [eV/Ang]": f"{mean_fmax_sella2_desorp:1.3f} ({std_fmax_sella2_desorp:1.3f})",
                "Energy MAE [eV]": f"{mae_e_sella2_desorp:1.3f} ({std_e_sella2_desorp:1.3f})",
                "Convergence [%]": f"{p_converged_sella2_desorp*100:1.2f}", "Success when converged [%]": f"{p_success_sella2_desorp*100:1.2f}", "Success overall [%]": f"{p_success_sella2_overall_desorp*100:1.2f}"}   

    cts_base_disoc = {"Reaction": "dissociation", "Method": "CatTSunami", "Mean fmax [eV/Ang]": f"{mean_fmax_baseline_disoc:1.3f} ({std_fmax_baseline_disoc:1.3f})",
                "Energy MAE [eV]": f"{mae_e_baseline_disoc:1.3f} ({std_e_baseline_disoc:1.3f})",
                "Convergence [%]": f"{p_converged_baseline_disoc*100:1.2f}", "Success when converged [%]": f"{p_success_baseline_disoc*100:1.2f}", "Success overall [%]": f"{p_success_overall_baseline_disoc*100:1.2f}"}

    sella_c_disoc = {"Reaction": "dissociation", "Method": "Sella refined (any converged)", "Mean fmax [eV/Ang]": f"{mean_fmax_sella_disoc:1.3f} ({std_fmax_sella_disoc:1.3f})",
                "Energy MAE [eV]": f"{mae_e_sella_disoc:1.3f} ({std_e_sella_disoc:1.3f})",
                "Convergence [%]": f"{p_converged_sella_disoc*100:1.2f}", "Success when converged [%]": f"{p_success_sella_disoc*100:1.2f}", "Success overall [%]": f"{p_success_sella_overall_disoc*100:1.2f}"}

    sella_s_disoc = {"Reaction": "dissociation", "Method": "Sella refined (both converged)", "Mean fmax [eV/Ang]": f"{mean_fmax_sella2_disoc:1.3f} ({std_fmax_sella2_disoc:1.3f})",
                "Energy MAE [eV]": f"{mae_e_sella2_disoc:1.3f} ({std_e_sella2_disoc:1.3f})",
                "Convergence [%]": f"{p_converged_sella2_disoc*100:1.2f}", "Success when converged [%]": f"{p_success_sella2_disoc*100:1.2f}", "Success overall [%]": f"{p_success_sella2_overall_disoc*100:1.2f}"}

    cts_base_transfer = {"Reaction": "transfer", "Method": "CatTSunami", "Mean fmax [eV/Ang]": f"{mean_fmax_baseline_transfer:1.3f} ({std_fmax_baseline_transfer:1.3f})",
                "Energy MAE [eV]": f"{mae_e_baseline_transfer:1.3f} ({std_e_baseline_transfer:1.3f})",
                "Convergence [%]": f"{p_converged_baseline_transfer*100:1.2f}", "Success when converged [%]": f"{p_success_baseline_transfer*100:1.2f}", "Success overall [%]": f"{p_success_overall_baseline_transfer*100:1.2f}"}

    sella_c_transfer = {"Reaction": "transfer", "Method": "Sella refined (any converged)", "Mean fmax [eV/Ang]": f"{mean_fmax_sella_transfer:1.3f} ({std_fmax_sella_transfer:1.3f})",
                "Energy MAE [eV]": f"{mae_e_sella_transfer:1.3f} ({std_e_sella_transfer:1.3f})",
                "Convergence [%]": f"{p_converged_sella_transfer*100:1.2f}", "Success when converged [%]": f"{p_success_sella_transfer*100:1.2f}", "Success overall [%]": f"{p_success_sella_overall_transfer*100:1.2f}"}

    sella_s_transfer = {"Reaction": "transfer", "Method": "Sella refined (both converged)", "Mean fmax [eV/Ang]": f"{mean_fmax_sella2_transfer:1.3f} ({std_fmax_sella2_transfer:1.3f})",
                "Energy MAE [eV]": f"{mae_e_sella2_transfer:1.3f} ({std_e_sella2_transfer:1.3f})",
                "Convergence [%]": f"{p_converged_sella2_transfer*100:1.2f}", "Success when converged [%]": f"{p_success_sella2_transfer*100:1.2f}", "Success overall [%]": f"{p_success_sella2_overall_transfer*100:1.2f}"}


    df_table = pd.DataFrame([cts_base_all, sella_c_all, sella_s_all, cts_base_desorp, sella_c_desorp, sella_s_desorp, cts_base_disoc, sella_c_disoc, sella_s_disoc, cts_base_transfer, sella_c_transfer, sella_s_transfer])
    # print latex table dropping indices
    print(df_table.to_latex(index=False))
    

    # Make fmax density plot
    size = 17

    val_fig, val_ax = plt.subplots(1, 1, figsize=(6,6))

    pre_opt_data = df_m[(~df_m.residuals_neb.isnull()) & (df_m.all_converged) & (df_m.both_barriered)].ML_neb_fmax.tolist()
    post_opt_data = df_m[(~df_m.residuals_sella.isnull()) & (df_m.TS_opt_ML_fmax < 0.01) & (df_m.both_barriered)].sella_opt_fmax.tolist()
    bins = np.linspace(min(pre_opt_data + post_opt_data), max(pre_opt_data + post_opt_data), 100)
    val_ax.hist(pre_opt_data, bins = bins, alpha=0.5, color='tab:blue', label=f'Pre-optimization', density = True)
    val_ax.hist(post_opt_data, bins = bins, alpha=0.5, color='tab:orange', label=f'Sella optimized', density = True)

    val_ax.legend()
    val_ax.set_xlabel("maximum force ($eV/\AA$)", fontsize=size)
    val_ax.set_ylabel("density", fontsize=size)
    

    val_ax.tick_params(axis='both', which='major', labelsize=size-3)
    val_fig.patch.set_facecolor('white')

    val_fig.savefig("fmax_residual_distribution_shift_sella.svg")
    val_fig.savefig("fmax_residual_distribution_shift_sella.png", dpi=300, bbox_inches="tight")
    
    # Make converged / success bar chart
    dfsum = pd.DataFrame([
        {"Approach": "CatTSunami baseline", "Failure": 100-p_success_baseline*100, "Unconverged": 100-p_converged_baseline*100},
        {"Approach": "Sella refined (any converged)", "Failure": 100-p_success_sella*100, "Unconverged": 100-p_converged_sella*100},
        {"Approach": "Sella refined (both converged)", "Failure": 100-p_success_sella2*100, "Unconverged": 100-p_converged_sella2*100},
    ])
    
    fig = px.bar(dfsum, x="Approach", y="Failure", color="Approach", barmode="group", template="plotly_white",
                 color_discrete_map={"CatTSunami baseline": "#0C023E", "Sella success optimized": "#9FC131", "Sella convergence optimized": "#DBF227"})
    fig.update_layout( yaxis_title = "% Failure", font_family="Arial", autosize=False, width=500, height=500)
    fig.update_yaxes(range = [0,30])

    fig.write_image("success_bar_plot.svg")
    
    fig = px.bar(dfsum, x="Approach", y="Unconverged", color="Approach", barmode="group", template="plotly_white",
                 color_discrete_map={"CatTSunami baseline": "#0C023E", "Sella success optimized": "#9FC131", "Sella convergence optimized": "#DBF227"})
    fig.update_layout( yaxis_title = "% Unconverged", font_family="Arial", autosize=False, width=500, height=500)
    fig.update_yaxes(range = [0,30])

    fig.write_image("convergence_bar_plot.svg")
    fig.show()    
    