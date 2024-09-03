import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from gibby.utils.ase_utils import get_fmax
from sklearn.metrics import mean_absolute_error

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
    return n_converged/n_calculated, n_success/n_converged, np.mean(fmax_conv), np.std(fmax_conv), np.mean(e_all), np.std(e_all)

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
    
    
    return n_converged/n_calculated, n_success/n_converged, np.mean(fmax_all), np.std(fmax_all), np.mean(e_all), np.std(e_all)

def get_sella_success_biased(df):
    n_converged= df[((df.TS_opt_ML_fmax  <= 0.01) & (df.converged_x) & (~df.failed_sp)) | df.barrierless_converged].shape[0]
    n_calculated = df[(~df.failed_sp)].shape[0]
    
    # success
    n_success = len(df[(((abs(df.residuals_sella) <= 0.1) & (df.TS_opt_ML_fmax <= 0.01) & (df.converged_x)) & (~df.failed_sp)) | df.barrierless_converged])
    
    #fmax
    fmax_all = df[(df.TS_opt_ML_fmax <= 0.01) & (df.converged_x)].sella_opt_fmax.dropna().tolist()
    
    # E
    e_all = np.abs(df[(df.TS_opt_ML_fmax <= 0.01) & (df.converged_x)].residuals_sella.dropna().values)
    
    return n_converged/n_calculated, n_success/n_converged, np.mean(fmax_all), np.std(fmax_all), np.mean(e_all), np.std(e_all)

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
    
    df_m["ML_neb_fmax"] = df_m.F_TS_SP.apply(get_fmax_wrapper)
    df_m["sella_opt_fmax"] = df_m.DFT_forces.apply(get_fmax_wrapper)
    
    # Find convergence and success for CatTSunami baseline and sella
    p_converged_baseline, p_success_baseline, mean_fmax_baseline, std_fmax_baseline, mae_e_baseline, std_e_baseline = get_cattsunami_results(df_m)
    
    p_converged_sella, p_success_sella, mean_fmax_sella, std_fmax_sella, mae_e_sella, std_e_sella = get_sella_fallback_results(df_m)
    
    p_converged_sella2, p_success_sella2, mean_fmax_sella2, std_fmax_sella2, mae_e_sella2, std_e_sella2 = get_sella_success_biased(df_m)
    
    
    print(f"\nCatTSunami baseline:\n% Converged = {p_converged_baseline*100:1.2f}\n% Success = {p_success_baseline*100:1.2f}")
    print(f"MAE energy [eV]: {mae_e_baseline:1.3f} ({std_e_baseline:1.3f})")
    print(f"Mean fmax [eV/Ang]: {mean_fmax_baseline:1.3f} ({std_fmax_baseline:1.3f})\n")
    
    print(f"Sella convergence biased:\n% Converged = {p_converged_sella*100:1.2f}\n% Success = {p_success_sella*100:1.2f}")
    print(f"MAE energy [eV]: {mae_e_sella:1.3f} ({std_e_sella:1.3f})")
    print(f"Mean fmax [eV/Ang]: {mean_fmax_sella:1.3f} ({std_fmax_sella:1.3f})\n")
    
    print(f"Sella success biased:\n% Converged = {p_converged_sella2*100:1.2f}\n% Success = {p_success_sella2*100:1.2f}")
    print(f"MAE energy [eV]: {mae_e_sella2:1.3f} ({std_e_sella2:1.3f})")
    print(f"Mean fmax [eV/Ang]: {mean_fmax_sella2:1.3f} ({std_fmax_sella2:1.3f})\n\n")
    
    
    cts_base = {"Method": "CatTSunami", "Mean fmax [eV/Ang]": f"{mean_fmax_baseline:1.3f} ({std_fmax_baseline:1.3f})",
               "Energy MAE [eV]": f"{mae_e_baseline:1.3f} ({std_e_baseline:1.3f})",
               "Convergence [%]": f"{p_converged_baseline*100:1.2f}", "Success [%]": f"{p_success_baseline*100:1.2f}"}
    
    sella_c = {"Method": "Sella refined (any converged)", "Mean fmax [eV/Ang]": f"{mean_fmax_sella:1.3f} ({std_fmax_sella:1.3f})",
               "Energy MAE [eV]": f"{mae_e_sella:1.3f} ({std_e_sella:1.3f})",
               "Convergence [%]": f"{p_converged_sella*100:1.2f}", "Success [%]": f"{p_success_sella*100:1.2f}"}
    
    sella_s = {"Method": "Sella refined (both converged)", "Mean fmax [eV/Ang]": f"{mean_fmax_sella2:1.3f} ({std_fmax_sella2:1.3f})",
               "Energy MAE [eV]": f"{mae_e_sella2:1.3f} ({std_e_sella2:1.3f})",
               "Convergence [%]": f"{p_converged_sella2*100:1.2f}", "Success [%]": f"{p_success_sella2*100:1.2f}"}
    df_table = pd.DataFrame([cts_base, sella_c, sella_s])
    print(df_table.to_latex())
    


    size = 17

    val_fig, val_ax = plt.subplots(1, 1, figsize=(6,6))

    pre_opt_data = df_m[(~df_m.residuals_neb.isnull()) & (df_m.all_converged) & (df_m.both_barriered)].ML_neb_fmax.tolist()
    post_opt_data = df_m[(~df_m.residuals_sella.isnull()) & (df_m.TS_opt_ML_fmax < 0.01) & (df_m.both_barriered)].sella_opt_fmax.tolist()
    bins = np.linspace(min(pre_opt_data + post_opt_data), max(pre_opt_data + post_opt_data), 100)
    val_ax.hist(pre_opt_data, bins = bins, alpha=0.5, color='tab:blue', label=f'Pre-optimization', density = True)
    val_ax.hist(post_opt_data, bins = bins, alpha=0.5, color='tab:orange', label=f'Sella optimized', density = True)
    # val_ax.set_title("Local Mininum Residuals")

    val_ax.legend()
    val_ax.set_xlabel("maximum force ($eV/\AA$)", fontsize=size)
    val_ax.set_ylabel("density", fontsize=size)
    # val_ax.set_ylim(0, 400)
    val_ax.tick_params(axis='both', which='major', labelsize=size-3)
    val_fig.patch.set_facecolor('white')

    val_fig.savefig("fmax_residual_distribution_shift_sella.svg")
    val_fig.savefig("fmax_residual_distribution_shift_sella.png", dpi=300, bbox_inches="tight")
    
    