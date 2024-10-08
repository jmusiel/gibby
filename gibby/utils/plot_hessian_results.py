import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator
import ase
import scipy.optimize
from tqdm import tqdm
import scipy


def plot_corrections_histogram(pred_dataframe, true_dataframe, name=None, size=16):
    """
    Plot a histogram of the corrections from a dataframe containing corrections.
    Args:
        pred_dataframe: ML predictions from hessian_to_corrections
        true_dataframe: VASP results from hessian_to_corrections
        name: for title of figure (optional)
        size: default 16
    Returns:
        None
    """
    pass

    fig, axs = plt.subplots(1, 4, figsize=(28, 6))
    fig.suptitle(f"{name} Gibbs Energy Correction Term Errors", fontsize=size)
    for i, (correction_key, correction_name) in enumerate(
        [
            ("ts", "Entropy correction"),
            ("zpe", "Zero point energy correction"),
            ("deltah", "Heat capacity correction"),
            ("total", "Total correction"),
        ]
    ):
        ml_values = pred_dataframe[correction_key].values
        vasp_values = true_dataframe[correction_key].values
        errors = np.abs(ml_values - vasp_values)
        errors_greater_than_threshold = len(errors[errors > 0.1]) / len(errors)
        axs[i].hist(
            errors,
            bins=100,
            color="tab:blue",
            alpha=1,
            density=True,
            label=f"MAE: {np.mean(np.abs(errors)):.3f} [eV] -- percent >0.1: {100*errors_greater_than_threshold:.3f}%",
        )
        axs[i].set_xlabel(f"{correction_name} abs err [eV]", fontsize=size)
        axs[i].legend()

        print(
            f"{correction_name.ljust(30)} -- MAE: {np.mean(np.abs(errors)):.3f} -- percent >0.1: {100*errors_greater_than_threshold:.3f}"
        )

    fig.patch.set_facecolor("white")
    return fig


def _get_value_metadata(value_name):
    if value_name == "eigenvalues":
        get_values = get_eigenvalues
        units = "$eV/\AA^2$"
    elif value_name == "total":
        get_values = get_total_corrections
        units = "$eV$"
    elif value_name == "freq":
        get_values = get_frequencies
        units = "$cm^{-1}$"
    elif value_name == "all_freq":
        get_values = get_all_frequencies
        units = "$cm^{-1}$"
    return get_values, units


def plot_hexbin_corrections(
        pred_dataframes_list,
        true_dataframe,
        pred_dataframes_names: list=None,
        value_name: str = "eigenvalues",
        title_name=None, 
        size=16,
        include_mae=True,
        color_max=5,
        cbar_ticks=[1, 2, 3, 4, 5],
        include_parity_line=False,
    ):
    """
    Plot a hexbin plot of the corrections from a dataframe containing corrections.
    Args:
        pred_dataframe: ML predictions from hessian_to_corrections
        true_dataframe: VASP results from hessian_to_corrections
        name: for title of figure (optional)
        size: default 16
    Returns:
        None
    """

    fig_cols = 4
    if len(pred_dataframes_list) < 4:
        fig_cols = len(pred_dataframes_list)
    fig_rows = int(math.ceil(len(pred_dataframes_list) / fig_cols))

    fig, axs = plt.subplots(
        fig_rows, fig_cols, figsize=(8 * fig_cols + 4, 6 * fig_rows)
    )
    if title_name is not None:
        fig.suptitle(title_name, fontsize=size)

    if type(axs) is np.ndarray:
        axs = axs.flatten()
    for i, df in enumerate(pred_dataframes_list):
        if type(axs) is np.ndarray:
            ax = axs[i]
        else:
            ax = axs

        i_name = "ML"
        if pred_dataframes_names is not None:
            i_name = pred_dataframes_names[i]

        hexbin0 = apply_hexbin_plot_to_axes(
            ax,
            df,
            true_dataframe,
            pred_df_name=i_name,
            value_name=value_name,
            size=size,
            include_mae=include_mae,
            color_max=color_max,
            ax_min=None,
            ax_max=None,
            include_parity_line=include_parity_line,
        )

    apply_hexbin_colorbar(hexbin0, fig, axs, size=size, cbar_ticks=cbar_ticks)

    fig.patch.set_facecolor("white")

    return fig


import matplotlib.ticker as ticker
@ticker.FuncFormatter
def major_formatter(x, pos):
    label = f"{-x:.0f}i" if x < 0 else f"{x:.0f}"
    return label


def apply_hexbin_plot_to_axes(
    ax,
    pred_df,
    true_df,
    pred_df_name="ML",
    value_name: str = "eigenvalues",
    size=20,
    include_mae=True,
    color_max=5,
    ax_min=None,
    ax_max=None,
    include_parity_line=False,
):
    """
    Apply hexbin plot to axes
    """

    # special case for matching frequencies
    if value_name == "all_matching_freqs":
        units = "$cm^{-1}$"
        xlabel = f"matching ML freq. ({units})"
        ylabel = f"LI/SR DFT freq. ({units})"
        ml_values = get_all_matching_frequencies(pred_df, true_df)
        vasp_values = get_frequencies(true_df)
        ax.xaxis.set_major_formatter(major_formatter)
        ax.yaxis.set_major_formatter(major_formatter)

    # normal case for eigenvalues, total, and freq
    else:
        get_values, units = _get_value_metadata(value_name)

        if value_name == "eigenvalues":
            xlabel = f"eigenvalues ({units})"
            ylabel = f"DFT eigenvalues ({units})"
        elif value_name == "total":
            xlabel = f"total Gibbs correction ({units})"
            ylabel = f"DFT total correction ({units})"
        elif value_name == "freq":
            xlabel = f"LI/SR frequency ({units})"
            ylabel = f"DFT LI/SR frequency ({units})"
            ax.xaxis.set_major_formatter(major_formatter)
            ax.yaxis.set_major_formatter(major_formatter)
        elif value_name == "all_freq":
            xlabel = f"all frequencies ({units})"
            ylabel = f"DFT all frequencies ({units})"
            ax.xaxis.set_major_formatter(major_formatter)
            ax.yaxis.set_major_formatter(major_formatter)

        ml_values = get_values(pred_df)
        vasp_values = get_values(true_df)


    min_max = (np.inf, -np.inf)
    min_max = (min(min(ml_values), min_max[0]), max(max(ml_values), min_max[1]))
    min_max = (min(min(vasp_values), min_max[0]), max(max(vasp_values), min_max[1]))
    min_max = (
        min_max[0] - (np.max(np.abs(min_max)) * 0.1),
        min_max[1] + (np.max(np.abs(min_max)) * 0.1),
    )
    if ax_min is not None:
        min_max = (ax_min, min_max[1])
    if ax_max is not None:
        min_max = (min_max[0], ax_max)

    hexbin0 = ax.hexbin(
        ml_values,
        vasp_values,
        gridsize=100,
        cmap="viridis",
        vmin=1,
        vmax=color_max,
        mincnt=1,
        extent=[min_max[0], min_max[1], min_max[0], min_max[1]],
    )
    ax.set_xlabel(f"{pred_df_name} {xlabel}", fontsize=size)
    ax.set_ylabel(ylabel, fontsize=size)
    ax.set_aspect("equal", "box")
    ax.tick_params(axis='both', which='major', labelsize=size-5)
    # ax.set_xlim(min_max)
    # ax.set_ylim(min_max)

    if include_parity_line:
        ax.plot(min_max, min_max, color='tab:gray', linestyle='--')

    mae = np.mean(np.abs(ml_values - vasp_values))
    # rmae = np.mean([np.abs((x-y)/y) for x, y in zip(ml_values, vasp_values)])# relative mean error
    if include_mae:
        ax.text(
            0.02,
            0.98,
            f"MAE: {mae:.3f} {units}",
            transform=ax.transAxes,
            verticalalignment="top",
        )

    return hexbin0


def apply_hexbin_colorbar(
    hexbin0,
    fig,
    axs,
    size=15,
    pad=0.05,
    cbar_ticks=[1, 2, 3, 4, 5],
):
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    else:
        axs = axs.flatten()
    # Create a formatter function that formats numbers as integers
    formatter = FuncFormatter(lambda x, pos: f"{x:.0f}")
    # add shared colorbar
    cbar = fig.colorbar(hexbin0, ax=axs, format=formatter, pad=pad)
    # set colorbar ticks fontsize
    cbar.ax.tick_params(labelsize=size)
    cbar.set_ticks(cbar_ticks)


def savefig(fig, filename):
    fig.patch.set_facecolor("white")
    fig.savefig(filename, dpi=300, bbox_inches="tight")


def get_eigenvalues(given_df):
    eigen_values_list = [sorted(eig) for eig in given_df["eigenvalues"].values]
    result = np.concatenate(eigen_values_list)
    result = np.real(result)
    return np.array(result)


def get_total_corrections(given_df):
    result = given_df["total"].values
    return np.array(result)


def get_frequencies(given_df):
    unsorted_freq_list = [freq for freq in given_df["freq"].values]
    sorted_freq_list = []
    for freq_sublist in unsorted_freq_list:
        sorted_args = np.argsort(np.array(freq_sublist) ** 2)
        sorted_freq_sublist = freq_sublist[sorted_args]
        sorted_freq_list.append(sorted_freq_sublist)
    values_list = []
    for freq_sublist in sorted_freq_list:
        leftmost = freq_sublist[0]
        real = np.real(leftmost)
        imag = np.imag(leftmost)
        value = real - imag
        values_list.append(value)
    return np.array(values_list)

def get_all_frequencies(given_df):
    freq_list = [sorted(np.real(freq) - np.imag(freq)) for freq in given_df["freq"].values]
    result = np.concatenate(freq_list)
    result = np.real(result) - np.imag(result)
    return np.array(result)

def get_all_matching_frequencies(ml_df, dft_df):
    values_list = []
    for j, ml_row in tqdm(ml_df.iterrows(), total=len(ml_df), desc="Matching frequencies"):
        dft_row = dft_df.iloc[j]

        atoms = ml_row["atoms"]
        free_indices = [i for i in range(len(atoms)) if not i in atoms.constraints[0].index]

        # get ml frequencies and eigenvectors from ase
        ml_hessian = ml_row["hessian"]
        ml_vibdata = ase.vibrations.VibrationsData.from_2d(atoms, hessian_2d=ml_hessian, indices=free_indices)
        ml_active_atoms = ml_vibdata._atoms[ml_vibdata.get_mask()]
        ml_masses = ml_active_atoms.get_masses()
        ml_mass_weights = np.repeat(ml_masses**-0.5, 3)
        ml_omega2, ml_vectors = np.linalg.eigh(ml_mass_weights * ml_vibdata.get_hessian_2d() * ml_mass_weights[:, np.newaxis])
        ml_freqs = ml_vibdata.get_frequencies()

        # get dft frequencies and eigenvectors from ase
        dft_hessian = dft_row["hessian"]
        dft_vibdata = ase.vibrations.VibrationsData.from_2d(atoms, hessian_2d=dft_hessian, indices=free_indices)
        dft_active_atoms = dft_vibdata._atoms[dft_vibdata.get_mask()]
        dft_masses = dft_active_atoms.get_masses()
        dft_mass_weights = np.repeat(dft_masses**-0.5, 3)
        dft_omega2, dft_vectors = np.linalg.eigh(dft_mass_weights * dft_vibdata.get_hessian_2d() * dft_mass_weights[:, np.newaxis])
        dft_freqs = dft_vibdata.get_frequencies()

        # organize frequencies to match eigenvectors to dft eigenvectors
        cost_matrix = np.array([[-np.abs(scipy.spatial.distance.cosine(vec_ml, vec_dft)-1) for vec_ml in ml_vectors] for vec_dft in dft_vectors])
        row_indices, col_indices = scipy.optimize.linear_sum_assignment(cost_matrix)
        matched_ml_freqs = ml_freqs[col_indices]
        matched_ml_vectors = ml_vectors[col_indices]

        pre_cost = [-np.abs(scipy.spatial.distance.cosine(vec_ml, vec_dft)-1) for vec_ml, vec_dft in zip(ml_vectors, dft_vectors)]
        post_cost = [-np.abs(scipy.spatial.distance.cosine(vec_ml, vec_dft)-1) for vec_ml, vec_dft in zip(matched_ml_vectors, dft_vectors)]
        # if not np.mean(pre_cost) == np.mean(post_cost):
        #     print(f"{j}: {np.mean(pre_cost)} -> {np.mean(post_cost)}")
        assert np.mean(post_cost) <= np.mean(pre_cost)

        # sort matched frequencies according to leftmost dft, and take the corresponding ml frequency
        sorted_args = np.argsort(np.array(dft_freqs) ** 2)
        leftmost = matched_ml_freqs[sorted_args[0]]
        real = np.real(leftmost)
        imag = np.imag(leftmost)
        value = real - imag
        values_list.append(value)
    
    return np.array(values_list)


def get_cosine_similarity_of_leftmost_eigenvector(ml_df, dft_df, eigmatching=False, all_vectors=False):
    cossim_list = []
    dft_freq_list = []
    for j, ml_row in tqdm(ml_df.iterrows(), total=len(ml_df), desc="Getting cosine similarity of leftmost eigenvector"):
        dft_row = dft_df.iloc[j]

        atoms = ml_row["atoms"]
        free_indices = [i for i in range(len(atoms)) if not i in atoms.constraints[0].index]

        # get ml frequencies and eigenvectors from ase
        ml_hessian = ml_row["hessian"]
        ml_vibdata = ase.vibrations.VibrationsData.from_2d(atoms, hessian_2d=ml_hessian, indices=free_indices)
        ml_active_atoms = ml_vibdata._atoms[ml_vibdata.get_mask()]
        ml_masses = ml_active_atoms.get_masses()
        ml_mass_weights = np.repeat(ml_masses**-0.5, 3)
        ml_omega2, ml_vectors = np.linalg.eigh(ml_mass_weights * ml_vibdata.get_hessian_2d() * ml_mass_weights[:, np.newaxis])
        ml_freqs = ml_vibdata.get_frequencies()
        ml_sorted_args = np.argsort(np.array(ml_freqs) ** 2)
        ml_freqs = ml_freqs[ml_sorted_args]
        ml_vectors = ml_vectors[ml_sorted_args]

        # get dft frequencies and eigenvectors from ase
        dft_hessian = dft_row["hessian"]
        dft_vibdata = ase.vibrations.VibrationsData.from_2d(atoms, hessian_2d=dft_hessian, indices=free_indices)
        dft_active_atoms = dft_vibdata._atoms[dft_vibdata.get_mask()]
        dft_masses = dft_active_atoms.get_masses()
        dft_mass_weights = np.repeat(dft_masses**-0.5, 3)
        dft_omega2, dft_vectors = np.linalg.eigh(dft_mass_weights * dft_vibdata.get_hessian_2d() * dft_mass_weights[:, np.newaxis])
        dft_freqs = dft_vibdata.get_frequencies()
        dft_sorted_args = np.argsort(np.array(dft_freqs) ** 2)
        dft_freqs = dft_freqs[dft_sorted_args]
        dft_vectors = dft_vectors[dft_sorted_args]

        if eigmatching:
            cost_matrix = np.array([[-np.abs(scipy.spatial.distance.cosine(vec_ml, vec_dft)-1) for vec_ml in ml_vectors] for vec_dft in dft_vectors])
            row_indices, col_indices = scipy.optimize.linear_sum_assignment(cost_matrix)
            matched_ml_freqs = ml_freqs[col_indices]
            matched_ml_vectors = ml_vectors[col_indices]
            ml_vectors = matched_ml_vectors
            ml_freqs = matched_ml_freqs

        if all_vectors:
            cossim = [np.abs(scipy.spatial.distance.cosine(vec_ml, vec_dft)-1) for vec_ml, vec_dft in zip(ml_vectors, dft_vectors)]
            cossim_list.extend(cossim)
            dft_freq_all = np.real(dft_freqs) - np.imag(dft_freqs)
            dft_freq_list.extend(dft_freq_all)
        else: # leftmost only
            # get the cosine similarity score of the leftmost eigenvector of the ml and dft hessians
            cossim = np.abs(scipy.spatial.distance.cosine(ml_vectors[0], dft_vectors[0])-1)
            cossim_list.append(cossim)
            dft_leftmost = np.real(dft_freqs[0]) - np.imag(dft_freqs[0])
            dft_freq_list.append(dft_leftmost)


    # sort the cossim and dft_leftmost arrays by the magnitude of the dft leftmost frequency values
    cossim_arr = np.array(cossim_list)
    dft_freq_arr = np.array(dft_freq_list)
    sorted_args = np.argsort(dft_freq_arr)
    cossim_arr = cossim_arr[sorted_args]
    dft_freq_arr = dft_freq_arr[sorted_args]

    return cossim_arr, dft_freq_arr



def plot_mae_vs_key(
    pred_dataframes_list,
    true_dataframe,
    pred_dataframes_names: list = None,
    value_name: str = "eigenvalues",
    title_name=None,
    xlabel: str = None,
    size=20,
):

    get_values, units = _get_value_metadata(value_name)

    if value_name == "eigenvalues":
        ylabel = f"eigenvalues MAE ({units})"
    elif value_name == "total":
        ylabel = f"total Gibbs correction MAE ({units})"
    elif value_name == "freq":
        ylabel = f"LI/SR frequency MAE ({units})"
    elif value_name == "all_freq":
        ylabel = f"all frequencies MAE ({units})"

    vasp_values = get_values(true_dataframe)
    mae_list = []
    for pred_df in pred_dataframes_list:
        ml_values = get_values(pred_df)
        mae = np.mean(np.abs(ml_values - vasp_values))
        mae_list.append(mae)

    fig, axs = plt.subplots(1, 1, figsize=(6, 6))
    axs.plot(pred_dataframes_names, mae_list, marker="o", color="tab:blue")
    axs.set_xlabel(xlabel, fontsize=size)
    axs.set_ylabel(ylabel, fontsize=size)

    # Set x-axis major locator to integer values
    axs.xaxis.set_major_locator(MaxNLocator(integer=True))
    axs.tick_params(axis='both', which='major', labelsize=size-5)

    if title_name:
        axs.set_title(title_name, fontsize=size)

    fig.patch.set_facecolor("white")
    return fig, axs
