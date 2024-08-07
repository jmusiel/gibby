from ase.vibrations import Vibrations
from ase.thermochemistry import HarmonicThermo
import argparse
import pickle
import numpy as np
import os

from ocpmodels.common.relaxation.ase_utils import OCPCalculator
from gibby.utils.potential_energy_sampling import PotentialEnergySampling, PESThermo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Args that must be provided.
    parser.add_argument("--sys_id", type=int)
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--outdir", type=str)
    parser.add_argument("--checkpoint_path", type=str)

    # Args with default values.
    parser.add_argument("--spacing", type=float, default=0.50)
    parser.add_argument("--spacing_surrogate", type=float, default=0.05)
    parser.add_argument("--reduce_cell", type=bool, default=True)
    parser.add_argument("--fix_com", type=bool, default=True)
    parser.add_argument("--fmax", type=int, default=0.03)
    parser.add_argument(
        "--n_rotations", type=int, default=6
    ) 
    parser.add_argument("--kwargs_opt", type=dict, default={})
    parser.add_argument("--scipy_integral", type=bool, default=False)
    parser.add_argument("--temperature", type=int, default=300)
    parser.add_argument("--show_plot", default=False, action='store_true')
    parser.add_argument("--save_plot", default=False, action='store_true')
    parser.add_argument("--sklearn_model", type=None, default=None)
    parser.add_argument("--gpu", default=False, action='store_true')
    args = parser.parse_args()

    calc = OCPCalculator(checkpoint_path=str(args.checkpoint_path), cpu=not args.gpu)
    # TODO: add import of the adsorbate db to extract the adsorbate binding index? or is COM approach better?

    # Prep outputs.
    os.makedirs(args.outdir, exist_ok=True)
    outdir = os.path.join(args.outdir, f"{args.sys_id}")
    os.makedirs(outdir, exist_ok=True)

    entry = {}

    # Load the adslab and attach the calculator.
    with open(args.input_file, "rb") as f:
        adslabs = pickle.load(f)
    adslab = adslabs[args.sys_id]
    adslab.calc = calc

    # Run vibration calculation.
    indices_vib = [atom.index for atom in adslab if atom.tag == 2]
    vib = Vibrations(atoms=adslab, indices=indices_vib, delta=0.0075, nfree=2)
    vib.clean(empty_files=True)
    vib.run()

    # Harmonic approximation thermo.
    thermo = HarmonicThermo(vib_energies=[e for e in vib.get_energies() if not np.iscomplex(e)])
    entropy = thermo.get_entropy(temperature=args.temperature, verbose=True)
    print(f"HarmonicThermo entropy: {entropy*1e3:+7.4f} [meV/K]")
    entry["harmonic_approx_entropy"] = entropy

    # Run potential energy sampling.
    pes = PotentialEnergySampling(
        slab=adslab[[atom.index for atom in adslab if atom.tag != 2]].copy(),
        ads=adslab[indices_vib].copy(),
        calc=calc,
        indices_surf=[atom.index for atom in adslab if atom.tag == 1],
        e_min=adslab.get_potential_energy(),
        spacing=args.spacing,
        spacing_surrogate=args.spacing_surrogate,
        reduce_cell=args.reduce_cell,
        n_rotations=args.n_rotations,
        fix_com=args.fix_com,
        index=None,
        fmax=args.fmax,
        kwargs_opt=args.kwargs_opt,
        scipy_integral=args.scipy_integral,
        trajectory=os.path.join(outdir, "all_min_e_relaxed_systems.traj"),
    )
    pes.clean(empty_files=True)
    pes.run()
    pes.surrogate_pes(sklearn_model=args.sklearn_model)
    entropy = pes.get_entropy_pes(temperature=args.temperature)
    print(f"PotentialEnergySampling entropy: {entropy*1e3:+7.4f} [meV/K]")
    entry["pes_entropy"] = entropy

    # Save the results.
    if args.show_plot is True:
        pes.show_surrogate_pes()
    if args.save_plot is True:
        pes.save_surrogate_pes(filename=os.join(outdir, "pes.png"))

    # Potential Energy Sampling thermo.
    thermo = PESThermo(
        vib_energies=vib.get_energies(),
        pes=pes,
    )
    entropy = thermo.get_entropy(temperature=args.temperature, verbose=True)
    print(f"PESThermo entropy: {entropy*1e3:+7.4f} [meV/K]")
    entry["pes_thermo_entropy"] = entropy

    # Save the results.
    entry["x_grid"] = pes.xs_grid
    entry["y_grid"] = pes.ys_grid
    entry["e_grid"] = pes.es_grid

    with open(os.path.join(outdir, "results.pkl"), "wb") as f:
        pickle.dump(entry, f)
