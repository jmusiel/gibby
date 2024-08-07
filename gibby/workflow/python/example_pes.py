# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import numpy as np
from ase import Atoms
from ase import units
from ase.io import read
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from ase.thermochemistry import HarmonicThermo

from gibby.utils.potential_energy_sampling import PotentialEnergySampling, PESThermo

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Parameters.
    spacing = 0.50
    spacing_surrogate = 0.05
    reduce_cell = True
    distance = 2.
    all_hookean = True
    fix_com = True
    index = 0
    fmax = 0.03
    n_rotations = 1
    kwargs_opt = {}
    scipy_integral = False
    trajectory = "atoms_pes.traj"
    temperature = 300 # [K]
    show_plot = False
    save_plot = True
    sklearn_model = None
    entropies_plot = True

    # Read trajectory with updated energies.
    read_trajectory = False
    trajectory_new = "atoms_pes.traj"

    # Atoms objects of slab and adsorbate.
    from ase.build import fcc111
    from ase.constraints import FixAtoms
    slab = fcc111('Pt', size=(3, 3, 4), vacuum=6, a=3.92)
    indices = [aa.index for aa in slab]
    slab.constraints = FixAtoms(indices=indices)
    ads = Atoms("CO", positions=[[0., 0., 0.], [0., 0., 1.3]])
    indices_surf = range(27, 36)
    indices_site = [27, 28, 30]
    
    # Ase calculator.
<<<<<<< HEAD
    from ocpmodels.common.relaxation.ase_utils import OCPCalculator

    checkpoint_path = (
        "/home/jovyan/checkpoints/eq2_31M_ec4_allmd.pt"
    )
    calc = OCPCalculator(checkpoint_path=checkpoint_path, cpu=False)

=======
    if read_trajectory is True:
        calc = None
    else:
        from ocpmodels.common.relaxation.ase_utils import OCPCalculator
        checkpoint_path = (
            "/home/jovyan/PythonLibraries/arkimede/checkpoints/eq2_31M_ec4_allmd.pt"
        )
        calc = OCPCalculator(checkpoint_path=checkpoint_path, cpu=False)
    
>>>>>>> ac866497c9d4938aed51a05d713b18cff36b3e4b
    # Add adsorbate.
    slab_opt = slab.copy()
    ads_opt = ads.copy()
    position = np.average([slab_opt.positions[ii] for ii in indices_site], axis=0)
    ads_opt.translate(position+[0., 0., distance])
    slab_opt += ads_opt
    
    # Optimize structure.
    slab_opt.calc = calc
    if os.path.isfile("atoms_relaxed.traj"):
        slab_opt = read("atoms_relaxed.traj")
    else:
        opt = BFGS(atoms=slab_opt, **kwargs_opt)
        opt.run(fmax=fmax)
        slab_opt.write("atoms_relaxed.traj")
    e_min = slab_opt.get_potential_energy()
    
    # Run vibrations.
    indices_vib = [aa.index+len(slab) for aa in ads_opt]
    vib = Vibrations(atoms=slab_opt, indices=indices_vib, delta=0.01, nfree=2)
    vib.clean(empty_files=True)
    vib.run()
    
    # Harmonic approximation thermo.
    thermo_ha = HarmonicThermo(vib_energies=vib.get_energies())
    entropy = thermo_ha.get_entropy(temperature=temperature, verbose=True)
    print(f"HarmonicThermo entropy: {entropy*1e3:+7.4f} [meV/K]")
    
    # Potential Energy Sampling.
    pes = PotentialEnergySampling(
        slab=slab,
        ads=ads,
        calc=calc,
        indices_surf=indices_surf,
        distance=distance,
        all_hookean=all_hookean,
        e_min=e_min,
        spacing=spacing,
        spacing_surrogate=spacing_surrogate,
        reduce_cell=reduce_cell,
        n_rotations=n_rotations,
        fix_com=fix_com,
        index=index,
        fmax=fmax,
        kwargs_opt=kwargs_opt,
        scipy_integral=scipy_integral,
        trajectory=trajectory,
    )
    pes.clean(empty_files=True)
    if read_trajectory is True:
        pes.read(trajectory=trajectory_new)
    else:
        pes.run()
    pes.surrogate_pes(sklearn_model=sklearn_model)
    entropy = pes.get_entropy_pes(temperature=temperature)
    print(f"PotentialEnergySampling entropy: {entropy*1e3:+7.4f} [meV/K]")
    
    if show_plot is True:
        pes.show_surrogate_pes()
    
    if save_plot is True:
        pes.save_surrogate_pes(filename="pes.png")

    # Potential Energy Sampling thermo.
    thermo_pes = PESThermo(
        vib_energies=vib.get_energies(),
        pes=pes,
    )
    entropy = thermo_pes.get_entropy(temperature=temperature, verbose=True)
    print(f"PESThermo entropy: {entropy*1e3:+7.4f} [meV/K]")
    
    if entropies_plot is True:
        from gibby.utils.potential_energy_sampling import plot_entropies
        plot_entropies(
            thermo_dict={"HA": thermo_ha, "PES": thermo_pes},
            temperature_range=[200, 1000],
            filename="entropies.png",
        )

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
