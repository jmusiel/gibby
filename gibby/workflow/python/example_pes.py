# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from ase import Atoms
from ase import units
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

    # Atoms objects of slab and adsorbate.
    from ase.build import fcc111
    from ase.constraints import FixAtoms
    slab = fcc111('Pt', size=(3, 3, 4), vacuum=6, a=3.92)
    indices = [aa.index for aa in slab]
    slab.constraints = FixAtoms(indices=indices)
    indices_surf = range(27, 36)
    ads = Atoms("CO", positions=[[0., 0., 0.], [0., 0., 1.3]])
    index_top = 27
    #from ase.build import fcc211
    #from ase.constraints import FixAtoms
    #slab = fcc211('Pt', size=(3, 3, 4), vacuum=6, a=3.92)
    #indices_fix = [aa.index for aa in slab]
    #slab.constraints = FixAtoms(indices=indices_fix)
    #indices_surf = range(0, 9)
    #ads = Atoms("CO", positions=[[0., 0., 0.], [0., 0., 1.3]])
    #index_top = 0
    
    # Ase calculator.
    from ocpmodels.common.relaxation.ase_utils import OCPCalculator
    checkpoint_path = (
        "/home/jovyan/PythonLibraries/arkimede/checkpoints/eq2_31M_ec4_allmd.pt"
    )
    calc = OCPCalculator(checkpoint_path=checkpoint_path, cpu=False)
    
    # Add adsorbate.
    slab_opt = slab.copy()
    ads_opt = ads.copy()
    ads_opt.translate(slab_opt[index_top].position+[0., 0., distance])
    slab_opt += ads_opt
    
    # Optimize structure.
    slab_opt.calc = calc
    opt = BFGS(atoms=slab_opt, **kwargs_opt)
    opt.run(fmax=fmax)
    e_min = slab_opt.get_potential_energy()
    
    # Run vibrations.
    indices_vib = [aa.index+len(slab) for aa in ads_opt]
    vib = Vibrations(atoms=slab_opt, indices=indices_vib, delta=0.01, nfree=2)
    vib.clean(empty_files=True)
    vib.run()
    
    # Harmonic approximation thermo.
    thermo = HarmonicThermo(vib_energies=vib.get_energies())
    entropy = thermo.get_entropy(temperature=temperature, verbose=True)
    print(f"HarmonicThermo entropy: {entropy*1e3:+7.4f} [meV/K]")
    
    # Potential Energy Sampling.
    pes = PotentialEnergySampling(
        slab=slab,
        ads=ads,
        calc=calc,
        indices_surf=indices_surf,
        distance=distance,
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
    pes.run()
    pes.surrogate_pes(sklearn_model=sklearn_model)
    entropy = pes.get_entropy_pes(temperature=temperature)
    print(f"PotentialEnergySampling entropy: {entropy*1e3:+7.4f} [meV/K]")
    
    if show_plot is True:
        pes.show_surrogate_pes()
    
    if save_plot is True:
        pes.save_surrogate_pes(filename="pes.png")

    # Potential Energy Sampling thermo.
    thermo = PESThermo(
        vib_energies=vib.get_energies(),
        pes=pes,
    )
    entropy = thermo.get_entropy(temperature=temperature, verbose=True)
    print(f"PESThermo entropy: {entropy*1e3:+7.4f} [meV/K]")
    
# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
