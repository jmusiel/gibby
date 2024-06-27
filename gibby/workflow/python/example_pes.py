# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from ase import units
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from ase.thermochemistry import HarmonicThermo

from gibby.utils.potential_energy_sampling import PotentialEnergySampling, PESThermo

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    height = 14.5
    spacing = 0.50
    spacing_surr = 0.05
    reduce_cell = True
    fix_com = False
    index = 0
    fmax = 0.03
    kwargs_opt = {}
    kwargs_surr = {"random_state": 0}
    scipy_integral = False
    temperature = 300 # [K]
    save_plot = True

    from ase.build import fcc111
    slab = fcc111('Pt', size=(3, 3, 4), vacuum=6)
    
    from ase import Atoms
    ads = Atoms("O")
    
    #from ase.calculators.emt import EMT
    #calc = EMT()
    
    from ocpmodels.common.relaxation.ase_utils import OCPCalculator
    checkpoint_path = (
        "/home/jovyan/PythonLibraries/arkimede/checkpoints/eq2_31M_ec4_allmd.pt"
    )
    calc = OCPCalculator(checkpoint_path=checkpoint_path, cpu=False)
    
    # Add adsorbate.
    slab_opt = slab.copy()
    ads_opt = ads.copy()
    position = np.dot([1/9, 1/9, 0], slab_opt.cell)
    position[2] = height
    ads_opt.translate(position)
    slab_opt += ads_opt
    
    # Optimize structure.
    slab_opt.calc = calc
    opt = BFGS(atoms=slab_opt, **kwargs_opt)
    opt.run(fmax=fmax)
    e_min = slab_opt.get_potential_energy()
    
    # Run vibrations.
    indices = [aa.index+len(slab) for aa in ads_opt]
    vib = Vibrations(atoms=slab_opt, indices=indices, delta=0.01, nfree=2)
    vib.run()
    
    # Harmonic approximation thermo.
    thermo = HarmonicThermo(vib_energies=vib.get_energies())
    entropy = thermo.get_entropy(temperature=temperature, verbose=False)
    print(f"HarmonicThermo entropy: {entropy*1e3:+7.4f} [meV/K]")
    
    # Potential Energy Sampling.
    pes = PotentialEnergySampling(
        slab=slab,
        ads=ads,
        calc=calc,
        height=height,
        e_min=e_min,
        spacing=spacing,
        spacing_surr=spacing_surr,
        reduce_cell=reduce_cell,
        fix_com=fix_com,
        index=index,
        fmax=fmax,
        kwargs_opt=kwargs_opt,
        kwargs_surr=kwargs_surr,
        scipy_integral=scipy_integral,
    )
    pes.run()
    pes.surrogate_pes()
    entropy = pes.get_entropy_pes(temperature=temperature)
    print(f"PotentialEnergySampling entropy: {entropy*1e3:+7.4f} [meV/K]")
    
    if save_plot is True:
        pes.save_surrogate_pes(filename="pes.png")

    # Potential Energy Sampling thermo.
    thermo = PESThermo(
        vib_energies=vib.get_energies(),
        pes=pes,
    )
    entropy = thermo.get_entropy(temperature=temperature)
    print(f"PESThermo entropy: {entropy*1e3:+7.4f} [meV/K]")
    
# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
