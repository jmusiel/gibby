# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from ase import units
from ase.parallel import world
from ase.utils.filecache import get_json_cache
from ase.optimize import BFGS
from ase.thermochemistry import HarmonicThermo

# -------------------------------------------------------------------------------------
# GET 1X1 SLAB CELL
# -------------------------------------------------------------------------------------

def get_1x1_slab_cell(atoms, symprec=1e-7, primitive=True):
    
    from ase.spacegroup.symmetrize import check_symmetry
    from ase.cell import Cell
    
    dataset = check_symmetry(atoms, symprec=symprec)
    transl = dataset["translations"]
    xx_vect = [
        vv[0] for vv in transl
        if abs(vv[0]) > symprec if abs(vv[1]) < symprec if abs(vv[2]) < symprec
    ]
    yy_vect = [
        vv[1] for vv in transl
        if abs(vv[0]) < symprec if abs(vv[1]) > symprec if abs(vv[2]) < symprec
    ]
    xx_fract = np.min(xx_vect) if len(xx_vect) > 0 else 1.
    yy_fract = np.min(yy_vect) if len(yy_vect) > 0 else 1.
    
    return Cell(np.dot(atoms.cell, np.identity(3)*[xx_fract, yy_fract, 1]))

# -------------------------------------------------------------------------------------
# GET MESHGRID
# -------------------------------------------------------------------------------------

def get_meshgrid(cell, height, zz_function=None, spacing=0.2):

    nx = int(np.ceil(cell.lengths()[0]/spacing))
    ny = int(np.ceil(cell.lengths()[1]/spacing))
    xr_vect = np.linspace(0, 1, nx+1)
    yr_vect = np.linspace(0, 1, ny+1)
    zr = height/cell.lengths()[2]

    xx_grid = np.zeros([len(xr_vect), len(yr_vect)])
    yy_grid = np.zeros([len(xr_vect), len(yr_vect)])
    zz_grid = np.zeros([len(xr_vect), len(yr_vect)])
    for ii, xr in enumerate(xr_vect):
        for jj, yr in enumerate(yr_vect):
            xx, yy, zz = np.dot([xr, yr, zr], cell)
            if zz_function is not None:
                zz = zz_function(xx, yy)
            xx_grid[ii, jj] = xx
            yy_grid[ii, jj] = yy
            zz_grid[ii, jj] = zz

    return xx_grid, yy_grid, zz_grid

# -------------------------------------------------------------------------------------
# GET XYZ POINTS
# -------------------------------------------------------------------------------------

def get_xyz_points(cell, height, zz_function=None, spacing=0.2):

    nx = int(np.ceil(cell.lengths()[0]/spacing))
    ny = int(np.ceil(cell.lengths()[1]/spacing))
    xr_vect = np.linspace(0, 1, nx+1)
    yr_vect = np.linspace(0, 1, ny+1)
    zr = height/cell.lengths()[2]

    xyz_points = np.zeros([len(xr_vect)*len(yr_vect), 3])
    for ii, xr in enumerate(xr_vect):
        for jj, yr in enumerate(yr_vect):
            xx, yy, zz = np.dot([xr, yr, zr], cell)
            if zz_function is not None:
                zz = zz_function(xx, yy)
            xyz_points[ii*len(yr_vect)+jj] = xx, yy, zz

    return xyz_points

# -------------------------------------------------------------------------------------
# CONSTRAINED RELAXATION
# -------------------------------------------------------------------------------------

def constrained_relaxation(
    slab,
    ads,
    position,
    calc,
    fix_com=False,
    index=0,
    optimizer=BFGS,
    kwargs_opt={},
    fmax=0.01,
):
    
    slab_new = slab.copy()
    ads_new = ads.copy()
    ads_new.translate(position)
    
    slab_new += ads_new
    indices = [aa.index for aa in slab_new if aa.index >= len(slab)]
    
    if fix_com is True:
        from ase.constraints import FixSubsetCom
        slab_new.constraints.append(FixSubsetCom(indices=indices))
    else:
        from ase.constraints import FixCartesian
        slab_new.constraints.append(FixCartesian(a=indices[index], mask=(1,1,0)))

    slab_new.calc = calc
    opt = optimizer(atoms=slab_new, **kwargs_opt)
    opt.run(fmax=fmax)

    return slab_new

# -------------------------------------------------------------------------------------
# POTENTIAL ENERGY SAMPLING
# -------------------------------------------------------------------------------------

class PotentialEnergySampling:
    
    def __init__(
        self,
        slab,
        ads,
        calc,
        height,
        e_min,
        spacing=0.20,
        spacing_surr=0.05,
        reduce_cell=True,
        fix_com=False,
        index=0,
        fmax=0.01,
        kwargs_opt={},
        scipy_integral=False,
        name="pes",
    ):
        self.slab = slab.copy()
        self.ads = ads.copy()
        self.calc = calc
        self.height = height
        self.e_min = e_min
        self.spacing = spacing
        self.spacing_surr = spacing_surr
        self.reduce_cell = reduce_cell
        self.fix_com = fix_com
        self.index = index
        self.fmax = fmax
        self.kwargs_opt = kwargs_opt
        self.scipy_integral = scipy_integral
        
        self.cache = get_json_cache(name)

    @property
    def name(self):
        return str(self.cache.directory)

    def run(self):
        """Run Potential Energy Sampling method."""
    
        # Set the x axis parallel to [1,0,0]
        angle = np.arctan(self.slab.cell[0,1]/self.slab.cell[0,0])*180/np.pi
        self.slab.rotate(-angle, 'z', rotate_cell=True)

        # Reduced cell.
        self.cell = self.slab.cell
        if self.reduce_cell is True:
            self.cell = get_1x1_slab_cell(atoms=self.slab)

        # Get grid of points from spacing.
        xyz_points = get_xyz_points(
            cell=self.cell,
            height=self.height,
            spacing=self.spacing,
            zz_function=None,
        )

        # Do constrained relaxations.
        xye_points = xyz_points.copy()
        for ii, position in enumerate(xyz_points):
            with self.cache.lock(f"{ii:04d}") as handle:
                if handle is None:
                    xye_points[ii] = self.cache[f"{ii:04d}"]
                    continue
                slab_new = constrained_relaxation(
                    slab=self.slab,
                    ads=self.ads,
                    position=position,
                    calc=self.calc,
                    fix_com=self.fix_com,
                    index=self.index,
                    kwargs_opt=self.kwargs_opt,
                )
                xye_points[ii,2] = slab_new.get_potential_energy()
                if world.rank == 0:
                    handle.save(xye_points[ii])

        self.xye_points = xye_points
        self.es_grid = None

        return xye_points

    def clean(self, empty_files=False):
        """Remove json files."""
        if world.rank != 0:
            return 0
        if empty_files:
            self.cache.strip_empties()
        else:
            self.cache.clear()

    def surrogate_pes(self, model=None):
        """Train the surrogate model for the PES."""
        if model is None:
            from scipy.interpolate import griddata
            self.e_func = lambda xx, yy: griddata(
                points=self.xye_points[:,:2],
                values=self.xye_points[:,2],
                xi=[xx, yy],
                method='cubic',
                rescale=False,
            )
        else:
            model.fit(self.xye_points[:,:2], self.xye_points[:,2])
            self.e_func = lambda xx, yy: model.predict([[xx, yy]])[0]
    
    def get_meshgrid_surrogate(self):
        """Get mesh grid of PES from surrogate model."""
        self.xs_grid, self.ys_grid, self.es_grid = get_meshgrid(
            cell=self.cell,
            height=self.height,
            spacing=self.spacing_surr,
            zz_function=self.e_func,
        )
    
    def save_surrogate_pes(self, filename="pes.png"):
        """Save 2D plot of PES to file."""
        import matplotlib.pyplot as plt
        if self.es_grid is None:
            self.get_meshgrid_surrogate()
        plt.pcolor(self.xs_grid, self.ys_grid, self.es_grid)
        plt.savefig(filename)

    def show_surrogate_pes(self):
        """Show 3D plot of PES function."""
        import matplotlib.pyplot as plt
        if self.es_grid is None:
            self.get_meshgrid_surrogate()
        ax = plt.figure().add_subplot(projection='3d')
        ax.plot_surface(self.xs_grid, self.ys_grid, self.es_grid)
        ax.scatter(
            self.xye_points[:,0],
            self.xye_points[:,1],
            self.xye_points[:,2],
        )
        plt.show()

    def get_integral_pes_scipy(self, temperature):
        """Integrate the function of the PES with scipy."""
        from scipy.integrate import dblquad
        func = lambda yy, xx: (
            np.exp(-(self.e_func(yy, xx)-self.e_min)/(units.kB*temperature))
        )
        integral_cpes = dblquad(
            func=func,
            a=0.,
            b=self.cell[1, 1],
            gfun=lambda xx: self.cell[1,0]/self.cell[1,1]*xx,
            hfun=lambda xx: self.cell[1,0]/self.cell[1,1]*xx + self.cell[0,0],
        )[0]
    
        return integral_cpes
    
    def get_integral_pes_grid(self, temperature):
        """Integrate the function of the PES with finite differences."""
        if self.es_grid is None:
            self.get_meshgrid_surrogate()
        vv_grid = np.exp(-(self.es_grid-self.e_min)/(units.kB*temperature))
        integral_cpes = np.average(vv_grid)*self.cell[0,0]*self.cell[1,1]
    
        return integral_cpes
    
    def get_entropy_pes(self, temperature):
        """Calculate the entropy from the PES."""
        if self.scipy_integral is True:
            integral_cpes = self.get_integral_pes_scipy(temperature)
        else:
            integral_cpes = self.get_integral_pes_grid(temperature)
        
        hP = units._hplanck*units.kJ*1e-3
        mass = sum(self.ads.get_masses())/units.kg
        part_fun = 2*np.pi*mass*units.kB*temperature*integral_cpes/(hP**2)
        entropy = units.kB*np.log(part_fun)

        return entropy

# -------------------------------------------------------------------------------------
# PES THERMO
# -------------------------------------------------------------------------------------

class PESThermo(HarmonicThermo):

    def __init__(
        self,
        pes,
        vib_energies,
        potentialenergy=0.,
    ):
        self.pes = pes
        # Remove the 2 lowest vibrational energies.
        super().__init__(
            vib_energies=vib_energies[2:],
            potentialenergy=potentialenergy,
        )

    def get_entropy(self, temperature, verbose=True):
        """Returns the entropy, in eV/K, with potential energy sampling
        at a specified temperature (K)."""

        self.verbose = verbose
        write = self._vprint
        fmt = '%-15s%13.7f eV/K%13.3f eV'
        write('Entropy components at T = %.2f K:' % temperature)
        write('=' * 49)
        write('%15s%13s     %13s' % ('', 'S', 'T*S'))

        S = 0.

        S_v = self._vibrational_entropy_contribution(temperature)
        write(fmt % ('S_harm', S_v, S_v * temperature))
        S += S_v

        S_p = self.pes.get_entropy_pes(temperature)
        write(fmt % ('S_pes', S_p, S_p * temperature))
        S += S_p

        write('-' * 49)
        write(fmt % ('S', S, S * temperature))
        write('=' * 49)
        return S

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
