# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from ase import units
from ase.io import Trajectory
from ase.parallel import world
from ase.constraints import FixConstraint, FixCartesian, FixAtoms
from ase.build.tools import rotation_matrix
from ase.utils.filecache import get_json_cache
from ase.optimize import BFGS
from ase.thermochemistry import HarmonicThermo
from scipy.interpolate import griddata

# -------------------------------------------------------------------------------------
# GET 1X1 SLAB CELL
# -------------------------------------------------------------------------------------


def get_1x1_slab_cell(atoms, symprec=1e-7, repetitions=None):
    """Get the 1x1 cell of an ase.Atoms object by checking the translational
    symmetries or by using the provided list of repetitions in x and y directions.

    Args:
        atoms (ase.Atoms): ase.Atoms object.
        symprec (float, optional): precision for the calculation of symmetries
        with ase.spacegroup.symmetrize.check_symmetry. Defaults to 1e-7.
        repetitions (list, optional): list of repetitions in x and y directions
        used to produce a NxN slab from a 1x1 slab. Defaults to None.

    Returns:
        ase.cell.Cell: cell reduced by the translational symmetries.
    """
    from ase.spacegroup.symmetrize import check_symmetry
    from ase.cell import Cell

    if repetitions is None:
        dataset = check_symmetry(atoms, symprec=symprec)
        transl = dataset["translations"]
        xx_vect = [
            vv[0]
            for vv in transl
            if abs(vv[0]) > symprec
            if abs(vv[1]) < symprec
            if abs(vv[2]) < symprec
        ]
        yy_vect = [
            vv[1]
            for vv in transl
            if abs(vv[0]) < symprec
            if abs(vv[1]) > symprec
            if abs(vv[2]) < symprec
        ]
        xx_fract = np.min(xx_vect) if len(xx_vect) > 0 else 1.0
        yy_fract = np.min(yy_vect) if len(yy_vect) > 0 else 1.0
    else:
        xx_fract = 1 / repetitions[0]
        yy_fract = 1 / repetitions[1]

    return Cell(np.dot(atoms.cell, np.identity(3) * [xx_fract, yy_fract, 1]))


# -------------------------------------------------------------------------------------
# GET MESHGRID
# -------------------------------------------------------------------------------------


def get_meshgrid(cell, height, z_func=None, spacing=0.5):
    """Get a meshgrid of [x, y, z] points from an atoms.cell object and a grid spacing.
    If z_func is None, z is equal to height.

    Args:
        cell (ase.cell.Cell): cell of the atoms object.
        height (float): height (z axis) of the adsorbate atoms.
        z_func (function, optional): function of x and y to use instead of the
        adsorbate height. Defaults to None.
        spacing (float, optional): spacing of the 2d grid of positions of the
        adsorbates. Defaults to 0.50.

    Returns:
        list[numpy.ndarray]: list of meshgrid numpy arrays (x_grid, y_grid, z_grid).
    """
    nx = int(np.ceil(cell.lengths()[0] / spacing))
    ny = int(np.ceil(cell.lengths()[1] / spacing))
    xr_vect = np.linspace(0, 1 - 1 / nx, nx)
    yr_vect = np.linspace(0, 1 - 1 / ny, ny)
    zr = height / cell.lengths()[2]

    xx_grid = np.zeros([len(xr_vect), len(yr_vect)])
    yy_grid = np.zeros([len(xr_vect), len(yr_vect)])
    zz_grid = np.zeros([len(xr_vect), len(yr_vect)])
    for ii, xr in enumerate(xr_vect):
        for jj, yr in enumerate(yr_vect):
            xx, yy, zz = np.dot([xr, yr, zr], cell)
            if z_func is not None:
                zz = z_func(xx, yy)
            xx_grid[ii, jj] = xx
            yy_grid[ii, jj] = yy
            zz_grid[ii, jj] = zz

    return xx_grid, yy_grid, zz_grid


# -------------------------------------------------------------------------------------
# GET XYZ POINTS
# -------------------------------------------------------------------------------------


def get_xyz_points(cell, height, z_func=None, spacing=0.5):
    """Get an array of [x, y, z] points from an atoms.cell object and a grid spacing.
    If z_func is None, z is equal to height.

    Args:
        cell (ase.cell.Cell): cell of the atoms object.
        height (float): height (z axis) of the adsorbate atoms.
        z_func (function, optional): function of x and y to use instead of the
        surface height. Defaults to None.
        spacing (float, optional): spacing of the 2d grid of positions of the
        adsorbates. Defaults to 0.50.

    Returns:
        numpy.ndarray: array of [x, y, z] points.
    """
    nx = int(np.ceil(cell.lengths()[0] / spacing))
    ny = int(np.ceil(cell.lengths()[1] / spacing))
    xr_vect = np.linspace(0, 1 - 1 / nx, nx)
    yr_vect = np.linspace(0, 1 - 1 / ny, ny)
    zr = height / cell.lengths()[2]

    xyz_points = np.zeros([len(xr_vect) * len(yr_vect), 3])
    for ii, xr in enumerate(xr_vect):
        for jj, yr in enumerate(yr_vect):
            xx, yy, zz = np.dot([xr, yr, zr], cell)
            if z_func is not None:
                zz = z_func(xx, yy)
            xyz_points[ii * len(yr_vect) + jj] = xx, yy, zz

    return xyz_points


# -------------------------------------------------------------------------------------
# EXTEND XYZ POINTS
# -------------------------------------------------------------------------------------


def extend_xyz_points(xyz_points, cell, border=3.0):
    """Extend the grid points to a border surrounding the atoms.cell object in x
    and y axes.

    Args:
        xye_points (numpy.ndarray): [x, y, z] points.
        cell (ase.cell.Cell): cell of the atoms object.
        border (float, optional): border surrounding the atoms cell in which the
        [x, y, z] points are extended to. Defaults to 3 Angstrom.

    Returns:
        numpy.ndarray: array of [x, y, z] points extended.
    """
    xyz_points_ext = xyz_points.copy()
    for ii in (-1, 0, +1):
        for jj in (-1, 0, +1):
            if (ii, jj) == (0, 0):
                continue
            translation = np.hstack([ii * cell[0, :2] + jj * cell[1, :2], [0.0]])
            xyz_points_copy = xyz_points.copy()
            xyz_points_copy += translation
            xyz_points_ext = np.vstack([xyz_points_ext, xyz_points_copy])

    del_indices = []
    for ii, (xx, yy, zz) in enumerate(xyz_points_ext):
        if (
            yy < -border
            or yy > cell[1, 1] + border
            or xx < cell[1, 0] / cell[1, 1] * yy - border
            or xx > cell[1, 0] / cell[1, 1] * yy + cell[0, 0] + border
        ):
            del_indices += [ii]
    xyz_points_ext = np.delete(xyz_points_ext, del_indices, axis=0)

    return xyz_points_ext


# -------------------------------------------------------------------------------------
# CONSTRAINED RELAXATIONS WITH ROTATIONS
# -------------------------------------------------------------------------------------


def constrained_relaxations_with_rotations(
    slab,
    ads,
    position,
    calc,
    n_rotations=1,
    distance=2.0,
    fix_com=False,
    index=0,
    optimizer=BFGS,
    kwargs_opt={},
    fmax=0.01,
    z_func=None,
    delta=0.05,
):
    """Perform multiple constrained relaxation at different adsorbate rotations.
    The x and y of the centre of mass (fix_com=True) or of the Nth atom of the
    adsorbate (index=N) are fixed.

    Args:
        slab (ase.Atoms): slab atoms.
        ads (ase.Atoms): adsorbate atoms.
        position (numpy.ndarray, list): position (x, y, z) of the adsorbate.
        calc (ase.calculators.Calculator): ase calculator.
        n_rotations (int, optional): number of rotations sampled. Defaults to 1.
        distance (float, optional): distance of the adsorbate from the surface.
        Defaults to 2.
        fix_com (bool, optional): fix centre of mass. Defaults to False.
        index (int, optional): index of the adsorbate to fix. Defaults to 0.
        optimizer (ase.optimize.Optimizer, optional): optimizer for constrained
        relaxation. Defaults to BFGS.
        kwargs_opt (dict, optional): dictionary of options for the optimizer.
        Defaults to {}.
        fmax (float, optional): maximum forces for convergence of constrained
        relaxation. Defaults to 0.01.
        z_func (function, optional): function of x and y to use instead of the
        adsorbate height. Defaults to None.
        delta (float, optional): small length to calculate derivative of z_func
        with respect to x and y. Defaults to 0.05.

    Returns:
        ase.Atoms: slab + adsorbate atoms relaxed.
    """

    xx, yy, zz = position

    # Calculate rotation_matrix to rotate in the direction normal to the surface.
    if z_func is not None:
        zz = z_func(xx, yy)[0]
        dz_dx = (z_func(xx + delta, yy)[0] - z_func(xx - delta, yy)[0]) / (2 * delta)
        dz_dy = (z_func(xx, yy + delta)[0] - z_func(xx, yy - delta)[0]) / (2 * delta)
        rot_matrix = rotation_matrix(
            a1=[1, 0, 0],
            a2=[1, 0, dz_dx],
            b1=[0, 1, 0],
            b2=[0, 1, dz_dy],
        )
    else:
        rot_matrix = None

    # Do multiple constrained optimizations with different initial rotations.
    atoms_list = []
    for rr in range(n_rotations):
        slab_new = slab.copy()
        ads_new = ads.copy()
        ads_new.rotate(rr * 360 / n_rotations, "z")
        if rot_matrix is not None:
            ads_new.positions[:] = np.dot(ads_new.positions, rot_matrix.T)
        ads_new.translate([xx, yy, zz + distance])
        atoms = constrained_relaxation(
            slab=slab_new,
            ads=ads_new,
            calc=calc,
            index=index,
            fix_com=fix_com,
            optimizer=optimizer,
            kwargs_opt=kwargs_opt,
            fmax=fmax,
        )
        atoms_list.append(atoms)

    index = np.argmin([atoms.get_potential_energy() for atoms in atoms_list])

    return atoms_list[index]


# -------------------------------------------------------------------------------------
# CONSTRAINED RELAXATION
# -------------------------------------------------------------------------------------


def constrained_relaxation(
    slab,
    ads,
    calc,
    index=0,
    fix_com=False,
    optimizer=BFGS,
    kwargs_opt={},
    fmax=0.01,
):
    """Perform a constrained relaxation. The x and y of the centre of mass
    (fix_com=True) or of the Nth atom of the adsorbate (index=N) are fixed.

    Args:
        slab (ase.Atoms): slab atoms.
        ads (ase.Atoms): adsorbate atoms.
        calc (ase.calculators.Calculator): ase calculator.
        fix_com (bool, optional): fix centre of mass. Defaults to False.
        index (int, optional): index of the adsorbate to fix. Defaults to 0.
        optimizer (ase.optimize.Optimizer, optional): optimizer for constrained
        relaxation. Defaults to BFGS.
        kwargs_opt (dict, optional): dictionary of options for the optimizer.
        Defaults to {}.
        fmax (float, optional): maximum forces for convergence of constrained
        relaxation. Defaults to 0.01.

    Returns:
        ase.Atoms: slab + adsorbate atoms relaxed.
    """

    atoms = slab + ads
    indices = [aa.index for aa in atoms if aa.index >= len(slab)]

    if fix_com is True:
        atoms.constraints.append(FixSubsetCom(indices=indices, mask=[1, 1, 0]))
    else:
        atoms.constraints.append(FixCartesian(a=indices[index], mask=[1, 1, 0]))

    atoms.calc = calc
    opt = optimizer(atoms=atoms, **kwargs_opt)
    opt.run(fmax=fmax)

    return atoms


# -------------------------------------------------------------------------------------
# POTENTIAL ENERGY SAMPLING
# -------------------------------------------------------------------------------------


class PotentialEnergySampling:
    def __init__(
        self,
        slab,
        ads,
        calc=None,
        height=None,
        indices_surf=None,
        distance=2.0,
        e_min=None,
        spacing=0.20,
        spacing_surrogate=0.05,
        reduce_cell=True,
        repetitions=None,
        border=3.0,
        n_rotations=1,
        fix_com=False,
        index=0,
        fmax=0.01,
        optimizer=BFGS,
        kwargs_opt={},
        delta=0.05,
        scipy_integral=False,
        trajectory=None,
        trajmode="w",
        name="pes",
    ):
        """Class to do a potential energy sampling calculation with constrained
        relaxations on a grid of positions (x, y). Used to evaluate the entropy
        of adsorbates that can translate on a surface.

        Args:
            slab (ase.Atoms): ase.Atoms object of the slab.
            ads (ase.Atoms): ase.Atoms object of the adsorbate.
            calc (ase.calculators.Calculator): ase calculator. Defaults to None.
            height (float, optional): height (z axis), in Angstrom, of the surface
                atoms. Defaults to None.
            indices_surf (list, optional): list of indices of the surface atoms.
            distance (float, optional): distance (in Angstrom) of the adsorbate
                from the surface (centre of mass if fix_com=True, position of index N
                of adsorbate otherwise). Defaults to 2.
            e_min (float, optional): minimum energy of the slab + adsorbate structure
                (obtained, e.g., from relaxation on different adsorption sites or from
                global optimization).
            spacing (float, optional): spacing of the 2d grid of positions of the
                adsorbates. Defaults to 0.50.
            spacing_surrogate (float, optional): spacing of the 2d grid on which
                the potential energy surface is evaluated. Used for integration
                (scipy_integral=False) and to produce the plots. Defaults to 0.10.
            cell (ase.cell.Cell, optional): reduced cell used to create the grid of
                positions for the constrained optimizations. Defaults to None.
            reduce_cell (bool, optional): reduce the cell accounting for
                translational symmetries (a 1x1 cell is obtained). Defaults to True.
                repetitions (list, optional): list of repetitions in x and y directions
                used to produce a NxN slab from a 1x1 slab. Defaults to None.
            border (float, optional): border surrounding the atoms cell in which
                the [x, y, z] points are extended to. Defaults to 3 Angstrom.
            n_rotations (int, optional): number of rotations sampled. Defaults to 1.
            fix_com (bool, optional): fix centre of mass of the adsorbate in the
                constrained optimizations instead of one atom. Defaults to False.
            index (int, optional): index of the adsorbate to fix in the
                constrained optimizations. Defaults to 0.
            fmax (float, optional): maximum forces for convergence of constrained
                relaxation. Defaults to 0.01.
            optimizer (ase.optimizer.Optimizer, optional): optimizer for constrained
                relaxation. Defaults to BFGS.
            kwargs_opt (dict, optional): dictionary of options for the optimizer.
                Defaults to {}.
            delta (float, optional): small length to calculate derivative of z_func
                with respect to x and y. Defaults to 0.05.
            scipy_integral (bool, optional): use scipy.integrate.dblquad to do the
                integration of the function of the potential energy surface for the
                calculation of the entropy. Defaults to False.
            trajectory (str, ase.io.Trajectory): trajectory name or object to store
                the final structures of the constrained optimizations.
            trajmode (str): mode for writing the trajectory file.
            name (str, optional): name of the simulation, a directory with this
                name is produce to store the cache. Defaults to "pes".
        """
        self.slab = slab.copy()
        self.ads = ads.copy()
        self.calc = calc
        self.height = height
        self.indices_surf = indices_surf
        self.distance = distance
        self.e_min = e_min
        self.spacing = spacing
        self.spacing_surrogate = spacing_surrogate
        self.reduce_cell = reduce_cell
        self.repetitions = repetitions
        self.border = border
        self.n_rotations = n_rotations
        self.fix_com = fix_com
        self.index = index
        self.fmax = fmax
        self.optimizer = optimizer
        self.kwargs_opt = kwargs_opt
        self.delta = delta
        self.scipy_integral = scipy_integral
        self.trajectory = trajectory
        self.trajmode = trajmode

        if fix_com is True:
            ads_pos = self.ads.get_center_of_mass()
        else:
            ads_pos = self.ads[index].position
        self.ads.translate(-ads_pos)

        self.cache = get_json_cache(name)

    @property
    def name(self):
        return str(self.cache.directory)

    def prepare(self):
        """Prepare Potential Energy Sampling method."""
    
        if len([cc for cc in self.slab.constraints if isinstance(cc, FixAtoms)]) == 0:
            raise Exception("Atoms must contain FixAtoms constraint.")

        # Reduced cell.
        self.cell = self.slab.cell
        if self.reduce_cell is True:
            self.cell = get_1x1_slab_cell(
                atoms=self.slab,
                repetitions=self.repetitions,
            )
            if self.repetitions is None:
                self.repetitions = (
                    int(self.slab.cell.lengths()[0] / self.cell.lengths()[0]),
                    int(self.slab.cell.lengths()[1] / self.cell.lengths()[1]),
                )

        # Calculate zz_fuction from indices_surf.
        if self.indices_surf is not None:
            xyz_points = self.slab[self.indices_surf].positions
            xyz_points_ext = extend_xyz_points(
                xyz_points=xyz_points,
                cell=self.cell,
                border=self.border,
            )
            self.z_func = lambda xx, yy: griddata(
                points=xyz_points_ext[:, :2],
                values=xyz_points_ext[:, 2],
                xi=[xx, yy],
                method="linear",
                rescale=False,
            )
            self.height = np.average([xyz[2] for xyz in xyz_points])
        else:
            self.z_func = None

        # Get grid of points from spacing.
        xyz_points = get_xyz_points(
            cell=self.cell,
            height=self.height,
            spacing=self.spacing,
            z_func=self.z_func,
        )
        self.xyz_points = xyz_points
        
        return xyz_points

    def run(self):
        """Run Potential Energy Sampling method."""
    
        # Prepare the grid.
        self.prepare()
    
        # Do constrained relaxations.
        xye_points = self.xyz_points.copy()
        for ii, position in enumerate(self.xyz_points):
            with self.cache.lock(f"{ii:04d}") as handle:
                if handle is None:
                    xye_points[ii] = self.cache[f"{ii:04d}"]
                    continue
                atoms = constrained_relaxations_with_rotations(
                    slab=self.slab,
                    ads=self.ads,
                    position=position,
                    calc=self.calc,
                    n_rotations=self.n_rotations,
                    distance=self.distance,
                    fix_com=self.fix_com,
                    index=self.index,
                    optimizer=self.optimizer,
                    kwargs_opt=self.kwargs_opt,
                    fmax=self.fmax,
                    z_func=self.z_func,
                    delta=self.delta,
                )
                xye_points[ii, 2] = atoms.get_potential_energy()
                if self.trajectory is not None:
                    if isinstance(self.trajectory, str):
                        self.trajectory = Trajectory(
                            filename=self.trajectory,
                            mode=self.trajmode,
                        )
                    atoms.constraints = []  # needed to read the trajectory.
                    self.trajectory.write(atoms)
                if world.rank == 0:
                    handle.save(xye_points[ii])

        self.xye_points = xye_points
        self.xye_points_ext = extend_xyz_points(
            xyz_points=xye_points,
            cell=self.cell,
            border=self.border,
        )
        self.es_grid = None

        return xye_points

    def read(self, trajectory):
        """Read energies from an ase trajectory."""
        from ase.io import read
        atoms_list = read(trajectory, ":")
        
        # Prepare the grid.
        self.prepare()
        
        assert len(atoms_list) == len(self.xyz_points)
        
        # Read the energies.
        xye_points = self.xyz_points.copy()
        for ii, position in enumerate(self.xyz_points):
            xye_points[ii, 2] = atoms_list[ii].get_potential_energy()
        
        self.xye_points = xye_points
        self.xye_points_ext = extend_xyz_points(
            xyz_points=xye_points,
            cell=self.cell,
            border=self.border,
        )
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

    def surrogate_pes(self, sklearn_model=None):
        """Train the surrogate model for the PES."""
        if sklearn_model is None:
            self.e_func = lambda xx, yy: griddata(
                points=self.xye_points_ext[:, :2],
                values=self.xye_points_ext[:, 2],
                xi=[xx, yy],
                method="cubic",
                rescale=False,
            )
        else:
            sklearn_model.fit(self.xye_points_ext[:, :2], self.xye_points_ext[:, 2])
            self.e_func = lambda xx, yy: sklearn_model.predict([[xx, yy]])[0]

    def get_meshgrid_surrogate(self):
        """Get mesh grid of PES from surrogate model."""
        self.xs_grid, self.ys_grid, self.es_grid = get_meshgrid(
            cell=self.cell,
            height=self.height,
            spacing=self.spacing_surrogate,
            z_func=self.e_func,
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
        ax = plt.figure().add_subplot(projection="3d")
        ax.plot_surface(self.xs_grid, self.ys_grid, self.es_grid)
        ax.scatter(
            self.xye_points[:, 0],
            self.xye_points[:, 1],
            self.xye_points[:, 2],
        )
        plt.show()

    def get_integral_pes_scipy(self, temperature):
        """Integrate the function of the PES with scipy."""
        from scipy.integrate import dblquad

        if np.isclose(self.cell[0], [1, 0, 0]) is False:
            raise Exception(
                "scipy_integral=True works only if cell[0] is equal to [1,0,0]."
            )
        func = lambda yy, xx: (
            np.exp(-(self.e_func(yy, xx) - self.e_min) / (units.kB * temperature))
        )
        integral_cpes = dblquad(
            func=func,
            a=0.0,
            b=self.cell[1, 1],
            gfun=lambda xx: self.cell[1, 0] / self.cell[1, 1] * xx,
            hfun=lambda xx: self.cell[1, 0] / self.cell[1, 1] * xx + self.cell[0, 0],
        )[0]

        return integral_cpes

    def get_integral_pes_grid(self, temperature):
        """Integrate the function of the PES with finite differences."""
        if self.es_grid is None:
            self.get_meshgrid_surrogate()
        vv_grid = np.exp(-(self.es_grid - self.e_min) / (units.kB * temperature))
        integral_cpes = np.average(vv_grid) * self.cell[0, 0] * self.cell[1, 1]

        return integral_cpes

    def get_entropy_pes(self, temperature):
        """Calculate the entropy from the PES."""
        if self.es_grid is None:
            self.get_meshgrid_surrogate()
            if self.e_min is None or self.e_min > np.min(self.es_grid):
                self.e_min = np.min(self.es_grid)
        if self.scipy_integral is True:
            integral_cpes = self.get_integral_pes_scipy(temperature)
        else:
            integral_cpes = self.get_integral_pes_grid(temperature)

        hP = units._hplanck * units.kJ * 1e-3
        mass = sum(self.ads.get_masses()) / units.kg
        part_fun = 2 * np.pi * mass * units.kB * temperature * integral_cpes / (hP**2)
        entropy = units.kB * np.log(part_fun)

        return entropy

    def get_ads_positions(self):
        """Return an atoms structure with the grid of positions (added X atoms)."""
        from ase import Atoms

        slab_new = self.slab.copy()
        for position in self.xyz_points:
            ads_new = Atoms("X")
            ads_new.translate(position)
            slab_new += ads_new
        return slab_new


# -------------------------------------------------------------------------------------
# PES THERMO
# -------------------------------------------------------------------------------------


class PESThermo(HarmonicThermo):
    def __init__(
        self,
        pes,
        vib_energies,
        potentialenergy=0.0,
    ):
        """Class for calculating thermodynamic properties in the approximation
        that all degrees of freedom are treated harmonically except for two, which
        are evaluated with the potential energy sampling method. Used for adsorbates
        which can translate on the slab.

        Args:
            pes (PotentialEnergySampling): _description_
            vib_energies (list): list of vibrational energies (in eV), obtained
            with, e.g., ase.vibrations.Vibrations class.
            potentialenergy (float, optional): the potential energy (in eV),
            obtained, e.g., from atoms.get_potential_energy. Defaults to 0.
        """
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
        fmt = "%-15s%13.7f eV/K%13.3f eV"
        write("Entropy components at T = %.2f K:" % temperature)
        write("=" * 49)
        write("%15s%13s     %13s" % ("", "S", "T*S"))

        S = 0.0

        S_v = self._vibrational_entropy_contribution(temperature)
        write(fmt % ("S_harm", S_v, S_v * temperature))
        S += S_v

        S_p = self.pes.get_entropy_pes(temperature)
        write(fmt % ("S_pes", S_p, S_p * temperature))
        S += S_p

        write("-" * 49)
        write(fmt % ("S", S, S * temperature))
        write("=" * 49)
        return S


# -------------------------------------------------------------------------------------
# FIXSUBSETCOM
# -------------------------------------------------------------------------------------


class FixSubsetCom(FixConstraint):
    """Constraint class for fixing the center of mass of a subset of atoms."""

    def __init__(self, indices, mask=(True, True, True)):
        self.index = np.asarray(indices, int)
        self.mask = np.asarray(mask, bool)

    def get_removed_dof(self, atoms):
        return self.mask.sum()

    def adjust_positions(self, atoms, new):
        masses = atoms.get_masses()[self.index]
        old_cm = atoms[self.index].get_center_of_mass()
        new_cm = masses @ new[self.index] / masses.sum()
        diff = old_cm - new_cm
        diff *= self.mask
        new += diff

    def adjust_momenta(self, atoms, momenta):
        """Adjust momenta so that the center-of-mass velocity is zero."""
        masses = atoms.get_masses()[self.index]
        velocity_com = momenta[self.index].sum(axis=0) / masses.sum()
        velocity_com *= self.mask
        momenta[self.index] -= masses[:, None] * velocity_com

    def adjust_forces(self, atoms, forces):
        masses = atoms.get_masses()[self.index]
        lmd = masses @ forces[self.index] / sum(masses**2)
        lmd *= self.mask
        forces[self.index] -= masses[:, None] * lmd

    def todict(self):
        return {
            "name": self.__class__.__name__,
            "kwargs": {"indices": self.index.tolist(), "mask": self.mask.tolist()},
        }

# -------------------------------------------------------------------------------------
# PLOT ENTROPIES
# -------------------------------------------------------------------------------------

def plot_entropies(
    thermo_dict,
    temperature_range=[200, 1000],
    step=10,
    filename="entropies.png",
):
    
    import matplotlib.pyplot as plt
    entropies_dict = {name: [] for name in thermo_dict}
    temperature_list = range(temperature_range[0], temperature_range[1]+step, step)
    for temperature in temperature_list:
        for name in thermo_dict:
            entropy = thermo_dict[name].get_entropy(
                temperature=temperature,
                verbose=False,
            )
            entropies_dict[name].append(entropy*1000) # [meV/K]
    
    plt.figure()
    for name in thermo_dict:
        plt.plot(temperature_list, entropies_dict[name], label=name)
    plt.legend(loc='upper left')
    plt.xlim(temperature_range)
    #plt.ylim([0.2, 1.2])
    plt.xlabel("temperature [K]")
    plt.ylabel("entropy [meV/K]")
    plt.savefig(filename)

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
