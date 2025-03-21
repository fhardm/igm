### <h1 align="center" id="title">IGM module `debris_cover` </h1>

# Description
This IGM module aims to represent the dynamics of a debris-covered glacier. It uses Eulerian particle tracking (adapted from the module `particles`) to simulate englacial and supraglacial debris transport, evaluating debris thickness and its feedback with surface mass balance.

The module provides the user with several options of where and how much debris should be seeded.

## Seeding

The module currently supports four options to define the area where particles should be seeded through the parameter `--part_seeding_type`:

- For `'conditions'`, the seeding area can be tied to some quantity. Currently, only a surface slope condition is implemented in the module. The surface gradient is computed from the input topography, then a minimum slope threshold `--part_seed_slope` is applied, resulting in a binary mask.
- For `'shapefile'`, the user can prepare a `.shp` file containing polygons (e.g. known rockfall source areas), which is then converted to a binary mask.
- For `'both'`, the two previously explained methods are combined.
- For `'slope_highres'`, the user can prepare a high-resolution boolean TIFF containing areas above a slope theshold, extracted from a high-resolution DEM (e.g. swissALTI3D 2m for a Swiss glacier) in manual pre-processing. The module then scales assigned debris volume per particle based on the steep area fraction within each seeding pixel.

Next, the parameter `--part_density_seeding`, defined by the user in an array (to enable variation over time), represents a debris input rate in mm per year per square meter. In the model, it corresponds to a debris volume per particle (dependent on seeding frequency `--part_frequency_seeding` and grid size). This volume is assigned to each particle as a permanent property `particle_w` when it is seeded.

The option `--part_slope_correction`, if set to `True`, corrects for the disparity between true surface area and flat pixel area for high slopes. This is done by directly scaling asssigned debris volume `particle_w`.

The option `--part_initial_rockfall`, if set to `True`, relocates seeding locations to a lower slope (lower than `--part_seed_slope`), where a rockfall would deposit on the glacier more realistically. The particles are iteratively moved in aspect direction until they reach a position below slope threshold. This is repeated at every seeding timestep to account for changes in the glaciated surface. The parameter `--part_max_runout` defines a maximum additional distance the particle will travel after reaching a slope < `--part_seed_slope`. Particles will be uniformly (randomly) distributed between 1 and 1 + `--part_max_runout` times the initial rockfall distance.

## Particle tracking and off-glacier particle options

Adapted from the `particles` module. The default tracking method is `'3d'`.
The boolean `--part_aggregate_immobile_particles` gives the user the option to aggregate immobile off-glacier particles into a single particle per pixel to reduce computation time while conserving assigned debris volumes.

The boolean `--part_moraine_builder` (given `--part_remove_immobile_particles` is `False`) gives the user the option to evaluate off-glacier particles to accumulate moraines as a thickness `state.debthick_offglacier`, based on debris volume within a pixel. This thickness is then added to the bed topography `state.topg`.


## Debris cover and SMB feedback

When a particle has a relative position within the ice column `particle_r` of 1, it is detected as surface debris. The assigned debris volumes `particle_w` of all particles within a pixel are summed up and distributed across the pixel as a debris thickness `debthick`.

Similarly, depth-averaged englacial debris concentration is saved to the variable `debcon`. Vertically resolved debris concentration is saved to `debcon_vert`. The amount of vertical layers is given by the vertical ice flow layers `iflo_Nz`.

Debris flux - defined as the volume of debris moving along the glacier per meter per year - is saved to the variable `debflux`.

Surface mass balance (SMB) is then adjusted according to debris thickness `debthick`. Currently, the module uses a simple Oestrem curve approach, where

$$a = \tilde{a}\frac{D_0}{D_0 + D},$$

where $a$ is the debris-covered mass balance, $\tilde{a}$ is the debris-free mass balance (from the SMB module; default: `smb_simple`), $D_0$ is the user-defined characteristic debris thickness `--smb_oestrem_D0`, and $D$ is the local debris thickness `debthick`.


## Trackable particle properties

Any property can be assigned to particles when seeded, tracked, and/or evaluated during a particle's lifetime. In the current version, these properties are defined in the module and are saved to `traj-xxxxxx.csv` files by the module `write_debris`. These include:

|name|description|
| :--- | :--- |
|`ID`|Unique particle identifier|
|`particle_x`|x coordinate of particle position (in coord. system, e.g. UTM32)|
|`particle_y`|y coordinate of particle position (in coord. system, e.g. UTM32)|
|`particle_z`|z coordinate of particle position (in m a.s.l.)|
|`particle_r`|Relative vertical position within the ice column (0 = bed, 1 = ice surface)|
|`particle_t`|Particle seeding timestamp|
|`particle_englt`|Total time the particle spent within the ice|
|`particle_topg`|Bed elevation at the particle's position|
|`particle_thk`|Ice thickness at the particle's position|
|`particle_w`|Assigned representative debris volume (m3)|
|`particle_srcid`|Source area identifier (shapefile FID when using shapefiles to seed)|


# Parameters

|short|long|default|help|
| :--- | :--- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
||`--working_dir`|``|Working directory (default empty string)|
||`--modules_preproc`|`['oggm_shop']`|List of pre-processing modules|
||`--modules_process`|`['iceflow', 'time', 'thk']`|List of processing modules|
||`--modules_postproc`|`['write_ncdf', 'plot2d', 'print_info']`|List of post-processing modules|
||`--logging`||Activate the looging|
||`--logging_file`|``|Logging file name, if empty it prints in the screen|
||`--print_params`||Print definitive parameters in a file for record|
||`--part_seeding_delay`|`0`|Optional delay in years before seeding starts at the beginning of the simulation|
||`--part_seeding_type`|`'conditions'`|Seeding type (`'conditions'`, `'shapefile'`, or `'both'`). `'conditions'` seeds particles based on conditions (e.g. slope, thickness, velocity), `'shapefile'` seeds particles in area defined by a shapefile, `'both'` applies conditions and shapefile|
||`--part_debrismask_shapefile`|`'debrismask.shp'`|Debris mask input file (shapefile)|
||`--part_frequency_seeding`|`10`|Debris input frequency in years (default: 10), should not go below `--time_save`|
||`--part_density_seeding`|``|Debris input rate (or seeding density) in mm/yr in a given seeding area, user-defined as a list with d_in values by year|
||`--part_seed_slope`|`45`|Minimum slope to seed particles (in degrees) for `--part_seeding_type = 'conditions'`|
||`--part_slope_correction`|`False`|Option to correct seeding debris volume for increased surface area at high slopes|
||`--part_initial_rockfall`|`False`|Option for iteratively relocating seeding locations to below-threshold slope|
||`--part_max_runout`|`0.5`|Maximum runout factor for particles after initial rockfall as a fraction of the previous rockfall distance. Particles will be uniformly distributed between 0 and this value|
||`--part_tracking_method`|`'3d'`|Method for tracking particles (simple or 3d)|
||`--part_aggregate_immobile_particles`|`False`|Option to aggregate immobile off-glacier particles to reduce memory use & computation time|
||`--part_moraine_builder`|`False`|Build a moraine using off-glacier immobile particles|
||`--smb_oestrem_D0`|`0.065`|Characteristic debris thickness (m) in Oestrem curve calculation|

Code written by F. Hardmeier. Partially adapted from the particles module, which was originally written by G. Jouvet and C.-M. Stucki.