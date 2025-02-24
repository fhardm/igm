### <h1 align="center" id="title">IGM module `debris_cover` </h1>

# Description
This IGM module aims to represent the dynamics of a debris-covered glacier. It uses Eulerian particle tracking (adapted from the module `particles`) to simulate englacial and supraglacial debris transport, evaluating debris thickness and its feedback with surface mass balance.

The module provides the user with several options of where and how much debris should be seeded.

## Seeding

The module currently supports four options to define the area where particles should be seeded through the parameter `--part_seeding_type`:

- For `'conditions'`, the seeding area can be tied to some quantity. Currently, only a surface slope is in the module. The surface gradient is computed from the input topography, then the minimum slope threshold `--part_seed_slope` is applied, resulting in a binary mask.
- For `'shapefile'`, the user can prepare a `.shp` file containing polygons (e.g. known rockfall source areas), which is then converted to a binary mask.
- For `'both'`, the two previously explained methods are combined.
- For `'slope_shp'`, the user can prepare a shapefile containing areas above a slope theshold, extracted from a high-resolution DEM (e.g. swissALTI3D 2m for a Swiss glacier). The module then calculates a seeding probability based on the steep area fraction within each pixel.

Next, the parameter `--part_density_seeding`, defined by the user in an array (to enable variation over time), correspons to a debris volume per particle (dependent on seeding frequency `--part_frequency_seeding` and grid size). This volume is assigned to each particle as a permanent property `particle_w` when it is seeded.

The option `--part_initial_rockfall`, if set to `True`, relocates seeding locations to a lower slope (lower than `--part_seed_slope`), where a rockfall would deposit on the glacier more realistically. The particles are iteratively moved in aspect direction until they reach a position below slope threshold. This is repeated at every seeding timestep to account for changes in the glaciated surface. The parameter `--part_max_runout` defines a maximum additional distance the particle will travel after reaching a slope < `--part_seed_slope`. Particles will be uniformly (randomly) distributed between 1 and 1 + `--part_max_runout` times the initial rockfall distance.

## Particle tracking and off-glacier particle options

Adapted from the `particles` module. The default tracking method is `'3d'`.
The boolean `--part_remove_immobile_particles` gives the user the option to remove immobile off-glacier particles to reduce computation time.

The boolean `--part_moraine_builder` (given `--part_remove_immobile_particles` is `False`) gives the user the option to evaluate off-glacier particles to accumulate moraines as a thickness `state.debthick_offglacier`, based on debris volume within a pixel. This thickness is then added to the bed topography `state.topg`.


## Debris cover and SMB feedback

When a particle has a relative position within the ice column `particle_r` of 1, it is counted as surface debris. The assigned debris volumes `particle_w` of all particles within a pixel are summed up and distributed across the pixel as a debris thickness `debthick`.

Surface mass balance (SMB) is then adjusted according to debris thickness. Currently, the module uses a simple Oestrem curve approach, where

$$a = \tilde{a}\frac{D_0}{D_0 + D},$$

where $a$ is the debris-covered mass balance, $\tilde{a}$ is the debris-free mass balance (from the SMB module; default: `smb_simple`), $D_0$ is the user-defined characteristic debris thickness `--smb_oestrem_D0`, and $D$ is the local debris thickness `debthick`.

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
||`--part_seeding_type`|`'conditions'`|Seeding type (`'conditions'`, `'shapefile'`, or `'both'`). `'conditions'` seeds particles based on conditions (e.g. slope, thickness, velocity), `'shapefile'` seeds particles in area defined by a shapefile, `'both'` applies conditions and shapefile|
||`--part_initial_rockfall`|`False`|Option for relocating seeding locations to below-threshold slope|
||`--part_debrismask_shapefile`|`'debrismask.shp'`|Debris mask input file (shapefile)|
||`--part_frequency_seeding`|`10`|Debris input frequency in years (default: 10), should not go below `--time_save`|
||`--part_density_seeding`|``|Debris input rate (or seeding density) in mm/yr in a given seeding area, user-defined as a list with d_in values by year|
||`--part_seed_slope`|`45`|Minimum slope to seed particles (in degrees) for `--part_seeding_type = 'conditions'`|
||`--part_tracking_method`|`'3d'`|Method for tracking particles (simple or 3d)|
||`--part_remove_immobile_particles`|`False`|Option to remove immobile off-glacier particles|
||`--smb_oestrem_D0`|`0.065`|Characteristic debris thickness (m) in Oestrem curve calculation|

Code written by F. Hardmeier. Partially adapted from the particles module, which was originally written by G. Jouvet, improved and tested by C.-M. Stucki.