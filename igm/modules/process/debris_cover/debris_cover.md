### <h1 align="center" id="title">IGM module `debris_cover` </h1>

# Description
This IGM module aims to represent the dynamics of a debris-covered glacier. It uses Eulerian particle tracking (adapted from the module `particles`) to simulate englacial and supraglacial debris transport, evaluating debris thickness and its feedback with surface mass balance.

The module provides the user with several options of where and how much debris should be seeded.

## Seeding

The module currently supports three options to define the area where particles should be seeded through the parameter `--part_seeding_type`:

- For `'conditions'`, the seeding area can be tied to some quantity. Currently, only a surface slope is in the module. The surface gradient is computed from the input topography, then the minimum slope threshold `--part_seed_slope` is applied, resulting in a binary mask.
- For `'shapefile'`, the user can prepare a `.shp` file containing polygons (e.g. known rockfall source areas), which is then converted to a binary mask.
- For `'both'`, the two previously explained methods are combined.

Next, the parameter `--part_density_seeding`, defined by the user in an array (to enable variation over time), correspons to a debris volume per particle (dependent on seeding frequency `--part_frequency_seeding` and grid size). This volume is assigned to each particle as a permanent property `particle_w` when it is seeded.

## Particle tracking

As explained in the `particles` module. The default tracking method is `'3d'`.
The boolean `--part_remove_immobile_particles` gives the user the option to remove immobile off-glacier particles to reduce computation time.


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
||`--part_debrismask_shapefile`|`'debrismask.shp'`|Debris mask input file (shapefile)|
||`--part_frequency_seeding`|`10`|Debris input frequency in years (default: 10), should not go below `--time_save`|
||`--part_density_seeding`|``|Debris input rate (or seeding density) in mm/yr in a given seeding area, user-defined as a list with d_in values by year|
||`--part_seed_slope`|`45`|Minimum slope to seed particles (in degrees) for `--part_seeding_type = 'conditions'`|
||`--part_tracking_method`|`'3d'`|Method for tracking particles (simple or 3d)|
||`--part_remove_immobile_particles`|`False`|Option to remove immobile off-glacier particles|
||`--smb_oestrem_D0`|`0.065`|Characteristic debris thickness (m) in Oestrem curve calculation|
