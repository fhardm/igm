### <h1 align="center" id="title">IGM module `load_output` </h1>

# Description:

This IGM module loads data from an IGM output file (parameter `--lncd_input_file`, default: output.nc) and transform all existing 2D fields into tensorflow variables.

The module finds the output variables at the last time index (the final state of the spin-up run) and uses them as input.

The option `--lncd_toggle_particles` enables importing the final state of spin-up particles from a separate CSV file with the file name `--lncd_particles_file`.

This requires having saved the final state of particles in the spin-up run either by enabling the option `--wpar_toggle_particles` or manually renaming the final temporary CSV particle file (`traj-TIME.csv`).

This module depends on `netCDF4`.
Code written by F. Hardmeier for the debris_cover module.