### <h1 align="center" id="title">IGM module `plot_debris` </h1>

# Description:

This IGM module produces 2D plan-view plots of variable defined by parameter `pltdeb_var` (e.g. `pltdeb_var` can be set to `thk`, or `ubar`, ...). The saving frequency is given by parameter `time_save` defined in module `time`.  The scale range of the colobar is controlled by parameter `pltdeb_varmax`.

By default, the plots are saved as png files in the working directory. However, one may display the plot "in live" by setting `pltdeb_live` to True. Note that if you use the spyder python editor, you need to turn `pltdeb_editor` to 'sp'.
 
If the `debris_cover` module is activated, one may plot particles on the top setting `pltdeb_particles` to True, or remove them form the plot seeting it to False.

Code written by F. Hardmeier for the debris_cover module.