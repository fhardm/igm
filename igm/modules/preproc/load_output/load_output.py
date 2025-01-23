#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

# Author: Florian Hardmeier, florian.hardmeier@geo.uzh.ch

import numpy as np
import os
import datetime, time
import tensorflow as tf
from netCDF4 import Dataset
from scipy.interpolate import RectBivariateSpline

from igm.modules.utils import *


def params(parser):
    parser.add_argument(
        "--lncd_input_file",
        type=str,
        default="input.nc",
        help="NetCDF input data file",
    )
    parser.add_argument(
        "--lncd_toggle_particles",
        type=bool,
        default=False,
        help="Toggle loading spin-up particles from particle csv file",
    )
    parser.add_argument(
        "--lncd_particles_file",
        type=str,
        default="particles_output.csv",
        help="Load particles from particle csv file",
    )
    
    
    
def initialize(params, state):
    if hasattr(state, "logger"):
        state.logger.info("LOAD NCDF file")

    nc = Dataset(params.lncd_input_file, "r")

    x = np.squeeze(nc.variables["x"]).astype("float32")
    y = np.squeeze(nc.variables["y"]).astype("float32")
    
    # load any field contained in the ncdf file at the final time value saved, replace missing entries by nan
    for var in nc.variables:
        if not var in ["x", "y", "z", "time"]:
            vars()[var] = np.squeeze(nc.variables[var][-1]).astype("float32")
            vars()[var] = np.where(vars()[var] > 10**35, np.nan, vars()[var])

    print("Loaded fields: ", list(nc.variables.keys()))
    # transform from numpy to tensorflow
    for var in nc.variables:
        if not var in ["z", "time"]:
            if var in ["x", "y"]:
                vars(state)[var] = tf.constant(vars()[var].astype("float32"))
            else:
                vars(state)[var] = tf.Variable(vars()[var].astype("float32"), trainable=False)

    nc.close()

    complete_data(state)
    
    if params.lncd_toggle_particles:
        # Load particles from the CSV file
        particles = np.genfromtxt(params.lncd_particles_file, delimiter=',', names=True)

        # Extract state.particle variables from the CSV file
        state.particle_x = tf.constant(particles['x'].astype("float32")) - state.x[0]
        state.particle_y = tf.constant(particles['y'].astype("float32")) - state.y[0]
        state.particle_z = tf.constant(particles['z'].astype("float32"))
        state.particle_r = tf.constant(particles['r'].astype("float32"))
        state.particle_t = tf.constant(particles['t'].astype("float32"))
        state.particle_englt = tf.constant(particles['englt'].astype("float32"))
        state.particle_topg = tf.constant(particles['topg'].astype("float32"))
        state.particle_thk = tf.constant(particles['thk'].astype("float32"))
        state.particle_w = tf.constant(particles['w'].astype("float32"))
        state.particle_srcid = tf.constant(particles['srcid'].astype("float32"))


def update(params, state):
    pass


def finalize(params, state):
    pass
