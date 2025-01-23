#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import os
import time
import tensorflow as tf
import shutil
from igm.modules.utils import *


def params(parser):
    parser.add_argument(
        "--wpar_add_topography",
        type=str2bool,
        default=True,
        help="Add topg",
    )
    parser.add_argument(
        "--wpar_toggle_particles",
        type=str2bool,
        default=False,
        help="Write the final particle properties to a separate CSV file",
    )
    parser.add_argument(
        "--wpar_vars_to_save",
        nargs='+',
        default=[
            "particle_x",
            "particle_y",
            "particle_z",
            "particle_r",
            "particle_t",
            "particle_englt",
            "particle_topg",
            "particle_thk",
            "particle_w",
            "particle_srcid"
        ],
        help="List of variables to be recorded in the ncdf file",
    )
def initialize(params, state):
    state.tcomp_write_particles = []

    directory = "trajectories"
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.mkdir(directory)

    os.system( "echo rm -r " + "trajectories" + " >> clean.sh" )

    if params.wpar_add_topography:
        ftt = os.path.join("trajectories", "topg.csv")
        array = tf.transpose(
            tf.stack(
                [state.X[state.X > 0], state.Y[state.X > 0], state.topg[state.X > 0]]
            )
        )
        np.savetxt(ftt, array, delimiter=",", fmt="%.2f", header="x,y,z")


def update(params, state):
    if state.saveresult:
        state.tcomp_write_particles.append(time.time())

        f = os.path.join(
            "trajectories",
            "traj-" + "{:06d}".format(int(state.t.numpy())) + ".csv",
        )

        ID = tf.cast(tf.range(state.particle_x.shape[0]), dtype="float32")
        vars_to_stack = [ID]
        for var in params.wpar_vars_to_save:
            if var == "particle_x":
                vars_to_stack.append(
                    getattr(state, var).numpy().astype(np.float64) + state.x[0].numpy().astype(np.float64),
                )
            elif var == "particle_y":
                vars_to_stack.append(
                    getattr(state, var).numpy().astype(np.float64) + state.y[0].numpy().astype(np.float64),
                )
            else:
                vars_to_stack.append(getattr(state, var))

        array = tf.transpose(tf.stack(vars_to_stack, axis=0))
        np.savetxt(f, array, delimiter=",", fmt="%.2f", header="ID," + ",".join(params.wpar_vars_to_save))

        ft = os.path.join("trajectories", "time.dat")
        with open(ft, "a") as f:
            print(state.t.numpy(), file=f)

        if params.wpar_add_topography:
            ftt = os.path.join(
                "trajectories",
                "usurf-" + "{:06d}".format(int(state.t.numpy())) + ".csv",
            )
            array = tf.transpose(
                tf.stack(
                    [
                        state.X[state.X > 1],
                        state.Y[state.X > 1],
                        state.usurf[state.X > 1],
                    ]
                )
            )
            np.savetxt(ftt, array, delimiter=",", fmt="%.2f", header="x,y,z")

        state.tcomp_write_particles[-1] -= time.time()
        state.tcomp_write_particles[-1] *= -1


def finalize(params, state):
    if params.wpar_toggle_particles:
        # Save particle properties to a CSV file (that will not be deleted when the next run starts)
        filename = "particles_" + params.wncd_output_file.replace(".nc", ".csv")
        ID = tf.cast(tf.range(state.particle_x.shape[0]), dtype="float32")
        vars_to_stack = [ID]
        for var in params.wpar_vars_to_save:
            if var in ["particle_x", "particle_y"]:
                vars_to_stack.append(
                    getattr(state, var).numpy().astype(np.float64) + getattr(state, var[0]).numpy().astype(np.float64)
                )
            else:
                vars_to_stack.append(getattr(state, var))

        array = tf.transpose(tf.stack(vars_to_stack, axis=0))
        np.savetxt(filename, array, delimiter=",", fmt="%.2f", header="ID," + ",".join(params.wpar_vars_to_save))
