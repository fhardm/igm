#!/usr/bin/env python3

# Author: Florian Hardmeier, florian.hardmeier@geo.uzh.ch
# First created: 01.11.2024

# Combines particle tracking and mass balance computation for debris-covered glaciers. Seeded particles represent a volume of debris
# and are tracked through the glacier. After reaching the glacier surface in the ablation area, 
# their number in each grid cell is used to compute debris thickness by distributing the particle debris volume over the grid cell. 
# Mass balance is adjusted based on debris thickness using a simple Oestrem curve.

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import igm
import datetime, time

import geopandas as gpd
from shapely.geometry import Point

from igm.modules.utils import *
import pandas as pd
import rasterio

def params(parser):
    # Particle seeding
    parser.add_argument(
        "--part_seeding_type",
        type=str,
        default="conditions",
        help="Seeding type (conditions, shapefile, or both). 'conditions' seeds particles based on conditions (e.g. slope, thickness, velocity), 'shapefile' seeds particles in area defined by a shapefile, 'both' applies conditions and shapefile (default: conditions)",
    )
    parser.add_argument(
        "--part_seeding_area_file",
        type=str,
        default="debrismask.shp",
        help="Debris mask input file (default: debrismask.shp)",
    )
    parser.add_argument(
        "--part_seeding_delay",
        type=int,
        default=0,
        help="Optional delay in years before seeding starts at the beginning of the simulation (default: 0 years)",
    )
    parser.add_argument(
        "--part_frequency_seeding",
        type=int,
        default=10,
        help="Debris input frequency in years (default: 10), should not go below time_save (default: 10 years)",
    )
    parser.add_argument(
        "--part_density_seeding",
        type=list,
        default=[],
        help="Debris input rate (or seeding density) in mm/yr in a given seeding area, user-defined as a list with d_in values by year.",
    )
    parser.add_argument(
        "--part_seed_slope",
        type=int,
        default=45,
        help="Minimum slope to seed particles (in degrees) for part_seeding_type = 'conditions'",
    )
    parser.add_argument(
        "--part_initial_rockfall",
        type=int,
        default=False,
        help="Moves particles right after seeding to a slope lower than part_seed_slope (default: False)",
    )
    parser.add_argument(
        "--part_max_runout",
        type=int,
        default=0.5,
        help="Maximum runout factor for particles after initial rockfall (default: 0.5), as a fraction of the previous rockfall distance. Particles will be uniformly distributed between 0 and this value.",
    )
    parser.add_argument(
        "--part_slope_correction",
        type=bool,
        default=False,
        help="Corrects for increased exposed rockwall area from slope in seeding the seeding area by scaling assigned volume (default: False)",
    )
        
    # Particle tracking
    parser.add_argument(
        "--part_tracking_method",
        type=str,
        default="3d",
        help="Method for tracking particles (simple or 3d)",
    )
    parser.add_argument(
        "--part_aggregate_immobile_particles",
        type=bool,
        default=False,
        help="Remove immobile particles (default: False)",
    )
    parser.add_argument(
        "--part_moraine_builder",
        type=bool,
        default=False,
        help="Build a moraine using off-glacier immobile particles (default: False)",
    )
    
    # SMB
    parser.add_argument(
        "--smb_oestrem_D0",
        type=int,
        default=0.065,
        help="Characteristic debris thickness in Oestrem curve calculation (default: 0.065)",
    )


def initialize(params, state):
    # initialize the seeding
    state = initialize_seeding(params, state)
    
def update(params, state):
    if state.t.numpy() >= params.time_start + params.part_seeding_delay:
        # update the particle tracking by calling the particles function, adapted from module particles.py
        state = deb_particles(params, state)
        # update debris thickness based on particle count in grid cells (at every SMB update time step)
        state = deb_thickness(params, state)
        # update the mass balance (SMB) depending by debris thickness, using clean-ice SMB from smb_simple.py
        state = deb_smb(params, state)

def finalize(params, state):
    pass



def initialize_seeding(params, state):
    # initialize the debris variables
    state.particle_attributes = ["ID", "particle_x", "particle_y", "particle_z", "particle_r", "particle_w",
                 "particle_t", "particle_englt", "particle_thk", "particle_topg", "particle_srcid"]
    state.debthick = tf.Variable(tf.zeros_like(state.usurf, dtype=tf.float32))
    state.debthick_offglacier = tf.Variable(tf.zeros_like(state.usurf, dtype=tf.float32))
    state.debcon = tf.Variable(tf.zeros_like(state.usurf, dtype=tf.float32))
    state.debcon_vert = tf.Variable(tf.zeros((params.iflo_Nz,) + state.usurf.shape, dtype=tf.float32))
    state.debflux_engl = tf.Variable(tf.zeros_like(state.usurf, dtype=tf.float32))
    state.debflux_supragl = tf.Variable(tf.zeros_like(state.usurf, dtype=tf.float32))
    state.debflux = tf.Variable(tf.zeros_like(state.usurf, dtype=tf.float32))
    state.thk_deb = tf.Variable(tf.zeros_like(state.usurf, dtype=tf.float32))
    state.seeded_particles = tf.Variable([0], dtype=tf.float32)
    state.seeded_debris_volume = tf.Variable([0], dtype=tf.float32)
    state.tlast_seeding = -1.0e5000
    state.tcomp_particles = []
    
    # initialize trajectories (if they do not exist already)
    if not hasattr(state, 'particle_x'):
        for attr in state.particle_attributes:
            setattr(state, attr, tf.Variable([]))
        state.srcid = tf.zeros_like(state.thk, dtype=tf.float32)
    
    state.pswvelbase = tf.Variable(tf.zeros_like(state.thk),trainable=False)
    state.pswvelsurf = tf.Variable(tf.zeros_like(state.thk),trainable=False)
        
    dzdx , dzdy = compute_gradient_tf(state.usurf,state.dx,state.dx)
    state.slope_rad = tf.atan(tf.sqrt(dzdx**2 + dzdy**2))
    # initialize d_in array
    if params.part_density_seeding != []:
        state.d_in_array = np.array(params.part_density_seeding[1:]).astype(np.float32)
    
    # Grid seeding based on conditions, written by Andreas H., adapted by Florian H.
    if params.part_seeding_type == "conditions":
        state.gridseed = np.ones_like(state.thk, dtype=bool)

        # seed where gridseed is True and the slope is steep
        state.gridseed = state.gridseed & np.array(state.slope_rad > params.part_seed_slope/180*np.pi)
        
    # Seeding based on shapefile, adapted from include_icemask (Andreas Henz)  
    elif params.part_seeding_type == "shapefile":
        # read_shapefile
        gdf = read_shapefile(params.part_seeding_area_file)

        # Create a mask and source ID grid from shapefiles
        mask_values, srcid_values = compute_mask_and_srcid(state, gdf)
        
        # Define debrismask and srcid
        state.gridseed = tf.constant(mask_values, dtype=tf.bool)
        state.srcid = tf.constant(srcid_values, dtype=tf.float32)
        
        # If gridseed is empty, raise an error
        if not np.any(state.gridseed):
            raise ValueError("Shapefile not within icemask! Watch out for coordinate system!")
        
        
    elif params.part_seeding_type == "both":
        # read_shapefile
        gdf = read_shapefile(params.part_seeding_area_file)

        # Create a mask and source ID grid from shapefiles
        mask_values, srcid_values = compute_mask_and_srcid(state, gdf)

        # define debrismask and srcid
        state.gridseed = tf.constant(mask_values, dtype=tf.bool)
        state.srcid = tf.constant(srcid_values, dtype=tf.float32)
        
        # if gridseed is empty, raise an error
        if not np.any(state.gridseed):
            raise ValueError("Shapefile not within icemask! Watch out for coordinate system!")

        # initialize d_in array
        if params.part_density_seeding != []:
            state.d_in_array = np.array(params.part_density_seeding[1:]).astype(np.float32)
        
        # apply the gradient condition on the shapefile mask
        state.gridseed = state.gridseed & np.array(state.slope_rad > params.part_seed_slope / 180 * np.pi)
        
    elif params.part_seeding_type == "slope_highres":
        # Read the tif file
        with rasterio.open(params.part_seeding_area_file) as src:
            print("Data type of src:", type(src))
            # Read the debris mask and reproject it to match the grid of X, Y
            debris_mask = src.read(1, out_shape=(state.X.shape[0], state.X.shape[1]), resampling=rasterio.enums.Resampling.average)
        
        # Clip the debris mask to 0 and 1 (NaN values are set to 0)
        debris_mask = np.clip(debris_mask, 0, 1)
        # Flip debris_mask along x and y direction
        debris_mask = np.flipud(debris_mask)
        
        # Convert the debris mask to a TensorFlow constant
        state.gridseed_fraction = tf.constant(debris_mask, dtype=tf.float32)
        state.gridseed = tf.cast(state.gridseed_fraction > 0, dtype=tf.bool)
    
    elif params.part_seeding_type == "csv_points":
        # Read seeding points from the CSV file
        x, y = read_seeding_points_from_csv(params.part_seeding_area_file)

        # Assign the imported x and y coordinates to nparticle_x and nparticle_y
        state.seeding_x = x
        state.seeding_y = y
    
    return state


# Particle tracking, adapted from particles.py (Guillaume Jouvet)
def deb_particles(params, state):
    if hasattr(state, "logger"):
        state.logger.info("Update particle tracking at time : " + str(state.t.numpy()))
        
    if (state.t.numpy() - state.tlast_seeding) >= params.part_frequency_seeding:
        deb_seeding_particles(params, state)
        
        # merge the new seeding points with the former ones
        for attr in state.particle_attributes:
            setattr(state, attr, tf.concat([getattr(state, attr), getattr(state, f"n{attr}")], axis=-1))

        state.tlast_seeding = state.t.numpy()

    if (state.particle_x.shape[0]>0)&(state.it >= 0):
        state.tcomp_particles.append(time.time())
        
        # find the indices of trajectories
        # these indicies are real values to permit 2D interpolations (particles are not necessary on points of the grid)
        i = (state.particle_x) / state.dx
        j = (state.particle_y) / state.dx
        

        indices = tf.expand_dims(
            tf.concat(
                [tf.expand_dims(j, axis=-1), tf.expand_dims(i, axis=-1)], axis=-1
            ),
            axis=0,
        )

        u = interpolate_bilinear_tf(
            tf.expand_dims(state.U, axis=-1),
            indices,
            indexing="ij",
        )[:, :, 0]

        v = interpolate_bilinear_tf(
            tf.expand_dims(state.V, axis=-1),
            indices,
            indexing="ij",
        )[:, :, 0]

        thk = interpolate_bilinear_tf(
            tf.expand_dims(tf.expand_dims(state.thk, axis=0), axis=-1),
            indices,
            indexing="ij",
        )[0, :, 0]
        state.particle_thk = thk

        topg = interpolate_bilinear_tf(
            tf.expand_dims(tf.expand_dims(state.topg, axis=0), axis=-1),
            indices,
            indexing="ij",
        )[0, :, 0]
        state.particle_topg = topg

        smb = interpolate_bilinear_tf(
            tf.expand_dims(tf.expand_dims(state.smb, axis=0), axis=-1),
            indices,
            indexing="ij",
        )[0, :, 0]

        zeta = _rhs_to_zeta(params, state.particle_r)  # get the position in the column
        I0 = tf.cast(tf.math.floor(zeta * (params.iflo_Nz - 1)), dtype="int32")
        I0 = tf.minimum(
            I0, params.iflo_Nz - 2
        )  # make sure to not reach the upper-most pt
        I1 = I0 + 1
        zeta0 = tf.cast(I0 / (params.iflo_Nz - 1), dtype="float32")
        zeta1 = tf.cast(I1 / (params.iflo_Nz - 1), dtype="float32")

        lamb = (zeta - zeta0) / (zeta1 - zeta0)

        ind0 = tf.transpose(tf.stack([I0, tf.range(I0.shape[0])]))
        ind1 = tf.transpose(tf.stack([I1, tf.range(I1.shape[0])]))


        wei = tf.zeros_like(u)
        wei = tf.tensor_scatter_nd_add(wei, indices=ind0, updates=1 - lamb)
        wei = tf.tensor_scatter_nd_add(wei, indices=ind1, updates=lamb)

        if params.part_tracking_method == "simple":
            # adjust the relative height within the ice column with smb 
            state.particle_r = tf.where(
                thk > 0.1,
                tf.clip_by_value(state.particle_r * (thk - smb * state.dt) / thk, 0, 1),
                1,
            )

            state.particle_x = state.particle_x + state.dt * tf.reduce_sum(wei * u, axis=0)
            state.particle_y = state.particle_y + state.dt * tf.reduce_sum(wei * v, axis=0)
            state.particle_z = topg + thk * state.particle_r

        elif params.part_tracking_method == "3d":
            # uses the vertical velocity w computed in the vert_flow module

            w = interpolate_bilinear_tf(
                tf.expand_dims(state.W, axis=-1),
                indices,
                indexing="ij",
            )[:, :, 0]

            state.particle_x = state.particle_x + state.dt * tf.reduce_sum(wei * u, axis=0)
            state.particle_y = state.particle_y + state.dt * tf.reduce_sum(wei * v, axis=0)
            state.particle_z = state.particle_z + state.dt * tf.reduce_sum(wei * w, axis=0)

            # make sure the particle vertically remain within the ice body
            state.particle_z = tf.clip_by_value(state.particle_z, topg, topg + thk)
            # relative height of the particle within the glacier
            state.particle_r = (state.particle_z - topg) / thk
            # relative height will be slightly above 1 or below 1 if the particle is at the surface
            state.particle_r = tf.where(state.particle_r > 0.99, tf.ones_like(state.particle_r), state.particle_r)
            #if thk=0, state.particle_r takes value nan, so we set particle_r value to one in this case :
            state.particle_r = tf.where(thk== 0, tf.ones_like(state.particle_r), state.particle_r)
        
        else:
            print("Error: Name of the particles tracking method not recognised")

        # make sure particles remain within the icemask
        if hasattr(state, "icemask"):
            mask = tf.cast(
                tf.gather_nd(
                state.icemask,
                tf.stack(
                    [
                    tf.cast(state.particle_y / state.dx, tf.int32),
                    tf.cast(state.particle_x / state.dx, tf.int32),
                    ],
                    axis=1,
                ),
                ),
                tf.bool,
            )
            state.particle_x = tf.where(mask, state.particle_x, tf.zeros_like(state.particle_x))
            state.particle_y = tf.where(mask, state.particle_y, tf.zeros_like(state.particle_y))
        # if there is no icemask, we make sure the particles remain within the grid domain
        else:
            state.particle_x = tf.clip_by_value(state.particle_x, 0, state.x[-1] - state.x[0])
            state.particle_y = tf.clip_by_value(state.particle_y, 0, state.y[-1] - state.y[0])
        
        indices = tf.concat(
            [
                tf.expand_dims(tf.cast(j, dtype="int32"), axis=-1),
                tf.expand_dims(tf.cast(i, dtype="int32"), axis=-1),
            ],
            axis=-1,
        )
        updates = tf.cast(tf.where(state.particle_r == 1, state.particle_w, 0), dtype="float32")

        
        # this computes the sum of the weight of particles on a 2D grid
        state.weight_particles = tf.tensor_scatter_nd_add(
            tf.zeros_like(state.thk), indices, updates
        )
        
        # compute the englacial time
        state.particle_englt = state.particle_englt + tf.cast(
            tf.where(state.particle_r < 1, state.dt, 0.0), dtype="float32"
        )

        #    if int(state.t)%10==0:
        #        print("nb of part : ",state.xpos.shape)

        state.tcomp_particles[-1] -= time.time()
        state.tcomp_particles[-1] *= -1
        
        # aggregate immobile particles in the off-glacier area
        if params.part_aggregate_immobile_particles and (state.t.numpy() - state.tlast_seeding) == 0:
            state = aggregate_immobile_particles(state)
            
        if params.part_moraine_builder and (state.t.numpy() - state.tlast_mb) == 0:
            # reset topography to initial state before re-evaluating the off-glacier debris thickness
            state.topg = state.topg - state.debthick_offglacier
        
            # set state.particle_r of all particles where state.particle_thk == 0 to 1
            state.particle_r = tf.where(state.particle_thk == 0, tf.ones_like(state.particle_r), state.particle_r)
            
            # count particles in grid cells
            state.engl_w_sum = deb_count_particles(params, state)
            
            # add the debris thickness of off-glacier particles to the grid cells
            state.debthick_offglacier.assign(tf.reduce_sum(state.engl_w_sum, axis=0) / state.dx**2) # convert to m thickness by multiplying by representative volume (m3 debris per particle) and dividing by dx^2 (m2 grid cell area)
            # apply off-glacier mask (where particle_thk < 0)
            mask = state.thk > 0
            state.debthick_offglacier.assign(tf.where(mask, 0.0, state.debthick_offglacier))
            # add the resulting debris thickness to state.topg
            state.topg = state.topg + state.debthick_offglacier

            
    return state


# Seeding of particles, adapted from particles.py (Guillaume Jouvet)
def deb_seeding_particles(params, state):
    """
    here we define (particle_x,particle_y) the horiz coordinate of tracked particles
    and particle_r is the relative position in the ice column (scaled bwt 0 and 1)

    here we seed only the accum. area (+ a bit more), where there is
    significant ice, with a density of part_density_seeding particles per grid cell.
    """
    # Calculating volume per particle
    if params.part_density_seeding == []:
        state.d_in = 1.0
    else:
        state.d_in = interp1d_tf(state.d_in_array[:, 0], state.d_in_array[:, 1], state.t)
        
    state.volume_per_particle = params.part_frequency_seeding * state.d_in/1000 * state.dx**2 # Volume per particle in m3

    # Compute the gradient of the current land/ice surface
    dzdx, dzdy = compute_gradient_tf(state.usurf, state.dx, state.dx)
    state.slope_rad = tf.atan(tf.sqrt(dzdx**2 + dzdy**2))
    state.aspect_rad = -tf.atan2(dzdx, -dzdy)
    
    if params.part_slope_correction:
         state.volume_per_particle = state.volume_per_particle / tf.cos(state.slope_rad)
    else:
         state.volume_per_particle = state.volume_per_particle * tf.ones_like(state.slope_rad)
    
    if params.part_seeding_type == "conditions" or params.part_seeding_type == "both":    
        # apply the gradient condition on gridseed
        state.gridseed = state.gridseed & np.array(state.slope_rad > params.part_seed_slope / 180 * np.pi)
    
    if params.part_seeding_type == "csv_points":
        if not hasattr(state, 'particle_counter'):
            state.particle_counter = 0
        num_new_particles = tf.size(state.seeding_x).numpy()
        state.nID = tf.range(state.particle_counter, state.particle_counter + num_new_particles, dtype=tf.float32) # particle ID
        state.particle_counter += num_new_particles
        state.nparticle_x = state.seeding_x - state.x[0]    # x position of the particle
        state.nparticle_y = state.seeding_y - state.y[0]    # y position of the particle
        
        # Compute particle z positions based on the surface & x, y positions
        indices = tf.expand_dims(
            tf.concat(
                [tf.expand_dims(state.nparticle_y / state.dx, axis=-1), tf.expand_dims(state.nparticle_x / state.dx, axis=-1)], axis=-1
            ),
            axis=0,
        )
        state.nparticle_z = interpolate_bilinear_tf(
            tf.expand_dims(tf.expand_dims(state.usurf, axis=0), axis=-1),
            indices,
            indexing="ij",
        )[0, :, 0]
        
        state.nparticle_r = tf.ones_like(state.nparticle_x)           # relative position in the ice column
        # Find the grid cell each particle is in
        grid_particle_x = tf.cast(tf.floor(state.nparticle_x / state.dx), tf.int32)
        grid_particle_y = tf.cast(tf.floor(state.nparticle_y / state.dx), tf.int32)
        # Combine grid indices into a single tensor
        grid_indices = tf.stack([grid_particle_y, grid_particle_x], axis=1)
        # Assign w, thk, topg, and srcid values based on the grid cell the particle is in
        state.nparticle_w = tf.gather_nd(state.volume_per_particle, grid_indices)
        state.nparticle_t = tf.ones_like(state.nparticle_x) * state.t # "date of birth" of the particle (useful to compute its age)
        state.nparticle_englt = tf.zeros_like(state.nparticle_x)      # time spent by the particle burried in the glacier
        state.nparticle_thk = tf.gather_nd(state.thk, grid_indices)
        state.nparticle_topg = tf.gather_nd(state.topg, grid_indices)
        state.nparticle_srcid = tf.gather_nd(state.srcid, grid_indices)

    else:
        # Seeding
        I = (state.thk > 0) & (state.smb > -2) & state.gridseed # conditions for seeding area: where thk > 0, smb > -2 and gridseed (defined in initialize) is True
        if not hasattr(state, 'particle_counter'):
            state.particle_counter = 0
        num_new_particles = tf.size(state.X[I]).numpy()
        state.particle_counter += num_new_particles
        attributes_values = {
            "ID": tf.range(state.particle_counter, state.particle_counter + num_new_particles, dtype=tf.float32),
            "particle_x": state.X[I] - state.x[0],
            "particle_y": state.Y[I] - state.y[0],
            "particle_z": state.usurf[I],
            "particle_r": tf.ones_like(state.X[I]),
            "particle_w": tf.ones_like(state.X[I]) * state.volume_per_particle[I],
            "particle_t": tf.ones_like(state.X[I]) * state.t,
            "particle_englt": tf.zeros_like(state.X[I]),
            "particle_thk": state.thk[I],
            "particle_topg": state.topg[I],
            "particle_srcid": state.srcid[I],
        }

        for attr in state.particle_attributes:
            setattr(state, f"n{attr}", attributes_values[attr])
        
        # Calculate the amount of seeded particles
        state.seeded_particles = tf.size(state.nparticle_x)
        # Calculate the sum of seeded debris volume
        state.seeded_debris_volume = tf.reduce_sum(state.nparticle_w)
        
        if params.part_initial_rockfall:
            state = deb_initial_rockfall(params, state)
        
        if params.part_seeding_type == "slope_highres":
            state.nparticle_w = state.nparticle_w * state.gridseed_fraction[I] # adjust the weight of the particle based on the fraction of the grid cell area inside the polygons

def deb_initial_rockfall(params, state):
    """
    Moves the particle in x, y, and z direction along the surface gradient right after seeding until it reaches a slope lower than the given threshold (part_seed_slope). 
    This is supposed to simulate rockfall, moving the particles instantaneously to a lower slope, where they are more likely to enter a real glacier.
    
    Parameters:
    params (object): An object containing parameters required for the calculation.
    state (object): An object representing the current state of the glacier, including
                    attributes such as particle positions and velocities.
    
    Returns:
    state: The function updates the particle positions in the `state` object in place.
    """
    moving_particles = tf.ones_like(state.nparticle_x, dtype=tf.bool)
    iteration_count = 0
    max_iterations = 1000 / state.dx # Maximum number of iterations to prevent infinite loop (caused by infinitely oscillating particles)
    
    # Initial positions of the particles
    initx =  state.nparticle_x
    inity = state.nparticle_y
    
    moving_particles_any = tf.reduce_any(moving_particles)
    while moving_particles_any and iteration_count < max_iterations:
        i = state.nparticle_x / state.dx
        j = state.nparticle_y / state.dx
        indices = tf.expand_dims(
            tf.concat(
                [tf.expand_dims(j, axis=-1), tf.expand_dims(i, axis=-1)], axis=-1
            ),
            axis=0,
        )
        
        # Interpolate slope and aspect at particle positions
        particle_slope = interpolate_bilinear_tf(
            tf.expand_dims(tf.expand_dims(state.slope_rad, axis=0), axis=-1),
            indices,
            indexing="ij",
        )[0, :, 0]
        
        particle_aspect = interpolate_bilinear_tf(
            tf.expand_dims(tf.expand_dims(state.aspect_rad, axis=0), axis=-1),
            indices,
            indexing="ij",
        )[0, :, 0]
        
        # Update moving_particles mask (remove particles that have reached a slope lower than the threshold in the previous iteration)
        moving_particles = moving_particles & tf.math.greater_equal(particle_slope, params.part_seed_slope / 180 * np.pi)
        
        # Move only the particles that are still moving
        if tf.reduce_any(moving_particles):
            # Move particles along the aspect direction
            state.nparticle_x = tf.where(moving_particles, state.nparticle_x + tf.math.sin(particle_aspect) * state.dx, state.nparticle_x)
            state.nparticle_y = tf.where(moving_particles, state.nparticle_y + tf.math.cos(particle_aspect) * state.dx, state.nparticle_y)
        
        moving_particles_any = tf.reduce_any(moving_particles)
        iteration_count += 1
    # Calculate the difference between the final and initial positions
    diff_x = state.nparticle_x - initx
    diff_y = state.nparticle_y - inity
    
    # Apply an additional runout factor to the differences and add to the positions
    runout_factor = np.random.uniform(0, params.part_max_runout, size=diff_x.shape)  # Additional runout of particles after rockfall, uniformly distributed between 0 and 20% of the rockfall distance
    state.nparticle_x += diff_x * runout_factor
    state.nparticle_y += diff_y * runout_factor

    # Ensure particles remain within the domain
    state.nparticle_x = tf.clip_by_value(state.nparticle_x, 0, state.x[-1] - state.x[0])
    state.nparticle_y = tf.clip_by_value(state.nparticle_y, 0, state.y[-1] - state.y[0])
    
    # Update particle z positions based on the surface & x, y positions
    indices = tf.expand_dims(
        tf.concat(
            [tf.expand_dims(state.nparticle_y / state.dx, axis=-1), tf.expand_dims(state.nparticle_x / state.dx, axis=-1)], axis=-1
        ),
        axis=0,
    )
    state.nparticle_z = interpolate_bilinear_tf(
        tf.expand_dims(tf.expand_dims(state.usurf, axis=0), axis=-1),
        indices,
        indexing="ij",
    )[0, :, 0]
    
    return state
    
    
def deb_thickness(params, state):
    """
    Calculates debris thickness based on particle volumes counted per pixel.
    
    Parameters:
    params (object): An object containing parameters required for the calculation.
    state (object): An object representing the current state of the glacier, including
                    attributes such as time, debris thickness, surface mass balance,
                    and thickness.
    Returns:
    state: The function updates the `debthick` attribute of the `state` object in place.
    """
    
    if (state.t.numpy() - state.tlast_mb) == 0:
        if not params.part_moraine_builder:
            state.engl_w_sum = deb_count_particles(params, state) # count particles and their volumes in grid cells
        state.debthick.assign(state.engl_w_sum[-1, :, :] / state.dx**2) # convert to m thickness by dividing representative volume (m3 debris per particle) by dx^2 (m2 grid cell area)
        state.debcon.assign(tf.reduce_sum(state.engl_w_sum[:-1, :, :], axis=0) / (state.dx**2 * state.thk)) # convert to m depth-averaged volumetric debris concentration by dividing representative volume (m3 debris per particle) by dx^2 (m2 grid cell area) and ice thickness thk
        state.debcon_vert.assign(tf.where(state.thk[None,:,:] > 0, state.engl_w_sum[:-1, :, :] / (state.dx**2 * state.thk[None,:,:]) * params.iflo_Nz, 0.0)) # vertically resolved debris concentration
        state.debflux_supragl.assign(state.debthick * getmag(state.uvelsurf,state.vvelsurf)) # debris flux (supraglacial)
        state.debflux_engl.assign(tf.reduce_sum(state.engl_w_sum[:-1, :, :] * tf.sqrt(state.U**2 + state.V**2), axis=0) / state.dx**2) # debris flux (englacial)
        state.debflux.assign(state.debflux_supragl + state.debflux_engl) # debris flux (englacial and supraglacial)
        state.thk_deb.assign(state.thk) # ice thickness at the beginning of the time step
        mask = (state.smb > 0) | (state.thk == 0) # mask out off-glacier areas and accumulation area
        state.debthick.assign(tf.where(mask, 0.0, state.debthick))
        mask = state.thk == 0 # mask out off-glacier areas and accumulation area
        state.debcon.assign(tf.where(mask, 0.0, state.debcon))
    return state

# Count surface particles in grid cells
def deb_count_particles(params, state):
    """
    Count surface and englacial particles within a grid cell.

    Parameters:
    state (object): An object containing particle coordinates (particle_x, particle_y) and grid cell boundaries (X, Y).

    Returns:
    engl_w_sum: A 3D array with the sum of particle debris volume (particle_w) of englacial particles in each grid cell, sorted into vertical bins.
    """
    sorted_x = np.sort(state.x)
    sorted_y = np.sort(state.y)
    grid_particle_y = np.digitize(state.particle_x, sorted_x - sorted_x[0]) - 1
    grid_particle_x = np.digitize(state.particle_y, sorted_y - sorted_y[0]) - 1
    # Create depth bins for each pixel
    depth_bins = tf.linspace(0.0, 1.0, params.iflo_Nz + 1)
    
    # Initialize a 3D array to hold the counts for each depth bin
    engl_w_sum = tf.zeros((params.iflo_Nz + 1,) + state.usurf.shape, dtype=tf.float32)
    
    # Iterate over each depth bin
    for k in range(params.iflo_Nz + 1):
        # Create a mask for particles within the current depth bin
        if depth_bins[k] < depth_bins[-1]:
            bin_mask = (state.particle_r >= depth_bins[k]) & (state.particle_r < depth_bins[k + 1])
        else:
            bin_mask = state.particle_r >= depth_bins[k]
            
        # Filter particles within the current depth bin
        filtered_particle_x = tf.boolean_mask(state.particle_x, bin_mask)
        filtered_particle_y = tf.boolean_mask(state.particle_y, bin_mask)
        
        # Convert particle positions to grid indices
        grid_particle_x = tf.cast(tf.floor(filtered_particle_x / state.dx), tf.int32)
        grid_particle_y = tf.cast(tf.floor(filtered_particle_y / state.dx), tf.int32)
        
        # Count particles in each grid cell and add their assigned volume
        indices = tf.stack([grid_particle_y, grid_particle_x], axis=1)
        engl_w_sum = tf.tensor_scatter_nd_add(engl_w_sum, tf.concat([tf.fill([tf.shape(indices)[0], 1], k), indices], axis=1), tf.boolean_mask(state.particle_w, bin_mask))
    
    return engl_w_sum


# debris-covered mass balance computation, adapted from smb_simple.py (Guillaume Jouvet)
def deb_smb(params, state):
    # update debris-SMB whenever SMB is updated (tlast_mb is set to state.t in smb_simple.py)
    if (state.t - state.tlast_mb) == 0:
        
        # adjust smb based on debris thickness
        if hasattr(state, "debthick"):
            mask = state.debthick > 0
            state.smb = tf.where(mask, state.smb * params.smb_oestrem_D0 / (params.smb_oestrem_D0 + state.debthick), state.smb)
        
    return state


# Conversion functions for zeta and rhs
def _zeta_to_rhs(params, zeta):
    return (zeta / params.iflo_vert_spacing) * (
        1.0 + (params.iflo_vert_spacing - 1.0) * zeta
    )
def _rhs_to_zeta(params, rhs):
    if params.iflo_vert_spacing == 1:
        rhs = zeta
    else:
        DET = tf.sqrt(
            1 + 4 * (params.iflo_vert_spacing - 1) * params.iflo_vert_spacing * rhs
        )
        zeta = (DET - 1) / (2 * (params.iflo_vert_spacing - 1))

    #           temp = params.iflo_Nz*(DET-1)/(2*(params.iflo_vert_spacing-1))
    #           I=tf.cast(tf.minimum(temp-1,params.iflo_Nz-1),dtype='int32')
    return zeta


# Debris mask shapefile reader, adapted from include_icemask.py (Andreas Henz)    
def read_shapefile(filepath):
    try:
        # Read the shapefile
        gdf = gpd.read_file(filepath)

        # Print the information about the shapefile
        print("-----------------------")
        print("Debris Mask Shapefile information:")
        print("Number of features (polygons):", len(gdf))
        print("EPSG code: ", gdf.crs.to_epsg())
        print("Geometry type:", gdf.geometry.type.unique()[0])
        print("-----------------------")

        # Return the GeoDataFrame
        return gdf
    
    except Exception as e:
        print("Error reading shapefile:", e)
        
def compute_mask_and_srcid(state, gdf):
    # Flatten the X and Y coordinates and convert to numpy
    flat_X = state.X.numpy().flatten()
    flat_Y = state.Y.numpy().flatten()

    # Create lists to store the mask values and source IDs
    mask_values = []
    srcid_values = []

    # Iterate over each grid point
    for x, y in zip(flat_X, flat_Y):
        point = Point(x, y)
        inside_polygon = False
        srcid = -1  # Default value if point is not inside any polygon

        # Check if the point is inside any polygon in the GeoDataFrame
        for idx, geom in enumerate(gdf.geometry):
            if point.within(geom):
                inside_polygon = True
                srcid = gdf.iloc[idx]['FID']  # Get the FID of the polygon
                break  # if it is inside one polygon, don't check for others

        # Append the corresponding mask value and source ID to the lists
        mask_values.append(1 if inside_polygon else 0)  # 1 for debris input area, 0 for no debris
        srcid_values.append(srcid)

    # Reshape
    mask_values = np.array(mask_values, dtype=np.float32).reshape(state.X.shape)
    srcid_values = np.array(srcid_values, dtype=np.int32).reshape(state.X.shape)

    return mask_values, srcid_values

def read_seeding_points_from_csv(filepath):
    """
    Reads seeding points from a CSV file and returns their x and y coordinates as separate tensors.

    Parameters:
    filepath (str): Path to the CSV file containing seeding points.

    Returns:
    x, y: Two tensors containing the x and y coordinates of the seeding points.
    """
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(filepath)

        # Define possible aliases for x and y columns
        x_aliases = ['x', 'particle_x', 'xcoord', 'xpos']
        y_aliases = ['y', 'particle_y', 'ycoord', 'ypos']

        # Find the actual x and y columns in the DataFrame
        x_column = next((col for col in x_aliases if col in df.columns), None)
        y_column = next((col for col in y_aliases if col in df.columns), None)

        if not x_column or not y_column:
            raise ValueError("Missing required x or y column in the CSV file. Ensure one of the following aliases is present: "
                                f"x: {x_aliases}, y: {y_aliases}")

        # Convert the DataFrame columns to tensors
        x = tf.constant(df[x_column].values, dtype=tf.float32)
        y = tf.constant(df[y_column].values, dtype=tf.float32)

        # Print information about the loaded seeding points
        print(f"Loaded {len(df)} seeding points from {filepath} using columns '{x_column}' and '{y_column}'.")
        return x, y
    
    except Exception as e:
        print(f"Error reading seeding points from CSV: {e}")
        raise

def aggregate_immobile_particles(state):
        J = (state.particle_thk > 1)
        immobile_particles = tf.logical_not(J)

        # Mask immobile particles
        immobile_data = {
            "x": tf.boolean_mask(state.particle_x, immobile_particles),
            "y": tf.boolean_mask(state.particle_y, immobile_particles),
            "w": tf.boolean_mask(state.particle_w, immobile_particles),
            "t": tf.boolean_mask(state.particle_t, immobile_particles),
            "englt": tf.boolean_mask(state.particle_englt, immobile_particles),
            "srcid": tf.boolean_mask(state.particle_srcid, immobile_particles),
        }

        # Compute grid indices
        grid_indices = tf.stack([
            tf.cast(tf.floor(immobile_data["y"] / state.dx), tf.int32),
            tf.cast(tf.floor(immobile_data["x"] / state.dx), tf.int32)
        ], axis=1)

        # Aggregate immobile particle data
        aggregated_data = {
            "w_sum": tf.tensor_scatter_nd_add(tf.zeros_like(state.usurf), grid_indices, immobile_data["w"]),
            "t_sum": tf.tensor_scatter_nd_add(tf.zeros_like(state.usurf), grid_indices, immobile_data["t"]),
            "englt_sum": tf.tensor_scatter_nd_add(tf.zeros_like(state.usurf), grid_indices, immobile_data["englt"]),
            "srcid_sum": tf.tensor_scatter_nd_add(tf.zeros_like(state.usurf), grid_indices, immobile_data["srcid"]),
            "count": tf.tensor_scatter_nd_add(tf.zeros_like(state.usurf), grid_indices, tf.ones_like(immobile_data["t"])),
        }

        # Compute means
        aggregated_data["t_mean"] = tf.math.divide_no_nan(aggregated_data["t_sum"], aggregated_data["count"])
        aggregated_data["englt_mean"] = tf.math.divide_no_nan(aggregated_data["englt_sum"], aggregated_data["count"])
        aggregated_data["srcid_mean"] = tf.math.divide_no_nan(aggregated_data["srcid_sum"], aggregated_data["count"])

        # Remove immobile particles
        for attr in state.particle_attributes:
            setattr(state, attr, tf.boolean_mask(getattr(state, attr), J))

        # Re-seed aggregated particles
        I = aggregated_data["w_sum"] > 0
        if tf.reduce_any(I):
            if not hasattr(state, 'particle_counter'):
                state.particle_counter = 0
            num_new_particles = tf.reduce_sum(tf.cast(I, tf.int32)).numpy()
            state.nID = tf.range(state.particle_counter, state.particle_counter + num_new_particles, dtype=tf.float32)
            state.particle_counter += num_new_particles

            # Weighted averages for re-seeding
            weighted_x_sum = tf.tensor_scatter_nd_add(tf.zeros_like(state.usurf), grid_indices, immobile_data["x"] * immobile_data["w"])
            weighted_y_sum = tf.tensor_scatter_nd_add(tf.zeros_like(state.usurf), grid_indices, immobile_data["y"] * immobile_data["w"])
            total_weight = aggregated_data["w_sum"]
            avg_x = tf.math.divide_no_nan(weighted_x_sum, total_weight)
            avg_y = tf.math.divide_no_nan(weighted_y_sum, total_weight)

            # Assign new particle attributes
            state.nparticle_x = avg_x[I]
            state.nparticle_y = avg_y[I]
            state.nparticle_z = state.usurf[I]
            state.nparticle_r = tf.ones_like(state.nparticle_x)
            state.nparticle_w = aggregated_data["w_sum"][I]
            state.nparticle_t = aggregated_data["t_mean"][I]
            state.nparticle_englt = aggregated_data["englt_mean"][I]
            state.nparticle_thk = state.thk[I]
            state.nparticle_topg = state.topg[I]
            state.nparticle_srcid = aggregated_data["srcid_mean"][I]

            # Merge new particles with existing ones
            for attr in state.particle_attributes:
                setattr(state, attr, tf.concat([getattr(state, attr), getattr(state, f"n{attr}")], axis=-1))
            
            return state