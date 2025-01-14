#!/usr/bin/env python3

# Author: Florian Hardmeier, florian.hardmeier@geo.uzh.ch
# Date: 01.11.2024

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

def params(parser):
    # Particle seeding
    parser.add_argument(
        "--part_seeding_type",
        type=str,
        default="conditions",
        help="Seeding type (conditions, shapefile, or both). 'conditions' seeds particles based on conditions (e.g. slope, thickness, velocity), 'shapefile' seeds particles in area defined by a shapefile, 'both' applies conditions and shapefile (default: conditions)",
    )
    parser.add_argument(
        "--part_debrismask_shapefile",
        type=str,
        default="debrismask.shp",
        help="Debris mask input file (default: debrismask.shp)",
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
    
    # Particle tracking
    parser.add_argument(
        "--part_tracking_method",
        type=str,
        default="3d",
        help="Method for tracking particles (simple or 3d)",
    )
    parser.add_argument(
        "--part_remove_immobile_particles",
        type=bool,
        default=False,
        help="Remove immobile particles (default: False)",
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
    
    # initialize the debris thickness
    state.debthick = tf.Variable(tf.zeros_like(state.usurf, dtype=tf.float32))

def update(params, state):
    # update the particle tracking by calling the particles function, adapted from module particles.py
    state = deb_particles(params, state)
    
    # update debris thickness based on particle count in grid cells (at every SMB update time step)
    state = deb_thickness(state)
        
    # update the mass balance (SMB) depending by debris thickness, using clean-ice SMB from smb_simple.py
    state = deb_smb(params, state)

def finalize(params, state):
    pass



def initialize_seeding(params, state):
    # initialize particle seeding
    state.tlast_seeding = -1.0e5000
    state.tcomp_particles = []

    # initialize trajectories
    state.particle_x = tf.Variable([])
    state.particle_y = tf.Variable([])
    state.particle_z = tf.Variable([])
    state.particle_r = tf.Variable([])
    state.particle_w = tf.Variable([])  # debris volume equivalent to the particle
    state.particle_t = tf.Variable([])
    state.particle_englt = tf.Variable([])  # englacial time
    state.particle_topg = tf.Variable([])
    state.particle_thk = tf.Variable([])
    state.particle_srcid = tf.Variable([]) # source area id from debris mask shapefile (if used)
    state.srcid = tf.zeros_like(state.thk, dtype=tf.float32)
    
    state.pswvelbase = tf.Variable(tf.zeros_like(state.thk),trainable=False)
    state.pswvelsurf = tf.Variable(tf.zeros_like(state.thk),trainable=False)
        
    # Grid seeding based on conditions, written by Andreas H., adapted by Florian H.
    # Seeds particles randomly within a grid cell that fulfills the conditions
    # Currently only seeding based on slope implemented, but can be extended to other conditions
    if params.part_seeding_type == "conditions":
        state.gridseed = np.ones_like(state.thk, dtype=bool)
              
        # compute the gradient of the ice surface
        dzdx , dzdy = compute_gradient_tf(state.usurf,state.dx,state.dx)
        state.slope_rad = tf.atan(tf.sqrt(dzdx**2 + dzdy**2))
        
        # initialize d_in array
        if params.part_density_seeding != []:
            state.d_in_array = np.array(params.part_density_seeding[1:]).astype(np.float32)
        
        # seed where gridseed is True and the slope is steep
        state.gridseed = state.gridseed & np.array(state.slope_rad > params.part_seed_slope/180*np.pi)
        
    # Seeding based on shapefile, adapted from include_icemask (Andreas Henz) NOT WORKING YET    
    elif params.part_seeding_type == "shapefile":
        # read_shapefile
        gdf = read_shapefile(params.part_debrismask_shapefile)

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

        # Initialize d_in array
        if params.part_density_seeding != []:
            state.d_in_array = np.array(params.part_density_seeding[1:]).astype(np.float32)
        
        # Define debrismask and srcid
        state.gridseed = tf.constant(mask_values, dtype=tf.bool)
        state.srcid = tf.constant(srcid_values, dtype=tf.float32)
        
        # If gridseed is empty, raise an error
        if not np.any(state.gridseed):
            raise ValueError("Shapefile not within icemask! Watch out for coordinate system!")
        
        
    elif params.part_seeding_type == "both":
        # read_shapefile
        gdf = read_shapefile(params.part_debrismask_shapefile)

        # Flatten the X and Y coordinates and convert to numpy
        flat_X = state.X.numpy().flatten()
        flat_Y = state.Y.numpy().flatten()

        # Create a list to store the mask values
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

        # define debrismask and srcid
        state.gridseed = tf.constant(mask_values, dtype=tf.bool)
        state.srcid = tf.constant(srcid_values, dtype=tf.float32)
        
        # if gridseed is empty, raise an error
        if not np.any(state.gridseed):
            raise ValueError("Shapefile not within icemask! Watch out for coordinate system!")

        # compute the gradient of the ice surface
        dzdx, dzdy = compute_gradient_tf(state.usurf, state.dx, state.dx)
        state.slope_rad = tf.atan(tf.sqrt(dzdx**2 + dzdy**2))

        # initialize d_in array
        if params.part_density_seeding != []:
            state.d_in_array = np.array(params.part_density_seeding[1:]).astype(np.float32)
        
        # apply the gradient condition on the shapefile mask
        state.gridseed = state.gridseed & np.array(state.slope_rad > params.part_seed_slope / 180 * np.pi)
        
    elif params.part_seeding_type == "slope_shp":
        # read_shapefile
        gdf = read_shapefile(params.part_debrismask_shapefile)

        # Flatten the X and Y coordinates and convert to numpy
        flat_X = state.X.numpy().flatten()
        flat_Y = state.Y.numpy().flatten()
        # Initialize an array to store the fraction of area per pixel inside the polygons
        fraction_area = np.zeros_like(state.X, dtype=np.float32)

        # Iterate over each grid cell
        for i in range(state.X.shape[0]):
            for j in range(state.X.shape[1]):
                # Create a polygon representing the grid cell
                cell_polygon = Point(state.X[i, j], state.Y[i, j]).buffer(state.dx / 2, cap_style = 3)

                # Calculate the intersection area with each polygon in the GeoDataFrame
                intersection_area = 0.0
                for geom in gdf.geometry:
                    intersection_area += cell_polygon.intersection(geom).area

                # Calculate the fraction of the grid cell area inside the polygons
                fraction_area[i, j] = intersection_area / cell_polygon.area

        # Convert the fraction_area array to a TensorFlow constant
        state.gridseed_fraction = tf.constant(fraction_area, dtype=tf.float32)
        print("Fraction area shape:", state.gridseed_fraction.shape)
        state.gridseed = tf.cast(state.gridseed_fraction > 0, dtype=tf.bool)    
        
    return state


# Particle tracking, adapted from particles.py (Guillaume Jouvet)
def deb_particles(params, state):
    if hasattr(state, "logger"):
        state.logger.info("Update particle tracking at time : " + str(state.t.numpy()))
        
    if (state.t.numpy() - state.tlast_seeding) >= params.part_frequency_seeding:
        deb_seeding_particles(params, state)
        
        # merge the new seeding points with the former ones
        state.particle_x = tf.Variable(tf.concat([state.particle_x, state.nparticle_x], axis=-1),trainable=False)
        state.particle_y = tf.Variable(tf.concat([state.particle_y, state.nparticle_y], axis=-1),trainable=False)
        state.particle_z = tf.Variable(tf.concat([state.particle_z, state.nparticle_z], axis=-1),trainable=False)
        state.particle_r = tf.Variable(tf.concat([state.particle_r, state.nparticle_r], axis=-1),trainable=False)
        state.particle_w = tf.Variable(tf.concat([state.particle_w, state.nparticle_w], axis=-1),trainable=False)   
        state.particle_t = tf.Variable(tf.concat([state.particle_t, state.nparticle_t], axis=-1),trainable=False)
        state.particle_englt = tf.Variable(tf.concat([state.particle_englt, state.nparticle_englt], axis=-1),trainable=False)
        state.particle_topg = tf.Variable(tf.concat([state.particle_topg, state.nparticle_topg], axis=-1),trainable=False)
        state.particle_thk = tf.Variable(tf.concat([state.particle_thk, state.nparticle_thk], axis=-1),trainable=False)
        state.particle_srcid = tf.Variable(tf.concat([state.particle_srcid, state.nparticle_srcid], axis=-1),trainable=False)

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
            print("Error : Name of the particles tracking method not recognised")

        # make sur the particle remains in the horiz. comp. domain
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
        
        # remove immobile particles (if option is set)
        if params.part_remove_immobile_particles:
            J = (state.particle_thk > 1)
            state.particle_x = tf.boolean_mask(state.particle_x, J)
            state.particle_y = tf.boolean_mask(state.particle_y, J)
            state.particle_z = tf.boolean_mask(state.particle_z, J)
            state.particle_r = tf.boolean_mask(state.particle_r, J)
            state.particle_w = tf.boolean_mask(state.particle_w, J)
            state.particle_t = tf.boolean_mask(state.particle_t, J)
            state.particle_englt = tf.boolean_mask(state.particle_englt, J)
            state.particle_thk = tf.boolean_mask(state.particle_thk, J)
            state.particle_topg = tf.boolean_mask(state.particle_topg, J)
            state.particle_srcid = tf.boolean_mask(state.particle_srcid, J)
            
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
    
    # Randomized seeding based on high-res slope threshold
    if params.part_seeding_type == "slope_shp":
        # Determine where to seed particles based on the probability (gridseed_fraction)
        seeding_mask = np.random.rand(*state.gridseed_fraction.shape) < state.gridseed_fraction
        # Apply the seeding mask to the gridseed
        state.gridseed = tf.cast(seeding_mask, dtype=tf.bool)
    elif params.part_seeding_type == "conditions" or params.part_seeding_type == "both":
        # Compute the gradient of the current land/ice surface
        dzdx, dzdy = compute_gradient_tf(state.usurf, state.dx, state.dx)
        state.slope_rad = tf.atan(tf.sqrt(dzdx**2 + dzdy**2))
        state.aspect_rad = -tf.atan2(dzdx, -dzdy)
        
        # apply the gradient condition on gridseed
        state.gridseed = state.gridseed & np.array(state.slope_rad > params.part_seed_slope / 180 * np.pi)
    
    # Seeding
    I = (state.thk > 0) & (state.smb > -2) & state.gridseed # conditions for seeding area: where thk > 0, smb > -2 and gridseed (defined in initialize) is True
    state.nparticle_x  = state.X[I] - state.x[0]            # x position of the particle
    state.nparticle_y  = state.Y[I] - state.y[0]            # y position of the particle
    state.nparticle_z  = state.usurf[I]                     # z position of the particle
    state.nparticle_r = tf.ones_like(state.X[I])            # relative position in the ice column
    state.nparticle_w  = tf.ones_like(state.X[I]) * state.volume_per_particle   # weight of the particle
    state.nparticle_t  = tf.ones_like(state.X[I]) * state.t # "date of birth" of the particle (useful to compute its age)
    state.nparticle_englt = tf.zeros_like(state.X[I])       # time spent by the particle burried in the glacier
    state.nparticle_thk = state.thk[I]                      # ice thickness at position of the particle
    state.nparticle_topg = state.topg[I]                    # z position of the bedrock under the particle
    state.nparticle_srcid = state.srcid[I]                  # source area id from debris mask shapefile (if used)
    
    if params.part_initial_rockfall:
        state = deb_initial_rockfall(params, state)
    

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
    
    # Save initial positions
    initial_x = state.nparticle_x
    initial_y = state.nparticle_y
    
    while tf.reduce_any(moving_particles) and iteration_count < max_iterations:
                
        # Interpolate particle positions to grid indices
        i = state.nparticle_x / state.dx
        j = state.nparticle_y / state.dx
        indices = tf.expand_dims(
            tf.concat(
                [tf.expand_dims(j, axis=-1), tf.expand_dims(i, axis=-1)], axis=-1
            ),
            axis=0,
        )
        
        # Interpolate slope at particle positions
        particle_slope = interpolate_bilinear_tf(
            tf.expand_dims(tf.expand_dims(state.slope_rad, axis=0), axis=-1),
            indices,
            indexing="ij",
        )[0, :, 0]
        
        # Interpolate aspect at particle positions
        particle_aspect = interpolate_bilinear_tf(
            tf.expand_dims(tf.expand_dims(state.aspect_rad, axis=0), axis=-1),
            indices,
            indexing="ij",
        )[0, :, 0]
        
        # Update moving_particles mask (remove particles that have reached a slope lower than the threshold in the previous iteration)
        moving_particles = moving_particles & (particle_slope >= params.part_seed_slope / 180 * np.pi)
        
        # Move only the particles that are still moving
        if tf.reduce_any(moving_particles):
            # Move particles along the aspect direction
            state.nparticle_x = tf.where(moving_particles, state.nparticle_x + tf.sin(particle_aspect) * state.dx, state.nparticle_x)
            state.nparticle_y = tf.where(moving_particles, state.nparticle_y + tf.cos(particle_aspect) * state.dx, state.nparticle_y)
                
            # Ensure particles remain within the domain
            state.nparticle_x = tf.clip_by_value(state.nparticle_x, 0, state.x[-1] - state.x[0])
            state.nparticle_y = tf.clip_by_value(state.nparticle_y, 0, state.y[-1] - state.y[0])
        
        iteration_count += 1
    
    # Calculate the difference between the final and initial positions
    diff_x = state.nparticle_x - initial_x
    diff_y = state.nparticle_y - initial_y
    
    # Apply an additional runout factor to the differences and add to the positions
    runout_factor = np.random.uniform(0, params.part_max_runout, size=diff_x.shape)  # Additional runout of particles after rockfall, uniformly distributed between 0 and 20% of the rockfall distance
    state.nparticle_x += diff_x * runout_factor
    state.nparticle_y += diff_y * runout_factor

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
    
    
def deb_thickness(state):
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
        counts = deb_count_particles_in_grid(state) # count particles and their volumes in grid cells
        state.debthick.assign(counts / state.dx**2) # convert to m thickness by multiplying by representative volume (m3 debris per particle) and dividing by dx^2 (m2 grid cell area)
        mask = (state.smb > 0) | (state.thk == 0) # mask out off-glacier areas and accumulation area
        state.debthick.assign(tf.where(mask, 0.0, state.debthick))

    return state

# Count surface particles in grid cells
def deb_count_particles_in_grid(state):
    """
    Count particle coordinates within a grid cell.

    Parameters:
    state (object): An object containing particle coordinates (particle_x, particle_y) and grid cell boundaries (X, Y).

    Returns:
    counts: A 2D array with the count of particles in each grid cell.
    """
    # Filter particles where particle_r is 1 (debris on the surface)
    mask = state.particle_r == 1
    filtered_particle_x = state.particle_x[mask]
    filtered_particle_y = state.particle_y[mask]
    
    # Convert particle positions to grid indices
    grid_particle_y = np.digitize(filtered_particle_x, state.x - state.x[0]) - 1
    grid_particle_x = np.digitize(filtered_particle_y, state.y - state.y[0]) - 1
    

    # Initialize a 2D array to hold the counts
    counts = np.zeros_like(state.usurf, dtype=int)
    
    # Count particles in each grid cell and add their assigned volume
    for x, y, w in zip(grid_particle_x, grid_particle_y, state.particle_w[mask]):
        if 0 <= x < counts.shape[0] and 0 <= y < counts.shape[1]:
            counts[x, y] += w
    
    return counts


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