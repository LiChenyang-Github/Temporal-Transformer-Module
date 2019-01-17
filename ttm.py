
import tensorflow as tf




### This func use two paras [scale, translation] to do the affine transformation to the input.
def temporal_transformer_network_2paras(input, final_frame_nb, theta, **kwargs):
    """
    input: shape (B, C), where B means batchsize and C is the dimensionality of each sample.
    final_frame_nb: the final frame number, EX: if we down or up sample all the samples to 39 frames, final_frame_nb is 39.
    theta: shape (B, 2), where 2 means [scale, translation]
    
    Note:
        In my case (skeleton base action recognition), each row of input is organized in order of [joint, axis(xyz), frame]
        For example, suppose that there are 6 joints and 39 frames, then C is 6 * 3 * 39 = 702.
        Each row of input is a vector look like [j0_x_f0, j0_x_f1, ... , j0_x_f38, j0_y_f0 , .. , j0_y_f38, ... , j0_z_f38, j1_x_f0, ... , j5_z_f38]
    """

    B = tf.shape(input)[0]
    C = tf.shape(input)[1]

    theta = tf.reshape(theta, (B, 2))

    batch_grids = affine_grid_generator_2paras(C, final_frame_nb, theta)

    out_feat = linear_sampler_2paras(input, batch_grids, C, final_frame_nb)

    return out_feat

### generate the grid after affine transformation (theta) for the linear_sampler func.
def affine_grid_generator_2paras(C, final_frame_nb, theta):

    B = tf.shape(theta)[0]
    N = C // final_frame_nb # N is joint * axis, 6 * 3 = 18 in my case.
    t = tf.linspace(-1.0, 1.0, final_frame_nb)

    ones = tf.ones_like(t)

    sampling_grid = tf.stack([t, ones])

    sampling_grid = tf.expand_dims(sampling_grid, axis=0)

    sampling_grid = tf.tile(sampling_grid, tf.stack([B, 1, N]))

    theta = tf.reshape(theta, [B, 1, 2])

    theta = tf.cast(theta, 'float32')

    sampling_grid = tf.cast(sampling_grid, 'float32')

    batch_grids = tf.matmul(theta, sampling_grid)   # the shape of batch_grids here is [B, 1, C]

    batch_grids = tf.reshape(batch_grids, [B, N, final_frame_nb])   # the shape of batch_grids here is [B, N, C]

    return batch_grids
    

def linear_sampler_2paras(input_feat, batch_grids, C, final_frame_nb):
    """
        input_feat: [B, C]
        batch_grids: [B, N, final_frame_nb], N = C // final_frame_nb
    """

    B = tf.shape(input_feat)[0]

    max_t = tf.cast(final_frame_nb - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    batch_grids = tf.cast(batch_grids, 'float32')

    batch_grids = 0.5 * ((batch_grids + 1.0) * tf.cast(max_t, 'float32'))

    t0 = tf.cast(tf.floor(batch_grids), 'int32')
    t1 = t0 + 1

    t0 = tf.clip_by_value(t0, zero, max_t)
    t1 = tf.clip_by_value(t1, zero, max_t)

    Ta = get_pixel_value_2paras(input_feat, t0, C, final_frame_nb)
    Tb = get_pixel_value_2paras(input_feat, t1, C, final_frame_nb)

    t0 = tf.cast(t0, 'float32')
    t1 = tf.cast(t1, 'float32')

    wa = t1 - batch_grids
    wb = batch_grids - t0

    out = tf.add(wa*Ta, wb*Tb)

    out = tf.reshape(out, [B, C])

    return out


def get_pixel_value_2paras(input_feat, t, C, final_frame_nb):

    B = tf.shape(input_feat)[0]
    N = C // final_frame_nb

    input_feat = tf.reshape(input_feat, [B, N, final_frame_nb])

    batch_idx = tf.range(0, B)
    batch_idx = tf.reshape(batch_idx, (B, 1, 1))
    b = tf.tile(batch_idx, (1, N, final_frame_nb))

    N_idx = tf.range(0, N)
    N_idx = tf.reshape(N_idx, (1, N, 1))
    n = tf.tile(N_idx, (B, 1, final_frame_nb))

    indices = tf.stack([b, n, t], axis=3)   # indices is 4-dimension [B, N, final_frame_nb, 3]

    input_feat = tf.gather_nd(input_feat, indices)  # input_feat is 3-dimension [B, N, final_frame_nb]

    return input_feat