import numpy as np

#Dead center on the depth camera
dp_c = (256.684, 207.085)
#Focal values
dp_f = (366.193, 366.193)

#Given a z-value for the depth and a 2d position in
#depth image index coordinates, convert to 2d rgb image index
#coordinates
def depth_to_rgb(z, depth_pos):
    x, y = depth_pos
    #rx = 2.99426 * x + 213.909
    #rx = 3.1 * x + 190

    #rx = 2.95 * x + 225
    rx = 2.90 * x + 240

    #ry = 2.90452 * y - 56.3971
    ry = 2.90452 * y - 53
 

    #rx = 3.13 * x + 186.2678910
    #ry = 3.153 * y - 91.69865414
    #rx = 2.80632 * x + 324.643 - 50.0
    #ry = 2.97718 * y - 59.6628 - 20.0
    return np.array([rx, ry])

index_array = np.transpose(np.indices((512, 424)), axes=[1, 2, 0]).astype(np.float32)
centered_focused_index_array = (index_array - dp_c) / dp_f
rgb_array = index_array * np.array([2.90, 2.90452], dtype=np.float32) + np.array([240, -53])
rgb_array = rgb_array.astype(np.int32)
rgb_height = 1080
rgb_width = 1920
rgb_xs = rgb_array[:, :, 0]
rgb_ys = rgb_array[:, :, 1]
rgb_inbounds = np.logical_not(np.logical_or(np.logical_or(rgb_xs < 0, rgb_xs >= rgb_width), 
                                            np.logical_or(rgb_ys < 0, rgb_ys >= rgb_height)))

#Given a matrix of z-values (indexed by depth index coordinates),
#convert to world coordinates
def depth_buffer_to_world(depth_buf):
    depth_buf = np.transpose(depth_buf)
    zs = np.expand_dims(depth_buf, axis=2)
    xys = zs * centered_focused_index_array
    return np.concatenate((xys, zs), axis=2)

def depth_buffer_to_rgb(depth_buf):
    return rgb_array

def depth_buffer_to_rgb_inbounds(depth_buf):
    return rgb_inbounds

#Given a z-value for the depth and a 2d position
#in depth image index coordinates, convert to 3d world coordinates
def depth_to_world(z, depth_pos):
    x, y = z * ((depth_pos - dp_c) / dp_f)
    return np.array([x, y, z])
