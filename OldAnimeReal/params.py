#NETWORK HYPERPARAMETERS
#The number of features in each layer (important hyperparameter)
F = 32
#The number of intermediate scale-convolutional layers in the network
L = 12
#The number of layers per densely connected block
B = 6

#The number of features that we downsample to from an initial stride-2
#convolution from 256x256 to 128x128
feats_128 = 32

#HEATMAP GENERATOR NETWORK TRAINING HYPERPARAMETERS
num_training_iters = 400000
batch_size = 20
init_training_rate=0.0001
#momentum_multiplier=0.5
momentum_multiplier=0.25

#Learning rate decay
decay_steps=6000
decay_rate=0.8


#Dataset manipulation parameters
shuffle_buffer_size = 1000

#Data augmentation parameters
augmentation_enabled = True
max_radian_tilt = 45.0 * (3.1415926 / 180.0)
max_brightness_shift = 0.1
#Minimum scale for scale augmentation -- for 480x480, with a 256x256 central region, ~.7552 is recommended
#(if the maximum degree tilt is up to 45 degrees)
min_scale = 0.7552
max_scale = 1.1
#Contrast shifts are multipliers
min_contrast_shift = 0.9
max_contrast_shift = 1.1

#HEATMAP GENERATION PARAMETERS

#The list of per-part scaling factors (for generating Gaussians)
#The resulting tensor is of shape [num_parts]
part_scaling_factors = [1.0, 1.0, 1.0, 1.0, 1.0
            ,1.0, 1.0, 1.0, 1.0, 1.0
            ,1.0, 1.0, 1.0, 1.0, 1.0]
#Gets the scaling factors (s, t) for determining the standard deviation of the
#Gaussians to draw. If the part scaling factor is p, and the z position for the part
#is given, then the Gaussian's standard deviation will be:
#p / (s * z + t)
scale_factors = (0.0002, 0.0)


#Performance parameters
CPU_PARALLEL_THREADS = 8

#DEBUGGING PARAMETERS
debug_input_pipeline = False
debug = True
rand_counter = 0
rand_debug_trip = 100


#IMAGE PARAMETERS

#Image width/image height is 256 by 256 for our purposes here
img_width = 256
img_height = 256

#Number of color channels on the full-color image
img_color_channels = 3

#Size of the _color_ image vector
img_vec_size = img_width * img_height * img_color_channels

#Size of the _grayscale_ image vector (From edge-detected version)
gray_img_vec_size = img_width * img_height

#Directory for a CMU-like dataset of folders of training sequences
TRAINING_DATASET_DIR = '/media/ramdisk/CMUPanopticTraining/PanOptic/panoptic-toolbox/'

#Directory for a CMU-like dataset of folders of validation sequences
VALIDATION_DATASET_DIR = '/media/ramdisk/CMUPanopticValidation/PanOptic/panoptic-toolbox/'


#Half-intensity and full-intensity constants for 16-bit images
MAX_INTENSITY = 1.0
HALF_INTENSITY = MAX_INTENSITY / 2.0
