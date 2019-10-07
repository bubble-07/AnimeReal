#NETWORK HYPERPARAMETERS
#The number of features in each layer (important hyperparameter)
F = 32
#The number of intermediate scale-convolutional layers in the network
L = 24 #16 #12
#The number of layers per densely connected block
B = 6 #8 #6

#The number of features that we downsample to from an initial stride-2
#convolution from 256x256 to 128x128
feats_128 = 32

#The number of features that we downsample to from an initial stride-2
#convolution from 512x512 to 256x256
feats_256 = 8

#Features at the 512x512 size, only used for output layers
feats_512 = 6

#HEATMAP GENERATOR NETWORK TRAINING HYPERPARAMETERS
num_training_iters = 50000
batch_size = 5 #5 #10 #10 #10 #10 #5 #10 #10 #20
init_training_rate=0.0001
#momentum_multiplier=0.5
momentum_multiplier=0.25

#Learning rate decay
decay_steps=6000
decay_rate=0.8

#Augmentation parameters
AUGMENT_FLIP = True

AUGMENT_SMALL_ANGLE = True
AUGMENT_SMALL_ANGLE_MAX_MM = 25.4

AUGMENT_SMALL_SCALE = True
AUGMENT_SMALL_SCALE_MAX_DELTA_Z = 50.0

AUGMENT_GAUSS_NOISE = True
AUGMENT_GAUSS_NOISE_STDEV = 1.0

AUGMENT_OMIT_NOISE = True
AUGMENT_OMIT_PROB = 0.001

AUGMENT_TRANSLATE = True
AUGMENT_TRANSLATE_X_PIX = 50
AUGMENT_TRANSLATE_Y_PIX = 50

AUGMENT_ROTATE = True
AUGMENT_ROTATE_ANGLE_STDEV = 3.14159 * 2.0 * (5.0 / 360.0)

AUGMENT_UNIFORM_SCALE = True #Uniform scaling (in the sense of "the whole world gets larger/smaller, isotropically)
AUGMENT_MIN_UNIFORM_SCALE = 0.8
AUGMENT_MAX_UNIFORM_SCALE = 1.2

AUGMENT_ASPECT_RATIO = True #Non-isotropic x and y scaling
AUGMENT_ASPECT_X_MIN = 0.8
AUGMENT_ASPECT_X_MAX = 1.2
AUGMENT_ASPECT_Y_MIN = 0.8
AUGMENT_ASPECT_Y_MAX = 1.2


#Dataset manipulation parameters
shuffle_buffer_size = 1000

#Data augmentation parameters
augmentation_enabled = True

#HEATMAP GENERATION PARAMETERS

#The list of per-part scaling factors (for generating Gaussians)
#The resulting tensor is of shape [num_parts]
part_scaling_factors = [1.0, 1.0, 1.0, 1.0, 1.0
            ,1.0, 1.0, 1.0, 1.0, 1.0
            ,1.0, 1.0, 1.0, 1.0, 1.0]

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
img_color_channels = 1

#Size of the _color_ image vector
img_vec_size = img_width * img_height * img_color_channels

#Size of the _grayscale_ image vector (From edge-detected version)
gray_img_vec_size = img_width * img_height

#Half-intensity and full-intensity constants for 16-bit images
MAX_INTENSITY = 1.0
HALF_INTENSITY = MAX_INTENSITY / 2.0
