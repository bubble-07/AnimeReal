#NETWORK HYPERPARAMETERS
#The number of features in each layer (important hyperparameter)
F = 64 #64
#The number of intermediate scale-convolutional layers in the network
L = 48 #24 #16 #18 #16 #18 #12
#The number of layers per densely connected block
B = 6 #8 #6 #4 #6

#The number of features that we downsample to from an initial stride-2
#convolution from 256x256 to 128x128
feats_128 = 32

#The number of features that we downsample to from an initial stride-2
#convolution from 512x512 to 256x256
feats_256 = 8

#Features at the 512x512 size, only used for output layers
feats_512 = 6

#HEATMAP GENERATOR NETWORK TRAINING HYPERPARAMETERS
num_training_iters = 120000 #2000 #120000 #10 #60000 #60000 #10 #99 #60000 #100 #60000 #100 #5000000 #50000
num_quantization_iters = 5000 #20000 #5 #20000 #20000
batch_size = 5 #5 #10 #10 #10 #5 #10 #10 #20
init_training_rate= 0.00001 #0.0001 #0.0005 #0.001
momentum_multiplier=0.5
#momentum_multiplier=0.25

#Learning rate decay
#decay_steps=6000
decay_steps=12000
decay_rate=0.8

#AUGMENTATION PARAMETERS

#Crop box randomization parameters for COCO
COCO_MIN_X_SPREAD = .25
COCO_MIN_Y_SPREAD = .25
COCO_MIN_ASPECT_RATIO = 9.0 / 16.0
COCO_MAX_ASPECT_RATIO = 16.0 / 9.0
COCO_MIN_X_MAG = 0.95
COCO_MAX_X_MAG = 2.0
COCO_MIN_Y_MAG = 0.95
COCO_MAX_Y_MAG = 2.0
#With this setting, the bounding box around any entity can be at worst 1/4 visible (if its in a corner)
COCO_MAX_X_SPREAD_SHIFT_FRAC = 0.5
COCO_MAX_Y_SPREAD_SHIFT_FRAC = 0.5

#Augmentation parameters for the person overlay (not necessarily the background)
AUGMENT_FLIP = True
AUGMENT_TRANSLATE = True
AUGMENT_TRANSLATE_X_PIX = 50
AUGMENT_TRANSLATE_Y_PIX = 50

AUGMENT_ROTATE = True
AUGMENT_ROTATE_ANGLE_STDEV = 3.14159 * 2.0 * (5.0 / 360.0)

AUGMENT_ASPECT_RATIO = True #Non-isotropic x and y scaling
AUGMENT_ASPECT_X_MIN = 0.8
AUGMENT_ASPECT_X_MAX = 1.2
AUGMENT_ASPECT_Y_MIN = 0.8
AUGMENT_ASPECT_Y_MAX = 1.2

AUGMENT_HUE_SHIFT = True
AUGMENT_HUE_STDEV = 0.1

AUGMENT_BRIGHTNESS_SHIFT = True
AUGMENT_MIN_BRIGHTNESS_DELTA = -0.2
AUGMENT_MAX_BRIGHTNESS_DELTA = 0

#Selection parameters for the background
SELECT_BACKGROUND_MIN_FRAC = .25 #Minimum fraction of the image for each dimension


#Augmentation parameters for the background
AUGMENT_BACKGROUND_HUE_SHIFT = True
AUGMENT_BACKGROUND_HUE_STDEV = 5.0
AUGMENT_BACKGROUND_BRIGHTNESS_SHIFT = True
AUGMENT_MIN_BACKGROUND_BRIGHTNESS_DELTA = 10
AUGMENT_MAX_BACKGROUND_BRIGHTNESS_DELTA = 0


#Augmentation parameters for the whole image
AUGMENT_GAUSS_NOISE = True
AUGMENT_GAUSS_NOISE_STDEV = 1.0

AUGMENT_GAUSS_SMOOTH = True
AUGMENT_GAUSS_SMOOTH_STDEV_MEAN = 2.0
AUGMENT_GAUSS_SMOOTH_STDEV_STDEV = 1.0

AUGMENT_GAMMA_SHIFT = True
AUGMENT_GAMMA_STDEV = 0.2

AUGMENT_SATURATION = True
AUGMENT_SATURATION_STDEV = 5.0



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
img_color_channels = 3

#Size of the _color_ image vector
img_vec_size = img_width * img_height * img_color_channels

#Size of the _grayscale_ image vector (From edge-detected version)
gray_img_vec_size = img_width * img_height

#Half-intensity and full-intensity constants for 16-bit images
MAX_INTENSITY = 1.0
HALF_INTENSITY = MAX_INTENSITY / 2.0
