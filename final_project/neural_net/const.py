# version
VERSION_MAJOR        = 0
VERSION_MINOR        = 5

# network model type
NET_TYPE_NIKHIL      = 0
NET_TYPE_MIR         = 1
NET_TYPE_NVIDIA      = 2
NET_TYPE_SQUEEZE     = 3
NET_TYPE_LSTM_FC6    = 4
NET_TYPE_LSTM_FC7    = 5
NET_TYPE_RESNET      = 6
NET_TYPE_CE491       = 7

# file extension
DATA_EXT             = '.csv'
IMAGE_EXT            = '.jpg'

# training
VALID_RATE           = 0.3
NUM_EPOCH            = 20
BATCH_SIZE           = 16
NUM_OUTPUT           = 1

# steering angle adjustment
RAW_SCALE            = 1.0 
JITTER_TOLERANCE     = 0.009

# TODO: find the right image size and capture area for
#       for your image dataset and neural network architecture 

# input image size to the neural network
IMAGE_WIDTH          = 160
IMAGE_HEIGHT         = 70
IMAGE_DEPTH          = 3

# crop (capture) area from a camera image
CROP_X1              = 0
CROP_Y1              = 380
CROP_X2              = 800
CROP_Y2              = 590
