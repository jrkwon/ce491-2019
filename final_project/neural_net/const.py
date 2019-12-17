###############################################################################
# constant definition

# config file name
# - If you want to change it, make your own copy like test_case1.yaml
# - 'name' will be postfixed to the neural network weight file name
CONFIG_YAML          = 'config_jr_lstm'   #'config_default'

# network model type
NET_TYPE_NIKHIL      = 0
NET_TYPE_MIR         = 1
NET_TYPE_NVIDIA      = 2
NET_TYPE_SQUEEZE     = 3
NET_TYPE_LSTM_FC6    = 4
NET_TYPE_LSTM_FC7    = 5
NET_TYPE_RESNET      = 6
NET_TYPE_CE491       = 7
NET_TYPE_JAEROCK     = 8
NET_TYPE_JR_LSTM     = 9

# file extension
DATA_EXT             = '.csv'
IMAGE_EXT            = '.jpg'
LOG_EXT              = '_log.csv'