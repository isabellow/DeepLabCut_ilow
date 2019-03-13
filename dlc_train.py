import sys
import deeplabcut

""" train deeplabcut on Sherlock (~6hrs on GPU for 200,000 iterations)
NOTE: should already have created a DLC project and labeled training set
	see: DeepLabCut/examples/Demo_yourowndata.ipynb

Params:
-------
path_config_file : string
	path to the config.yaml file containing project params

"""
# import vars and check path
path_config_file = sys.argv[1]
if not os.path.exists(path_config_file):
    raise ValueError('no config file at %s' %path_config_file)

# train the network
deeplabcut.train_network(path_config_file, gputouse=1, max_snapshots_to_keep=5)