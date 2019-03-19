import sys
import os
import deeplabcut

""" train deeplabcut on Sherlock (~6hrs on GPU for 200,000 iterations)
also creates training set again to ensure paths are updated for linux

NOTE: should already have created a DLC project and labeled training set
	see: DeepLabCut/examples/Demo_yourowndata.ipynb

NOTE: be sure to update paths in config and pose files
	(if project was created elsewhere)

Params:
-------
path_config_file : string
	path to the config.yaml file containing project params

"""

def run_program(path_config_file):
	# import vars and check path
	if not os.path.exists(path_config_file):
	    raise ValueError('no config file at %s' %path_config_file)

	# train the network
	deeplabcut.create_training_dataset(path_config_file, windows2linux=True)
	deeplabcut.train_network(path_config_file, gputouse=1, max_snapshots_to_keep=5)

if __name__ == "__main__":
	run_program(sys.argv[1])