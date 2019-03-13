import sys
import deeplabcut
""" analyze video on Sherlock (~100 FPS on 200x200 cropped video on GPU)
NOTE: should already have created a DLC project and labeled training set
	**train and evaluate network before using it to analyze video**
	see: DeepLabCut/examples/Demo_yourowndata.ipynb

Params:
-------
path_config_file : string
	path to the config.yaml file containing project params

"""

# import vars and check path
path_config_file = sys.argv[1]
videofile_path = sys.argv[2]
if not os.path.exists(path_config_file):
    raise ValueError('no config file at %s' %path_config_file)
if not os.path.exists(path_config_file):
    raise ValueError('no video file at %s' %videofile_path)

# analyze video and save results to csv file
deeplabcut.analyze_videos(path_config_file, videofile_path, shuffle=1, save_as_csv=True, videotype='.mp4')