import numpy as np
import glob
import sys
import random

def clean(num):
  destination_path = f"raw_data/youtube_keypoints_data/youtube_faces_with_keypoints_full_{num}"
  npzFilesFullPath = glob.glob(f"{destination_path}/*.npz")
  videoIDs = [x.split('/')[-1].split('.')[0] for x in npzFilesFullPath]
  fullPaths = {}
  for videoID, fullPath in zip(videoIDs, npzFilesFullPath):
    # print(fullPath)
    fullPaths[videoID] = fullPath

  # fullpaths - dict
  # key - name of npz file
  # value - path to file
  # sort the dictionary to get readable output - dict(sorted(fullPaths.items()))
  for key in dict(sorted(fullPaths.items())):
    print(key)
    my_file = np.load(fullPaths[key])
    np.savez_compressed(f"{destination_path}/{key}.npz", colorImages=my_file['colorImages'], landmarks2D=my_file['landmarks2D'])

if __name__ == '__main__':
  # python scripts/clean_data.py 1/2/3/4
  # number corresponds to folder containing keypoints
  # eg: python scripts/clean_data.py 1
  clean(sys.argv[1])
