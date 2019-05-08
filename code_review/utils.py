import os
import sys
from scipy.io import loadmat

def mat_reader(file, key):
    if os.path.exists(file):
        mat_file = loadmat(file)
        param = mat_file[key]
        return param
    else:
        print('{} does not exist, please check and retry...'.format(file))
        sys.exit(1)
