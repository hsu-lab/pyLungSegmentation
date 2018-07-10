import time

import nrrd
import numpy as np
import pandas as pd
from sklearn.svm import SVC

from utils.correction_utils import *


def concatenate_layers(data_3d, data):
    try:
        return np.dstack((data_3d, data))
    except:
        return data


# Variables
working_dir = ['../data_example']
output_path = '../example_output.nrrd'

im_correct = None
options = None

# Train classifier
training = pd.read_csv('../SVM_Lung_Training.csv')
training_array = np.asarray(training)
X = training_array[:, 0:3]
y = training_array[:, 3]
clf = SVC(degree=3)
clf.fit(X, y)

# Read DICOM files and run lung segmentation algorithm
for w in np.arange(0, len(working_dir), 1):
    start = time.time()
    print(working_dir[w])
    nodule_data = {}
    im_raw, im_corrected, slice_order = read_and_correct_dicom(working_dir[w], clf)
    im_correct = concatenate_layers(im_correct, im_corrected)
    end = time.time()
    print("Time Elapsed: " + str(end - start) + " seconds")

# Save NRRD file (use ITK Snap to open)
nrrd.write(output_path, im_correct)
