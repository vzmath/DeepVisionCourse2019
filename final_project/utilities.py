import torch
import torch.nn as nn
from models import LRCN

# take a list of binary numbers (0/1), returns 1 if the number of 1's
# are over the threshold, otherwise returns 0
def patient_level_label(labels, threshold):
    label_str = ''.join([str(label) for label in labels])
    target_str = '1'*threshold
    patient_label = 1 if target_str in label_str else 0
    return patient_label
