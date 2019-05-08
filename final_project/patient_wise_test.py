import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models, transforms
from torch.utils.data.sampler import WeightedRandomSampler
import copy, os, time, sys
from tqdm import tqdm
from dataset import CTA, MIPDataset, TestMIPDatasetWrapper
from utils import patient_level_label, load_trained_model

# parameters for generating MIPs
neg_patient_dir = '/home/vince/programming/python/research/rail/CTA-project/data/source/patient_wise_test_data/negative'
pos_patient_dir = '/home/vince/programming/python/research/rail/CTA-project/data/source/patient_wise_test_data/positive'
excel_path = '/home/vince/programming/python/research/rail/CTA-project/data/processed/test_data_annotations.xlsx'
neg_sheet = 'control'
pos_sheet = 'occlusion'
mips_axis = 'y'
THICKNESS = 20
TRAINED_MODEL = 'model_coronal_mips.pt'

BATCH_SIZE = 16
NUM_WORKERS = 10
NUM_CLASSES = 2

# threshold for number of positive MIPs
threshold = 5

data_transforms = {'test' : transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor()])}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# build a reference dictionary for ground-truth label look up
# the dictionary must be the form {patient_id : patient_level_label}
neg_ref_dict = {filename : 0 for filename in os.listdir(neg_patient_dir)}
pos_ref_dict = {filename : 1 for filename in os.listdir(pos_patient_dir)}
ref_dict = {**neg_ref_dict, **pos_ref_dict}
# then for each patient, generate MIPs along the target axis,
# use the trained model to perform the test and gather results

# check the number of positive MIPs as well as their spatial overlaps
# generate patient wise predictions based upon the comparison
correct_preds = 0
correct_pos_preds = 0
correct_neg_preds = 0
for filename, label in tqdm(ref_dict.items()):
    if label == 0:
        filepath = os.path.join(neg_patient_dir, filename)
        patient = CTA(filepath)
        bounding_box = patient.get_head_bounding_box(excel_path, neg_sheet)
    else:
        filepath = os.path.join(pos_patient_dir, filename)
        patient = CTA(filepath)
        bounding_box = patient.get_head_bounding_box(excel_path, pos_sheet)
    patient_mips = patient.generate_patient_mips(THICKNESS, bounding_box, mips_axis)
    patient_mips_dataset = TestMIPDatasetWrapper(patient_mips, data_transforms['test'])
    patient_mips_dataloader = torch.utils.data.DataLoader(patient_mips_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # load model
    trained_model = load_trained_model(TRAINED_MODEL, NUM_CLASSES, device)
    mips_preds = dict()
    with torch.no_grad():
        for samples, infos in patient_mips_dataloader:
            samples = samples.to(device)
            outputs = trained_model(samples)
            _, preds = torch.max(outputs.data, 1)
            start_idxs, end_idxs, _, _, _, _ = infos
            for start_idx, end_idx, pred in zip(start_idxs, end_idxs, preds):
                mip_range = ( start_idx.item(), end_idx.item() )
                mips_preds[mip_range] = pred.item()
    pred_labels = [pred for pred in mips_preds.values()]
    patient_label = patient_level_label(pred_labels, threshold)
    if ref_dict[patient.get_filename()] == patient_label:
        correct_preds += 1
        if label == 0:
            correct_neg_preds += 1
        else:
            correct_pos_preds += 1

# compute the accuracy using the predictions and look-up dictionary
print('There are {} patients in the test set, in which {} are control and {} are with occlusion'
      .format(len(ref_dict), len(neg_ref_dict), len(pos_ref_dict)))
print('The classifier made {} total correct predictions, in which {} are true negative and {} are true positive'
      .format(correct_preds, correct_neg_preds, correct_pos_preds))
