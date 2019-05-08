import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models, transforms
from torch.utils.data.sampler import WeightedRandomSampler
import copy, os, time, sys
from tqdm import tqdm
from dataset import CTA, MIPDataset
from dataset_wrapper import TestMIPDatasetWrapper, NeighborMIPDatasetWrapper
from utils import patient_level_label, load_trained_model

# parameters for generating MIPs
neg_patient_dir = '/home/vince/programming/python/research/rail/CTA-project/data/source/patient_wise_test_data/negative'
pos_patient_dir = '/home/vince/programming/python/research/rail/CTA-project/data/source/patient_wise_test_data/positive'
excel_path = '/home/vince/programming/python/research/rail/CTA-project/data/processed/test_data_annotations.xlsx'
neg_sheet = 'control'
pos_sheet = 'occlusion'
mips_axis = 'z'
num_neighbors = 3
THICKNESS = 20
TRAINED_MODEL = 'model_axial_mips.pt'

BATCH_SIZE = 16
NUM_WORKERS = 10
NUM_CLASSES = 2

# threshold for number of positive MIPs
threshold = 15

data_transforms = {'test' : transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor()])}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# build a reference dictionary for ground-truth label look up
# the dictionary must be the form {patient_id : patient_level_label}
neg_ref_dict = {filename : 0 for filename in os.listdir(neg_patient_dir)}
pos_ref_dict = {filename : 1 for filename in os.listdir(pos_patient_dir)}
ref_dict = {**neg_ref_dict, **pos_ref_dict}


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
    elif label == 1:
        filepath = os.path.join(pos_patient_dir, filename)
        patient = CTA(filepath)
        bounding_box = patient.get_head_bounding_box(excel_path, pos_sheet)
    else:
        print('Invalid label...')
        sys.exit(1)
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

    # neighbor mips evaluation
    for (start_idx, end_idx), pred in mips_preds.items():
        if pred == 1:
            neighbor_mips = patient.get_neighbor_mips(start_idx, end_idx, mips_axis, bounding_box, num_neighbors)
            neighbor_mips_dataset = NeighborMIPDatasetWrapper(neighbor_mips, data_transforms['test'])
            neighbor_mips_dataloader = torch.utils.data.DataLoader(neighbor_mips_dataset, batch_size=len(neighbor_mips_dataset), num_workers=NUM_WORKERS)

            neighbor_mips_preds = list()
            with torch.no_grad():
                for neighbor_mips in neighbor_mips_dataloader:
                    neighbor_mips = neighbor_mips.to(device)
                    results = trained_model(neighbor_mips)
                    _, predics = torch.max(results, 1)
                    for predic in predics:
                        neighbor_mips_preds.append(predic.item())
            neighbor_mips_preds_str = ''.join([str(neighbor_mips_pred) for neighbor_mips_pred in neighbor_mips_preds])
            target_str = '1'*len(neighbor_mips_preds)
            if not target_str in neighbor_mips_preds_str:
                mips_preds[(start_idx, end_idx)] = 0

    # patient level prediction
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
