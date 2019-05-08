import os
import torch
import torchvision.transforms as transforms

from dataset import MIPSeqDataset
from models import LRCN
from utilities import patient_level_label

# test data
test_dir = "/media/vince/FreeAgent Drive/test_mip_seqs"
pos_dirname = "occlusion"
neg_dirname = "control"

# model parameters
model_file = 'lrcn.pt'
base_cnn = "ResNet34"
pretrained = True
feature_extract = False
n_layers = 2
n_classes = 2
hidden_size = 512
bidirectional = False

# hyper parameters
batch_size = 8
num_workers = 4
threshold = 10

# create the patient level dirs and labels 
test_dirs = dict()      # {patient_seqs_dir : patient_label}
pos_dir = os.path.join(test_dir, pos_dirname)
neg_dir = os.path.join(test_dir, neg_dirname)
pos_count, neg_count = 0, 0
for patient_id in os.listdir(pos_dir):
    patient_seqs_dir = os.path.join(pos_dir, patient_id)
    test_dirs[patient_seqs_dir] = 1     # patient level label
    pos_count += 1
for patient_id in os.listdir(neg_dir):
    patient_seqs_dir = os.path.join(neg_dir, patient_id)
    test_dirs[patient_seqs_dir] = 0     # patient level label
    neg_count += 1

# data transforms and device configuration
data_transforms = {'test' : transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor()])}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

correct = 0
pos_correct = 0
neg_correct = 0
for patient_seqs_dir, patient_label in test_dirs.items():
    # current patient MIP sequences dataset and dataloader
    patient_seqs_dataset = MIPSeqDataset(
        patient_dir=patient_seqs_dir,
        transform=data_transforms['test'])
    patient_seqs_dataloader = torch.utils.data.DataLoader(
        dataset=patient_seqs_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False)
    # initialize and load trained model
    model = LRCN(hidden_dim=hidden_size,
                 n_layers=n_layers,
                 n_classes=n_classes,
                 bidirectional=bidirectional,
                 model=base_cnn,
                 use_pretrained=pretrained,
                 feature_extracting=feature_extract)
    model.load_state_dict(torch.load(model_file))
    model.eval()
    model.to(device)
    # MIP seqs predictions and use the threshold to obtain the patient-level prediction
    with torch.no_grad():
        pred_labels = list()
        for mip_seqs in patient_seqs_dataloader:
            mip_seqs = mip_seqs.to(device)
            outputs = model(mip_seqs)
            _, preds = torch.max(outputs.data, 1)
            for pred in preds:
                pred_labels.append(pred.item())
        patient_pred = patient_level_label(pred_labels, threshold)
        if patient_pred == patient_label:
            correct += 1
            if patient_label == 0:
                neg_correct += 1
            else:
                pos_correct += 1
print("There are {} positive patients and {} negative patients".format(pos_count, neg_count))
print("{} patients are correctly predicted, {} positive patients are correctly predicted and {} negative patients are correctly predicted"
      .format(correct, pos_correct, neg_correct))
