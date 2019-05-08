import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models, transforms
from torch.utils.data.sampler import WeightedRandomSampler
import copy, os, time, sys
from dataset import MIPDataset, MIPDatasetWrapper

# parameters for generating MIPs
neg_test_case = '../data/processed/test/control'
pos_test_case = '../data/processed/test/occlusion'
excel_path = '/home/vince/programming/python/research/rail/CTA-project/data/processed/head_indices.xlsx'
control_sheet = 'control'
occl_sheet = 'occlusion'
mips_axis = 'x'
THICKNESS = 20
TRAINED_MODEL = 'model_sagittal_mips.pt'

BATCH_SIZE = 16
SHUFFLE = False
NUM_WORKERS = 10
NUM_CLASSES = 2
data_transforms = {'test' : transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor()])}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('Start generating MIPs for test...')
neg_mips_testset = MIPDataset(neg_test_case).get_mips_dataset((excel_path, control_sheet), THICKNESS, False, mips_axis)
pos_mips_testset = MIPDataset(pos_test_case).get_mips_dataset((excel_path, occl_sheet), THICKNESS, True, mips_axis)
mips_testset = {**neg_mips_testset, **pos_mips_testset}     # only applicable in Python 3.5 or later
print('Complete generating MIPs for test...')
print()

# print out the basic info regarding the MIPs dataset
neg_mips_count = 0
pos_mips_count = 0
for _, label in mips_testset.values():
    if label == 0:
        neg_mips_count += 1
    elif label == 1:
        pos_mips_count += 1
    else:
        print('labeling error...')
        sys.exit(1)
print('There are {} MIPs in the dataset, in which {} are negative MIPs and {} are positive MIPs.'
      .format(len(mips_testset), neg_mips_count, pos_mips_count))
print()

# load the trained model and perform tests
test_dataset = MIPDatasetWrapper(mips_testset, data_transforms['test'])
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

trained_model = models.resnet152()
trained_model.avgpool = nn.AdaptiveAvgPool2d(1)
num_features = trained_model.fc.in_features
trained_model.fc = nn.Linear(num_features, NUM_CLASSES)
trained_model.load_state_dict(torch.load(TRAINED_MODEL))
trained_model.eval()
trained_model.to(device)

correct = 0
total = 0
correct_neg_preds = 0
correct_pos_preds = 0
total_neg = 0
total_pos = 0
with torch.no_grad():
    print('The test has started...')
    for samples, labels, _ in test_dataloader:
        samples = samples.to(device)                  # (batch_size, channels, width, height)
        labels = labels.to(device)                  # (batch_size, label)
        outputs = trained_model(samples)             # (batch_size, num_of_classes)
        _, preds = torch.max(outputs.data, 1)       # _ is the tensor of max probabilities for each sample in the batch
        total += labels.size(0)                     # size(0) returns the number of samples in each batch
        correct += (preds == labels).sum().item()   # numpy array
        for label in labels:
            if label == 1:
                total_pos += 1
            else:
                total_neg += 1
        for pred, label in zip(preds, labels):
            if pred == 1 and pred == label:
                correct_pos_preds += 1
            elif pred == 0 and pred == label:
                correct_neg_preds += 1
    print('Total negative MIPs: {}. Total positive MIPs: {}'.format(total_neg, total_pos))
    print('Total correct negative preds: {}. Total correct positive preds: {}'.format(correct_neg_preds, correct_pos_preds))
    print('Accuracy of the network: {:.4f}%'.format(100 * correct / total))
