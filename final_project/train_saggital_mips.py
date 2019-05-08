import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data.sampler import WeightedRandomSampler
import time
import os
import copy
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit

from models import initialize_model
from dataset import MIPDataset, MIPDatasetWrapper
from utils import train_val_split
from PIL import Image

# parameters for computing MIPs
neg_cases = '/home/vince/programming/python/research/rail/CTA-project/data/processed/control'
pos_cases = '/home/vince/programming/python/research/rail/CTA-project/data/processed/occlusion'
excel_path = '/home/vince/programming/python/research/rail/CTA-project/data/processed/train_val_data_annotations.xlsx'
control_sheet = 'control'
occl_sheet = 'occlusion'
mips_axis = 'x'

# hyper-parameters for computing MIPs and generating MIPs dataset
THICKNESS = 20
VAL_RATIO = 0.2

NUM_EPOCHS = 20
BATCH_SIZE = 20
SHUFFLE = True
NUM_WORKERS = 10
MODEL = 'resnet152'
TRAINED_MODEL = 'model_sagittal_mips.pt'
NUM_CLASSES = 2
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-5
STEP_SIZE = 7
GAMMA = 0.1

def train_model(device, model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_hist = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in tqdm(range(num_epochs)):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-'*10)

        # each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            if phase == 'train':
                scheduler.step()
                model.train()       # set model to training mode
            else:
                model.eval()        # set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels, _ in dataloaders[phase]:    # since I am using my own dataset class, _ is signature
                #inputs = inputs.float()
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward pass
                # track history only if in train phase
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'validation':
                val_acc_hist.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_hist

def main():
    # set up data transforms and device
    data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        #transforms.Lambda( lambda x : add_noise(x, 'speckle') ),
        #transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'validation': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # build MIPs dataset from negative and positive CTAs
    print('Start generating MIPs from the control patient group...')
    neg_mips_dataset = MIPDataset(neg_cases).get_mips_dataset((excel_path, control_sheet), THICKNESS, False, mips_axis)
    print('Complete generating MIPs from the control patient group...')
    print('Start generating MIPs from the occlusion patient group...')
    pos_mips_dataset = MIPDataset(pos_cases).get_mips_dataset((excel_path, occl_sheet), THICKNESS, True, mips_axis)
    print('Complete generating MIPs from the occlusion patient group...')
    mips_dataset = {**neg_mips_dataset, **pos_mips_dataset}     # only applicable in Python 3.5 or later
    print('MIPs dataset has been successfully generated...')

    # print basic information regarding the generated MIPs distribution
    neg_mips_count = 0
    pos_mips_count = 0
    for _, label in mips_dataset.values():
        if label == 0:
            neg_mips_count += 1
        elif label == 1:
            pos_mips_count += 1
        else:
            print('labeling error...')
            sys.exit(1)
    print()
    print('There are {} MIPs in the dataset, in which {} are negative MIPs and {} are positive MIPs.'
          .format(len(mips_dataset), neg_mips_count, pos_mips_count))

    # split the mips dataset into train and validation sets, set up datasets and corresponding dataloaders
    train_mips_dict, val_mips_dict = train_val_split(mips_dataset, VAL_RATIO)
    datasets = {'train' :      MIPDatasetWrapper(train_mips_dict, data_transforms['train']),
                'validation' : MIPDatasetWrapper(val_mips_dict, data_transforms['validation'])}
    train_zero_count, train_one_count = datasets['train'].get_label_counts()
    val_zero_count, val_one_count = datasets['validation'].get_label_counts()
    print('There are {} total MIPs in the training set, in which {} are negative MIPs and are {} positive MIPs'
          .format(len(datasets['train']), train_zero_count, train_one_count))
    print('There are {} total MIPs in the validation set, in which {} are negative MIPs and are {} positive MIPs'
          .format(len(datasets['validation']), val_zero_count, val_one_count))
    print()

    # create weighted random sampler for oversampling in train loader
    class_sample_count = [neg_mips_count, pos_mips_count]
    weights = 1 / torch.Tensor(class_sample_count)
    weights = weights.double()
    train_sampler = WeightedRandomSampler(weights=weights, num_samples=len(datasets['train']), replacement=True)

    dataloaders = {'train' :      torch.utils.data.DataLoader(datasets['train'], batch_size=BATCH_SIZE,
                                                              num_workers=NUM_WORKERS, shuffle=SHUFFLE),
                                                              #sampler=train_sampler),
                   'validation' : torch.utils.data.DataLoader(datasets['validation'], batch_size=BATCH_SIZE,
                                                              num_workers=NUM_WORKERS, shuffle=SHUFFLE)}

    # set up the model for training
    model_ft, params_to_optimize, _ = initialize_model(model_name=MODEL, num_classes=NUM_CLASSES, feature_extract=False, use_pretrained=True)
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params_to_optimize, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    # train and save model
    print('Model training starts...')
    trained_model, val_accs = train_model(device, model_ft, dataloaders, criterion, optimizer, exp_lr_scheduler, num_epochs=NUM_EPOCHS)
    print(val_accs)
    torch.save(trained_model.state_dict(), TRAINED_MODEL)
    print('Model training completes...')
    print()

if __name__ == "__main__":
    main()
