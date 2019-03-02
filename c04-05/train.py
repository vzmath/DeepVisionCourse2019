import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
import time
import os
import copy

from sample_datasets import SampleDataset
from models import initialize_model

data_dir = '/home/vince/academics/Hopkins/Spring 2019/deep_vision/c04-05/code_review/data/hymenoptera_data'
model_path = '/home/vince/academics/Hopkins/Spring 2019/deep_vision/c04-05/code_review/model_wts'

# hyper-parameters
PHASES = ['train', 'val']
PRETRAINED = True
EXTRACTOR = True
model_name = 'vgg_feature_extractor_wts.pt'

NUM_EPOCHS = 40
BATCH_SIZE = 16
SHUFFLE = True
NUM_WORKERS = 4
MODEL = 'vgg19'           # vgg19, resnet152
NUM_CLASSES = 2
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
STEP_SIZE = 50
GAMMA = 0.1

def train_model(device, model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_hist = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-'*10)

        # each epoch has a training and validation phase
        for phase in PHASES:
            if phase == 'train':
                scheduler.step()
                model.train()       # set model to training mode
            else:
                model.eval()        # set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels, _ in dataloaders[phase]:    # since I am using my own dataset class, _ is paths
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
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
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
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image_datasets = {phase : SampleDataset(root=os.path.join(data_dir, phase), transform=data_transforms[phase])
                      for phase in PHASES}
    dataloaders = {phase : torch.utils.data.DataLoader(dataset=image_datasets[phase], batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS)
                   for phase in PHASES}

    if not os.path.exists(model_path):
        os.mkdir(model_path)

    # set up model
    model, params_to_optimize, _ = initialize_model(model_name=MODEL, num_classes=NUM_CLASSES, feature_extract=EXTRACTOR, use_pretrained=PRETRAINED)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()   # classification task, cross entropy loss function
    optimizer = optim.SGD(params_to_optimize, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)    # stochastic gradient descent optimizer
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    print('model training starts...')
    trained_model, _ = train_model(device, model, dataloaders, criterion, optimizer, exp_lr_scheduler, num_epochs=NUM_EPOCHS)
    torch.save(trained_model.state_dict(), os.path.join(model_path, model_name))
    print('model training completes...')

if __name__ == "__main__":
    main()
