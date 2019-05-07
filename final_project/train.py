import os
import copy
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import lr_scheduler, SGD
from torchvision import transforms

from cta import CTAs, MIP_Seq_Dataset
from dataset import MIPSeqDataset
from create_dataset import train_val_split, train_val_counts
from models import LRCN

# dataset parameters
data_dir = "/media/vince/FreeAgent Drive/train_val_seqs"
pos_dir, neg_dir = "occlusion", "control"
val_ratio = 0.2
balanced = True
seed = 0

# model parameters
phases = ["train", "validation"]
base_cnn = "ResNet34"
pretrained = True
feature_extract = False
n_layers = 2
n_classes = 2
hidden_size = 512
bidirectional = False

# hyper parameters
n_epochs = 10
batch_size = 8
learning_rate = 1e-3
momentum = 0.9
weight_decay = 1e-5
step_size = 7
gamma = 0.1
num_workers = 4

def train(model, device, dataloaders, criterion, optimizer, scheduler, num_epochs=50):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0

    for epoch in tqdm(range(num_epochs)):
        print("Epoch {}/{}".format(epoch+1, num_epochs))
        print('-'*16)

        for phase in phases:
            if phase == "train":
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # forward pass
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs.data, 1)
                #print(preds, labels)
                # backward pass only if in train phase
                if phase == "train":
                    loss.backward()
                    optimizer.step()

                total_step = len(dataloaders[phase])
                if (i+1) % batch_size == 0:
                    print("{} epoch [{}/{}], step [{}/{}], loss: {:.4f}"
                          .format(phase, epoch+1, num_epochs, i+1, total_step, loss.item()))

                # compute statistics per batch
                running_loss += loss.item() * inputs.size(0)        # inputs.size(0) = batch_size
                running_corrects += torch.sum(preds == labels.data)

            # compute statistics per epoch
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            # update the best model in the validation phase of each epoch
            if phase == "validation" and epoch_acc > best_acc:
                best_epoch = epoch + 1
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best validation accuracy is {:.4f} and was attained at {} epoch".format(best_acc, best_epoch))
    # load best model weights and return the trained model
    model.load_state_dict(best_model_wts)
    return model

def main():
    # train/val split for precomputed MIP seqs
    train_seqs, val_seqs = train_val_split(
        root=data_dir,
        pos_dir=pos_dir,
        neg_dir=neg_dir,
        test_ratio=val_ratio,
        balanced=balanced,
        random_seed=seed)

    total_pos_seq, total_neg_seq, train_pos_seq, train_neg_seq, val_pos_seq, val_neg_seq = train_val_counts(train_seqs, val_seqs)
    print("The dataset has {} positive and {} negative MIP sequences.".format(total_pos_seq, total_neg_seq))
    print("Train set has {} positive and {} negative MIP sequences. Validation set has {} positive and {} negative MIP sequences."
          .format(train_pos_seq, train_neg_seq, val_pos_seq, val_neg_seq))

    # create data transforms, datasets, and dataloaders
    data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ]),
    'validation': transforms.Compose([
        transforms.Resize((224, 224)),
        #transforms.RandomCrop(224),
        transforms.ToTensor(),
    ])}
    train_dataset = MIPSeqDataset(
        seq_data=train_seqs,
        transform=data_transforms['train'])
    val_dataset = MIPSeqDataset(
        seq_data=val_seqs,
        transform=data_transforms['validation'])
    train_loader = torch.utils.data.DataLoader(
                    dataset=train_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=True)
    val_loader = torch.utils.data.DataLoader(
                    dataset=val_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=False)
    dataloaders = {"train" : train_loader, "validation" : val_loader}

    # device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LRCN(hidden_dim=hidden_size,
                 n_layers=n_layers,
                 n_classes=n_classes,
                 bidirectional=bidirectional,
                 model=base_cnn,
                 use_pretrained=pretrained,
                 feature_extracting=feature_extract).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # train and validation
    print("Model training starts...")
    best_model = train(model=model,
                       device=device,
                       dataloaders=dataloaders,
                       criterion=criterion,
                       optimizer=optimizer,
                       scheduler=exp_lr_scheduler,
                       num_epochs=n_epochs)
    torch.save(best_model.state_dict(), 'lrcn.pt')
    print("Model training completes...")

if __name__ == "__main__":
    main()
