import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import nibabel as nib
from create_dataset import train_val_split

class CTADataset(Dataset):
    def __init__(self, mip_seqs_dataset, test=False, transform=None):
        self.mip_seqs = mip_seqs_dataset
        self.test = test
        self.transform = transform
    def __len__(self):
        return len(self.mip_seqs)
    def __getitem__(self, index):
        seq_infos = [seq_info for seq_info in self.mip_seqs.keys()]
        if self.test:
            seqs = [seq for seq in self.mip_seqs.values()]
            mip_seq = [self.transform(Image.fromarray(mip, mode='RGB')) if self.transform else Image.fromarray(mip, mode='RGB')
                       for mip in seqs[index]]
            return torch.stack(mip_seq), seq_infos[index]
        else:
            seqs = [seq for seq, _ in self.mip_seqs.values()]
            mip_seq = [self.transform(Image.fromarray(mip, mode='RGB')) if self.transform else Image.fromarray(mip)
                       for mip in seqs[index]]
            labels = [label for _, label in self.mip_seqs.values()]
            return torch.stack(mip_seq), labels[index], seq_infos[index]

class MIPSeqDataset(Dataset):
    def __init__(self, seq_data, transform=None):
        self.seq_dataset = seq_data
        self.seq_paths = [seq_path for seq_path in seq_data.keys()]
        self.seq_labels = [seq_label for seq_label in seq_data.values()]
        self.transform = transform
    def __len__(self):
        return len(self.seq_dataset)
    def __getitem__(self, index):
        seq_path = self.seq_paths[index]
        seq_label = self.seq_labels[index]
        seq_len = len(os.listdir(seq_path))
        mip_seq = list()
        # extract prefix in the filename and sort the MIP sequences in order
        file, ext = os.listdir(seq_path)[0].split('.')
        prefix = '_'.join(file.split('_')[:-1])
        for seq_idx in np.arange(1, seq_len+1):
            mip_filename = "{}_{}.{}".format(prefix, seq_idx, ext)
            mip_path = os.path.join(seq_path, mip_filename)
            mip = nib.load(mip_path).get_fdata()
            if self.transform:
                mip = self.transform( Image.fromarray(mip, mode='RGB') )
            else:
                mip = Image.fromarray(mip, mode='RGB')
            mip_seq.append(mip)
        return torch.stack(mip_seq), seq_label

class MIPSeqTestset(Dataset):
    def __init__(self, patient_dir, transform=None):
        self.root = patient_dir
        self.seq_paths = [os.path.join(self.root, seq_dir) for seq_dir in os.listdir(self.root)]
    def __len__(self):
        return len(self.seq_paths)
    def __getitem__(self, index):
        seq_path = self.seq_paths[index]
        seq_len = len(os.listdir(seq_path))
        mip_seq = list()
        # extract prefix in the filename and sort the MIP sequences in order
        file, ext = os.listdir(seq_path)[0].split('.')
        prefix = '_'.join(file.split('_')[:-1])
        for seq_idx in np.arange(1, seq_len+1):
            mip_filename = "{}_{}.{}".format(prefix, seq_idx, ext)
            mip_path = os.path.join(seq_path, mip_filename)
            mip = nib.load(mip_path).get_fdata()
            if self.transform:
                mip = self.transform( Image.fromarray(mip, mode='RGB') )
            else:
                mip = Image.fromarray(mip, mode='RGB')
            mip_seq.append(mip)
        return torch.stack(mip_seq)

if __name__ == "__main__":
    root = "/media/vince/FreeAgent Drive/cta_mip_seqs"
    pos_dir = "occlusion"
    neg_dir = "control"
    test_ratio = 0.2
    balanced = True
    seed = 0

    train_data, val_data = train_val_split(root, pos_dir, neg_dir, test_ratio, balanced, seed)
    train_dataset = MIPSeqDataset(train_data)
    for item in train_dataset:
        print()
