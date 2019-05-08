from torch.utils.data import Dataset
from PIL import Image

#####################################################################################################################
# PyTorch dataset wrapper for train/validation MIPs datasets
class MIPDatasetWrapper(Dataset):
    def __init__(self, mips_dataset, transform=None, neighbor_mips=False):
        self.source_data = mips_dataset
        self.transform = transform
        self.neighbor_mips = neighbor_mips
    def __len__(self):
        return len(self.source_data)
    def __getitem__(self, index):
        #mips_info = [mip_info for mip_info in self.source_data.keys()]
        mips = [mip_numpy_array for mip_numpy_array, _ in self.source_data.values()]
        mips_labels = [mip_label for _, mip_label in self.source_data.values()]
        #mip_info = mips_info[index]
        mip_label = mips_labels[index]

        if self.neighbor_mips:
            mip = Image.fromarray(mips[index], mode='RGB')   # 3 channels containing left, current, and right mips
        else:
            mip = Image.fromarray(mips[index]).convert('RGB')   # turn 1 channel to 3 channels

        if self.transform:
            mip = self.transform(mip)
        return mip, mip_label
    def get_label_counts(self):
        zero_count = 0
        one_count = 0
        for _, mip_label in self.source_data.values():
            if mip_label == 0:
                zero_count += 1
            elif mip_label == 1:
                one_count += 1
        return zero_count, one_count

# PyTorch dataset wrapper for test MIPs datasets (patient-level test)
class TestMIPDatasetWrapper(Dataset):
    def __init__(self, patient_mips_dataset, transform=None):
        self.source_data = patient_mips_dataset
        self.transform = transform
    def __len__(self):
        return len(self.source_data)
    def __getitem__(self, index):
        mips_info = [mip_info for mip_info in self.source_data.keys()]
        mips = [mip_numpy_array for mip_numpy_array in self.source_data.values()]
        mip_info = mips_info[index]
        mip = Image.fromarray(mips[index]).convert('RGB')   # turn 1 channel to 3 channels
        if self.transform:
            mip = self.transform(mip)
        return mip, mip_info

class NeighborMIPDatasetWrapper(Dataset):
    def __init__(self, neighbor_mips, transform=None):
        self.neighbor_mips = neighbor_mips # a list of numpy arrays representing neighbor mips
        self.transform = transform
    def __len__(self):
        return len(self.neighbor_mips)
    def __getitem__(self, index):
        neighbor_mip = self.neighbor_mips[index]
        mip = Image.fromarray(neighbor_mip).convert('RGB')
        if self.transform:
            mip = self.transform(mip)
        return mip
#####################################################################################################################
