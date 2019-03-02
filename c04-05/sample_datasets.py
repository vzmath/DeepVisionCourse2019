from torchvision.datasets import ImageFolder

class SampleDataset(ImageFolder):
    def __getitem__(self, index):
        sample, label = super(SampleDataset, self).__getitem__(index)
        path = self.imgs[index][0]
        return sample, label, path
