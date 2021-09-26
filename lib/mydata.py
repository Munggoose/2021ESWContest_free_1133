from torch.utils.data import DataLoader, Dataset

class NutsDataset(Dataset):
    def __init__(self):
        if train:
            self.split = 'train'
            self._split2 = ['normal']
        else:
            self.split = 'test'
            self._split2 = ['normal', 'abnormal']

        for target_split in self._split2:
            