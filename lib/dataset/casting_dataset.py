from PIL import Image
import os
import os.path
import numpy as np

from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import DataLoader, Dataset

from skimage import io

def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames if filename.endswith(suffix)]

class Castingdataset(Dataset):
    """[summary]

    Args:
        Dataset ([type]): [description]
    """
    def __init__(self,root:str , train:bool,transform = None) -> None:
        super(Castingdataset, self).__init__()
        self.root = root
        self.transform = transform
        self.train = train
        
        
        # if not self._check_integrity():
        #     raise RuntimeError('Dataset not found or corrupted.')
        
        self.files = {}
        self.data: Any = []
        self.targets = []

        if train:
            self.split = 'train'
            self._split2 = ['normal'] 
        else:
            self.split = 'test'
            self._split2 = ['normal','abnormal'] 
        
        for target_split in self._split2:
            # self.jpeg_base = os.path.join(self.root,self.split, target_split)
            self.jpeg_base = 'C:\\Users\\Seungchan_HCI\\OneDrive - inha.edu\\VC\\python\\HCI\\Nuts\\GANOMALY\\ganomaly-master\\data\\casting\\train\\normal'
            self.jpeg_base = 'C:\\Users\\Seungchan_HCI\\OneDrive - inha.edu\\VC\\python\\HCI\\Nuts\\GANOMALY\\ganomaly-master\\data\\casting\\train\\normal'
            #self.jpeg_base = 'C:\\Users\\Seungchan_HCI\\Desktop\\train\\480\\train\\normal'
            self.files[self.split] = recursive_glob(rootdir=self.jpeg_base, suffix= '.bmp')
            if not self.files[self.split]:
                raise Exception("No files for split=[%s] found in %s" % (self.split, self.jpeg_base))

            for jpeg_path in self.files[self.split]:
                _img = io.imread(jpeg_path)
                self.data.append(_img)
                if target_split == 'normal':
                    self.targets.extend([0])
                else:
                    self.targets.extend([1])
            
        self.data = np.vstack(self.data).reshape(-1, 3, 64, 64)
        self.data = self.data.transpose((0, 2, 3, 1))
        
    def __len__(self)->int:
        return len(self.data)
    
    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

        
        
