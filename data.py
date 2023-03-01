import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, ConcatDataset
#from torchvision.io import read_image
import torchvision.transforms as transforms
import os
import torch
import cv2
from copy import deepcopy

img_dir = '/scratch/ovis-cls/images'
anno_dir = '/scratch/ovis-cls/annos'

ovis_categories = [
        'person', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 
        'giraffe', 'poultry', 'giant_panda', 'lizard',
        'parrot', 'monkey', 'rabbit', 'tiger', 'fish',
        'turtle', 'bicycle', 'motorcycle', 'airplane'#, 'boat', #'vehicle'
    ]

train_pipeline = transforms.Compose([transforms.Resize([448,448]),
                                    transforms.RandomResizedCrop([224,224]),
                                    transforms.RandomHorizontalFlip(0.5),
                                    transforms.Normalize(mean=[109.975, 117.167, 122.481], std=[34.966, 31.324, 34.713])])#,
                                    #transforms.ToTensor()])

val_pipeline   = transforms.Compose([transforms.Resize([224,224]),
                                    transforms.Normalize(mean=[109.975, 117.167, 122.481], std=[34.966, 31.324, 34.713])])#,
                                    #transforms.ToTensor()])

def read_image(path, torchcvt=True):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 3:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    if torchcvt: 
        img = np.transpose(img, (2,0,1))
        img = torch.Tensor(img)
    return img

def torchcvt(img):
    img = np.transpose(img, (2,0,1))
    return torch.Tensor(img)

def get_annos(mode: str, occ_level: int):
    '''
    Get annotations, including filenames and labels, for each occlusion level.
    mode (str): train or val
    occ_level (int): occlusion level
    '''
    dat = pd.read_csv(os.path.join(anno_dir, '%s%i.txt' % (mode, occ_level)), delimiter=' ', names=['file', 'label'])
    return dat

class OVISDataset(Dataset):
    def __init__(self, mode, occ_level, itrans=None, ltrans=None):
        super().__init__()
        self.annos = get_annos(mode, occ_level)
        if itrans is None: 
            self.itrans = train_pipeline if mode == 'train' else val_pipeline
        else:
            self.itrans = itrans
        self.ltrans = ltrans

    def __len__(self):
        return len(self.annos)
    
    def __getitem__(self, index):
        img_path = os.path.join(img_dir, self.annos.file.iloc[index])
        img = read_image(img_path)
        if self.itrans: img = self.itrans(img)
        lbl = self.annos.label.iloc[index]
        if self.ltrans: lbl = self.ltrans(lbl)
        return img, lbl
    
def get_dataloader(mode: str, occ_level, batch_size=64, shuffle=True, itrans=None, ltrans=None):
    if type(occ_level) == list:
        ds = {}
        for olidx in occ_level:
            dsidx = OVISDataset(mode, olidx, itrans, ltrans)
            ds['occ_%i' % olidx] = deepcopy(dsidx)
            del dsidx
        print('DataLoader initialized for %s partition at occlusion level(s): %s' % (mode, ''.join([str(o) for o in occ_level])))
        return [DataLoader(ds['occ_%i' % ol], batch_size=batch_size, shuffle=shuffle) for ol in occ_level]
    else: 
        ds = OVISDataset(mode, occ_level, itrans, ltrans)
        occ_level = [occ_level]
        print('DataLoader initialized for %s partition at occlusion level(s): %s' % (mode, ''.join([str(o) for o in occ_level])))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

class dv:
    def __init__(self, dir='deep_features'):
        print('Loading deep feature vectors from deep_features/')
        self.feats = []
        for f in os.listdir(dir):
            self.feats.append(torch.load(os.path.join(dir, f)))
    
    def __call__(self, bs):
        batchft = []
        for _ in range(bs):
            idx = np.random.randint(len(self.feats))
            batchft.append(self.feats[idx])
        return torch.cat(batchft, dim=0)

def occlude(image, occluder, mode='random'):
    # occluders have 4 dimensions: take [:,;,3] to figure out where to mask
    # imgs are 224x224; occluders have long dimension of 112
    image = cv2.resize(image, dsize=(224,224))
    occluder = cv2.resize(occluder, dsize=(112,112))
    if mode == 'center':
        occluder = np.pad(occluder, ((56,56),(56,56),(0,0)))
    else:
        randxy = np.random.randint(-55,55+1, size=2)
        occluder = np.pad(occluder, ((56-randxy[0],56+randxy[0]),(56-randxy[1],56+randxy[1]),(0,0)))
    mask3d = np.stack([occluder[:,:,3] for _ in range(3)], axis=-1)
    occluder = occluder[:,:,:3]
    out = np.where(mask3d, occluder, image)
    cv2.imwrite('test_images/out.png', out)
    return out