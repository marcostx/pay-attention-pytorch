import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import glob

class makeDataset(Dataset):
    def __init__(self, dataset, labels, spatial_transform, seqLen=20):
        self.spatial_transform = spatial_transform
        self.images = dataset
        self.labels = labels
        self.seqLen = seqLen
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        vid_name = self.images[idx]
        label = self.labels[idx]
        
        video_figures = glob.glob(vid_name + "/*.jpg")


        inpSeq = []
        self.spatial_transform.randomize_parameters()
        for vid in video_figures:
            img = Image.open(vid)
            inpSeq.append(self.spatial_transform(img.convert('RGB')))
        
        inpSeq = torch.stack(inpSeq, 0)
        return inpSeq, label
