import os

import matplotlib.pyplot as plt
import SimpleITK as sitk
os.environ["PATH"] = "/opt/homebrew/Caskroom/slicer/4.11.20210226,1442768/Slicer.app/Contents/MacOS:" + os.environ["PATH"]
os.environ["SITK_SHOW_COMMAND"] = "Slicer"

import torch
from torchvision.transforms import Compose
from torchio.transforms import CropOrPad
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl

from data import LIDCIDRIDataset, Register, RegisterUsingLungSegmentation, ResampleIsotropic, SITKImageToTensor

dataset = LIDCIDRIDataset(path="/Users/mariehane/Desktop/LIDC-IDRI")
ref_img = dataset._load_image("01-01-2000-CT THORAX WCONTRAST-56900")
translate = sitk.TranslationTransform(3)
translate.SetParameters((0, 40, 0)) # slight adjustment to put it closer to the center
ref_img = sitk.Resample(ref_img, translate)

dataset.transform = Compose([
        Register(ref_img),
        #RegisterUsingLungSegmentation(ref_img), # slow
        ResampleIsotropic(),
        SITKImageToTensor(),
        CropOrPad(target_shape=(256, 256, 256))  # Crop the center 256x256x256 voxels
    ])

# Super basic CNN using PyTorch lightning
class LungCNN(pl.LightningModule):
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Sequential( 
             nn.Conv3d(in_channels=1, out_channels=1, kernel_size=5, stride=5), 
             nn.ReLU(),
        )
        self.maxpool1 = nn.MaxPool3d(kernel_size=20, stride=1)
        self.conv2 = nn.Sequential(
             nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, dilation=3, stride=3), 
             nn.ReLU(),
        )
        self.maxpool2 = nn.MaxPool3d(kernel_size=6, stride=1)
        self.flatten = nn.Sequential(
            nn.Flatten(),
            nn.Flatten()
        )
        self.dense = nn.Sequential(
            nn.Linear(in_features=64, out_features=num_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):        # inputs: 256x256x256
        out = self.conv1(x)      # -> 51x51x51
        out = self.maxpool1(out) # -> 32x32x32
        out = self.conv2(out)    # -> 9x9x9
        out = self.maxpool2(out) # -> 4x4x4
        out = self.flatten(out)  # flatten to 64
        out = self.dense(out)    # -> num_classes (2)
        return out

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        x = x.float()
        y = F.one_hot(y, num_classes=self.num_classes).float()
        preds = self(x)
        loss = F.binary_cross_entropy(preds, y)
        # Logging to TensorBoard
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


#train_set, val_set = torch.utils.data.random_split(dataset, [125, 32])
#train_loader = DataLoader(train_set)
loader = DataLoader(dataset)
classifier = LungCNN(num_classes=4)
trainer = pl.Trainer() # num_processes=X, gpus=Y, ...
trainer.fit(classifier, loader)