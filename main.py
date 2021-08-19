import os
import pickle
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import SimpleITK as sitk
os.environ["PATH"] = "/opt/homebrew/Caskroom/slicer/4.11.20210226,1442768/Slicer.app/Contents/MacOS:" + os.environ["PATH"]
os.environ["SITK_SHOW_COMMAND"] = "Slicer"

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay

import torch
from torchvision.transforms import Compose
from torchio.transforms import CropOrPad
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.core import datamodule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from data import LIDCIDRIDataset, Register, RegisterUsingLungSegmentation, ResampleIsotropic, SITKImageToTensor

# Super basic CNN using PyTorch
class LungCNN(nn.Module):
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

# A deeper CNN architecture
class LungCNN2(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Sequential( 
             nn.Conv3d(in_channels=1, out_channels=1, kernel_size=5, stride=5), 
             nn.ReLU(),
        )

        conv_deep_layers = []
        for _ in range(15):
            conv_deep_layers.append(nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3))
            conv_deep_layers.append(nn.ReLU())
        self.conv_deep = nn.Sequential(*conv_deep_layers)

        self.flatten = nn.Sequential(
            nn.Flatten(),
            nn.Flatten()
        )
        self.dense = nn.Sequential(
            nn.Linear(in_features=21*21*21, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=num_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):         # inputs: 256x256x256
        out = self.conv1(x)       # -> 51x51x51
        out = self.conv_deep(out) # -> 21x21x21
        out = self.flatten(out)   # flatten to 21*21*21
        out = self.dense(out)     # -> num_classes (2)
        return out

class LungMalignancyClassifier(pl.LightningModule):
    def __init__(self, model="LungCNN"):
        super().__init__()
        if model == "LungCNN":
            self.model = LungCNN(num_classes=2)
        else:
            assert model == "LungCNN2", "Unknown model specified!"
            self.model = LungCNN2(num_classes=2)

    def forward(self, x):
        return self.model(x)

    def _process_batch(self, batch):
        """Extracts samples and labels from batch"""
        x, y, _ = batch
        x = x.float()
        y = F.one_hot(y, num_classes=2).float()
        return x, y

    def training_step(self, batch, batch_idx):
        x, y = self._process_batch(batch)
        preds = self(x)
        loss = F.binary_cross_entropy(preds, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self._process_batch(batch)
        preds = self(x)
        loss = F.binary_cross_entropy(preds, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = self._process_batch(batch)
        preds = self(x)
        loss = F.binary_cross_entropy(preds, y)
        self.log("test_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return preds, y
        
    def test_epoch_end(self, outputs):
        outs = self.all_gather(outputs)
        print("outs:",outs)
        preds = outs[0][0].squeeze(0)
        y = outs[0][1].squeeze(0)

        if self.trainer.is_global_zero:
            y = y.cpu()
            preds = preds.cpu()
            print("Preds:", preds)
            print("y:", y)

            print("Compute ROC curve and AUC")
            y_labels = torch.argmax(y, dim=-1)
            pos_preds = preds[:,-1]
            print("y_labels:", y_labels)
            print("pos_preds:", pos_preds)

            tpr, fpr, _ = roc_curve(y_labels, pos_preds)
            roc_auc = roc_auc_score(y, preds)
            plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
            plt.legend(loc = 'lower right')
            plt.plot([0, 1], [0, 1],'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.savefig("test_roc_curve.png", dpi=400, transparent=True)
            plt.show()

            self.log("test_auc", roc_auc, rank_zero_only=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=0.0001)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, threshold=0.0001)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "train_loss"}


class LIDCDataModule(pl.LightningDataModule):
    def __init__(self, path, batch_size=64, num_workers=1):
        super().__init__()
        self.path = Path(path)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.imgs_path = self.path / "imgs.pt"
        self.labels_path = self.path / "labels.pt"
        self.metadatas_path = self.path / "metadatas.pickle"

    def prepare_data(self):
        if not self.imgs_path.exists() and self.labels_path.exists() and self.metadatas_path.exists():
            print("Loading data...")
            # preprocess everything and save all to disk
            dataset = LIDCIDRIDataset(path=self.path)
            ref_img = dataset._load_image("01-01-2000-CT THORAX WCONTRAST-56900")
            translate = sitk.TranslationTransform(3)
            translate.SetParameters((0, 40, 0)) # slight adjustment to put it closer to the center
            ref_img = sitk.Resample(ref_img, translate)

            dataset.transform = Compose([
                    Register(ref_img),
                    #RegisterUsingLungSegmentation(ref_img), # slow, but more accurate
                    ResampleIsotropic(),
                    SITKImageToTensor(),
                    CropOrPad(target_shape=(256, 256, 256))  # Crop the center 256x256x256 voxels
                ])
            
            print("- Preprocessing everything to disk...")
            self.imgs = []
            self.labels = []
            self.metadatas = []
            for img, label, metadata in tqdm(dataset):
                self.imgs.append(img)
                self.labels.append(label)
                self.metadatas.append(metadata)
            self.imgs = torch.stack(self.imgs)
            self.labels = torch.tensor(np.array(self.labels))

            torch.save(self.imgs, str(self.imgs_path))
            torch.save(self.labels, str(self.labels_path))
            with open(self.metadatas_path, 'wb') as f:
                pickle.dump(self.metadatas, f)

    def setup(self, stage=None):
        self.imgs = torch.load(self.imgs_path)
        self.labels = torch.load(self.labels_path)
        with open(self.metadatas_path, 'rb') as f:
            self.metadatas = pickle.load(f)

        if self.trainer.is_global_zero:
            print("Filtering and preprocessing labels...")
        # remove samples with label 0 (unknown)
        mask = self.labels != 0
        self.imgs = self.imgs[mask]
        self.labels = self.labels[mask]
        self.metadatas = [x for (x, b) in zip(self.metadatas, mask) if b] # does same as above, but for lists

        # make label 1 (benign or non-malignant disease) into negative label
        self.labels[self.labels == 1] = 0
        # join label 2 (malignant primary lung cancer) and 3 (malignant metastatic)
        self.labels[self.labels == 2] = 1
        self.labels[self.labels == 3] = 1

        self.data = list(zip(self.imgs, self.labels, self.metadatas))

        train_val, self.test = train_test_split(self.data, train_size=0.8)
        self.train, self.val = train_test_split(train_val, train_size=0.8)
        if self.trainer.is_global_zero:
            print("Total no. of samples: ", len(self.data))
            print("Train samples:", len(self.train))
            print("Validation samples:", len(self.val))
            print("Test samples:", len(self.test))

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=len(self.test), num_workers=self.num_workers)

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--lidc-idri-dir", type=str, default="LIDC-IDRI")
    parser.add_argument("--batch-size", type=int, default=5)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--max-epochs", type=int, default=500000)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    datamodule = LIDCDataModule(path=args.lidc_idri_dir, batch_size=args.batch_size, num_workers=args.num_workers)
    classifier = LungMalignancyClassifier(model="LungCNN2")
    tb_logger = pl.loggers.TensorBoardLogger("logs/")
    
    trainer = pl.Trainer.from_argparse_args(args,
                                            #callbacks=[EarlyStopping(monitor="val_loss", patience=10)],
                                            log_every_n_steps=4,
                                            logger=tb_logger) # num_processes=X, gpus=Y, ...
                         
    trainer.fit(classifier, datamodule)

    trainer.test(classifier, datamodule)
