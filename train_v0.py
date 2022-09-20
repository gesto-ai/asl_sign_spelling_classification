import torch
from torch import nn
import matplotlib.pyplot as plt
import timm

import cv2


import pytorch_lightning as pl
import glob
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.profiler import SimpleProfiler


from torchmetrics.functional import accuracy
from pytorch_lightning.callbacks import ModelCheckpoint 

import numpy as np


model_checkpoint = ModelCheckpoint(monitor = "val_loss",
                                   verbose=True,
                                   filename="{epoch}_{val_loss:.4f}")


label_df = []
all_folds = glob.glob("data/asl_alphabet_train/asl_alphabet_train/*");
for i in all_folds:
    label_ = i.split("/")[-1]
    all_imgs = glob.glob(f"{i}/*.jpg")
    for inner_img in all_imgs:
        label_df.append((label_, inner_img))
df = pd.DataFrame(label_df, columns =['label_', 'img_path']); df.head(2)



class PL_resnet50(pl.LightningModule):
    def __init__(self,lr=0.001, classes=30):
        super(PL_resnet50, self).__init__()
        self.lr = lr
        self.classes = classes
        self.create_resnet50_model()
        self.model = self.create_resnet50_model()

     
    def create_resnet50_model(self):
        self.model =  timm.create_model("mobilenetv3_small_050",
                        pretrained=True,
                        features_only=False,
                        num_classes = self.classes)

        self.num_in_features = self.model.get_classifier().in_features

        self.model.fc = nn.Sequential(
                            nn.BatchNorm1d(self.num_in_features),
                            nn.Linear(in_features=self.num_in_features, out_features= 640, bias = False),
                            nn.ReLU(),
                            nn.BatchNorm1d(640),
                            nn.Dropout(0.4),
                            nn.Linear(in_features=640, out_features = 320, bias = False),
                            nn.ReLU(),
                            nn.BatchNorm1d(320),
                            nn.Linear(in_features = 320, out_features=self.classes, bias = False)
                        )
        return self.model

    def forward(self, x):
        output = self.model(x)
        return output

    def training_step(self, batch, batch_idx):
        images, target = batch
        preds = self.forward(images)
        loss = F.cross_entropy(preds, target)
        acc = accuracy(preds,target)
        self.log("train_acc",acc,on_step=False,on_epoch=True,prog_bar=True,logger=True),
        self.log("train_loss",loss,on_step=False,on_epoch=True,prog_bar=True,logger=True)
        return loss

    
    def validation_step(self, batch, batch_idx):
        images, target = batch
        preds = self.forward(images)
        loss = F.cross_entropy(preds, target)

        acc = accuracy(preds,target)
        self.log("val_acc",acc,prog_bar=True,logger=True),
        self.log("val_loss",loss,prog_bar=True,logger=True)


    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        print(avg_loss)
        self.log("avg_loss",avg_loss,prog_bar=True,logger=True)
        
    def test_step(self, batch, batch_idx):
        images, target = batch
        preds = self.forward(images)
        return {'test_loss': F.cross_entropy(preds, target)}

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        return [optimizer], [scheduler]


class ASLDataset(Dataset):
    def __init__(self, df, transforms = None):
        self.df = df.copy()
        self.transforms = transforms

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        image = np.asarray(cv2.imread((self.df.img_path[index])))
        label = self.df.category_[index]
        if self.transforms is not None:
            image = self.transforms(image)
        return image, label


data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        
    ])

class ASLDataModule(pl.LightningDataModule):
    def __init__(self,  all_df, dataset_definition, batch_size = 32):
        self.batch_size = batch_size
        self.all_df = all_df
        self.prepare_data_per_node = False
        self._log_hyperparams = False
        self.dataset_definition = dataset_definition

    def prepare_data(self):
        self.all_df.sample(frac=1, random_state=42)

        
    def setup(self, stage = None):
        self.prepare_data()
        train, validate, test = \
                np.split(self.all_df.sample(frac=1, random_state=42), 
                        [int(.6*len(self.all_df)), int(.8*len(self.all_df))])
        self.train_dataset =  self.dataset_definition(train.reset_index(), transforms = data_transform)
        self.val_dataset = self.dataset_definition(validate.reset_index(), transforms = data_transform)
        self.test_dataset = self.dataset_definition(test.reset_index(), transforms = data_transform)

    def train_dataloader(self):
        trainset_dataloader = torch.utils.data.DataLoader(
                                    self.train_dataset,
                                    batch_size=self.batch_size,
                                    shuffle = True,
                                    num_workers = 3
                                    )
        return trainset_dataloader

    def val_dataloader(self):
        valset_dataloader = torch.utils.data.DataLoader(
                                    self.val_dataset,
                                    batch_size=self.batch_size,
                                    shuffle = False,
                                    num_workers = 3
                                    )
        return valset_dataloader

    def test_dataloader(self):
        testset_dataloader = torch.utils.data.DataLoader(
                                    self.test_dataset,
                                    batch_size=self.batch_size,
                                    shuffle = False,
                                    num_workers = 3
                                    )
        return testset_dataloader
        




if __name__ == "__main__":
    label_df = []
    all_folds = glob.glob("data/asl_alphabet_train/asl_alphabet_train/*")
    for i in all_folds:
        label_ = i.split("/")[-1]
        all_imgs = glob.glob(f"{i}/*.jpg")
        for inner_img in all_imgs:
            label_df.append((label_, inner_img))
    df = pd.DataFrame(label_df, columns =['label_', 'img_path'])
    mapping_dict = {}
    counter = 1
    for x in sorted(df.label_.unique()):
        mapping_dict[x] = counter
        counter +=1
    df['category_'] = df.label_.map(mapping_dict)


    model = PL_resnet50(lr=0.001,classes=30)
    dm = ASLDataModule(df, ASLDataset)
    trainer = Trainer(max_epochs=22, 
                accelerator="gpu", devices = 1)
    trainer.fit(model, dm)