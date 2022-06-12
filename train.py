import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

from model import MRSClassfication
from dataloader import BrainDataModule
import json
import argparse
import yaml
import torch
import warnings

warnings.filterwarnings(action='ignore')

pl.seed_everything(42)


def train(model_name,model_hparams,data_dir,save_dir,epoch,accelerator,device,batch_size=32,num_workers=3,pin_memory=True):
    ## train 

    model = MRSClassfication(model_name,model_hparams)
    data_dm = BrainDataModule(data_dir,batch_size,num_workers,pin_memory)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=save_dir, save_top_k=1, monitor="val_auc",filename=f'{model_name}'+'-{epoch:02d}-{val_auc:.2f}')
    
    if device > 1:
        data_dm.prepare_data()
        trainer = pl.Trainer(accelerator=accelerator, devices=device, precision=16,max_epochs=epoch,callbacks=[checkpoint_callback],strategy="ddp")
    else :
        trainer = pl.Trainer(accelerator=accelerator, devices=device, precision=16,max_epochs=epoch,callbacks=[checkpoint_callback])

    ##모델학습 
    trainer.fit(model,data_dm)
    trainer.save_checkpoint(f"{model_name}.ckpt")

    ##test
    
    print(checkpoint_callback.best_model_path)
    data_dm.prepare_data()
    data_dm.setup()
    
    best_model = model.load_from_checkpoint(checkpoint_callback.best_model_path)
    
    trainer.test(best_model, dataloaders=data_dm.test_dataloader())

if __name__=="__main__" :

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_info", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--batch_size",type=int, default=16)
    parser.add_argument("--num_workers",type=int, default=3)
    parser.add_argument("--pin_memory",type=bool, default=True)
    parser.add_argument("--epoch",type=int, default=100)
    parser.add_argument("--accelerator",type=str, default='gpu')
    parser.add_argument("--device",type=int,default=1)

    args = parser.parse_args()


    with open(args.model_info) as f:
        model_info = yaml.load(f,Loader=yaml.FullLoader)
    

    train(model_name = model_info['model_name'],\
        model_hparams = model_info['model_hparams'],\
        data_dir = args.data_dir,\
        save_dir = args.save_dir,\
        epoch = args.epoch,\
        accelerator = args.accelerator,\
        device = args.device,\
        batch_size = args.batch_size,\
        num_workers = args.num_workers,\
        pin_memory = args.pin_memory)


