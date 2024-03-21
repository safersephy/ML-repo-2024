import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import mlflow
import numpy as np
import torch
import pandas as pd
from datetime import datetime,date
from lightning.pytorch import Trainer,LightningModule
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TemporalFusionTransformer, NHiTS  # Import other models as needed
from pytorch_forecasting.metrics import RMSE,QuantileLoss
from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer
from dataFunctions import getdata,createDatasetLightning,GetNaiveBaseline,smape
from lightningFunctions import hyperparameter_tuning_with_mlflow


np.random.seed(42)
samplesize = 50
startdate = date(2023,3,15)
max_prediction_length = 7


df, hierarchy_df = getdata(samplesize,startdate)

GetNaiveBaseline(df,max_prediction_length)

train_dataset, val_dataset = createDatasetLightning(df,False,max_prediction_length)

nh = NHiTS.load_from_checkpoint("../models/best_Optuna NHiTS 2024-03-21 11:26:03.ckpt")
val_dataloader = val_dataset.to_dataloader(train=False, batch_size=samplesize, num_workers=0)
nh_pred = nh.predict(val_dataloader,trainer_kwargs=dict(accelerator="cpu"), return_y=True)
print(RMSE()(nh_pred.output, nh_pred.y))

tft = TemporalFusionTransformer.load_from_checkpoint("../models/best_Optuna TFT 2024-03-21 12:10:10.ckpt")
val_dataloader = val_dataset.to_dataloader(train=False, batch_size=samplesize, num_workers=8)
tft_pred = tft.predict(val_dataloader,trainer_kwargs=dict(accelerator="cpu"), return_y=True, return_x=True,)
print(RMSE()(tft_pred.output, tft_pred.y))



import matplotlib.pyplot as plt


for sample in np.arange(0, samplesize-1):
    x = np.arange(0,len(tft_pred.x['encoder_target'][sample]))
    xpred = np.arange(len(tft_pred.x['encoder_target'][sample]),len(tft_pred.x['encoder_target'][sample]) + len(tft_pred.y[0][sample]) )


    smape_tft_score = smape(tft_pred.y[0][sample].numpy(), tft_pred.output[sample].numpy())
    smape_nh_score = smape(nh_pred.y[0][sample].numpy(), nh_pred.output[sample].numpy())

    title = (f"{val_dataset.categorical_encoders['groupId'].inverse_transform(tft_pred.x['encoder_cat'][sample][0][0].numpy())} ({val_dataset.categorical_encoders['staticCovariate1'].inverse_transform(tft_pred.x['encoder_cat'][sample][0][1].numpy())})")

    plt.plot(x,tft_pred.x['encoder_target'][sample])
    plt.plot(xpred,tft_pred.y[0][sample],c='black', label="gemeten")
    plt.plot(xpred,tft_pred.output[sample], label=f"prediction TFT ({np.round(100 - smape_tft_score)}%)")
    plt.plot(xpred,nh_pred.output[sample], label=f"prediction NHiTS ({np.round(100 - smape_nh_score)}%)")
    plt.title(title)
    plt.xlabel("prediction horizon")
    plt.ylabel("consumption")
    plt.legend()
    plt.show()
