import numpy as np
from datetime import datetime,date
from pytorch_forecasting import TemporalFusionTransformer, NHiTS  # Import other models as needed
from pytorch_forecasting.metrics import RMSE,QuantileLoss
from dataFunctions import getdata,createDatasetLightning,GetNaiveBaseline
from lightningFunctions import hyperparameter_tuning_with_mlflow


np.random.seed(42)
samplesize = 50
startdate = date(2023,3,15)
max_prediction_length = 7


df, hierarchy_df = getdata(samplesize,startdate)

GetNaiveBaseline(df,max_prediction_length)

train_dataset, val_dataset = createDatasetLightning(df,False,max_prediction_length)

#-------------------------------------------------------------------------------------------------------------------------

model_class = NHiTS  # Replace with your model class
param_space = {
    "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    "hidden_size": {"type": "int", "low": 128, "high": 512},
    "n_blocks": {"type": "categorical", "values": [[1, 1, 1],[2, 2, 2], [3, 2, 2], [3, 3, 3]]}, 
    # Add other hyperparameters here
}
experiment_name = f"Optuna NHiTS {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"  # Replace with your experiment name

# Call the function with the model class, parameter space, and datasets
best_model = hyperparameter_tuning_with_mlflow(model_class, param_space, train_dataset, val_dataset, experiment_name,samplesize)

#-------------------------------------------------------------------------------------------------------------------------

model_class = TemporalFusionTransformer  # Replace with your model class
param_space = {
    "learning_rate": {"type": "float", "low": 1e-5, "high": 3e-2, "log": True},
    "hidden_size": {"type": "categorical", "values": [64, 128, 256, 512]},
    "attention_head_size": {"type": "int", "low": 2, "high": 4},
    "dropout": {"type": "categorical", "values": [0.1,0.2, 0.3]},
    # Add other hyperparameters here
}
experiment_name = f"Optuna TFT {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"  # Replace with your experiment name

# Call the function with the model class, parameter space, and datasets
best_model = hyperparameter_tuning_with_mlflow(model_class, param_space, train_dataset, val_dataset, experiment_name,samplesize)

#-------------------------------------------------------------------------------------------------------------------------
