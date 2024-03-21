from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import polars as pl
import pandas as pd
import numpy as np
from datetime import datetime,date
import matplotlib.pyplot as plt
from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from lightning.pytorch.tuner import Tuner
from darts import TimeSeries
from darts.dataprocessing.transformers.static_covariates_transformer import StaticCovariatesTransformer 
from darts.dataprocessing.transformers import Scaler
from joblib import Parallel, delayed
from darts.models import NaiveSeasonal
from darts.metrics import mape,smape,rmse

def _backtests_local_estimator(_estimator, _ts_set, _horizons):
    model = _estimator
    model.fit(_ts_set)
    backtests_single_ts = model.predict(_horizons)
    return backtests_single_ts


def createdatasetDarts(df,samplesize = 378,max_prediction_length = 7):




    max_encoder_length = 45

    df['Date'] = pd.to_datetime(df['Date'])

    training_cutoff = df["dayindex"].max() - max_prediction_length

    df_train = df[lambda x: x.dayindex <= training_cutoff]

    ts_train = df_train[['groupId','Date','target']].copy()
    ts_val = df[['groupId','Date','target']].copy()

    ts_train = TimeSeries.from_group_dataframe(df=ts_train, 
                                                group_cols=['groupId'],
                                                time_col='Date', 
                                                value_cols='target',
                                                fill_missing_dates=True,
                                                freq='D',
                                                fillna_value=0
                                                )


    ts_val = TimeSeries.from_group_dataframe(df=ts_val, 
                                                group_cols=['groupId'],
                                                time_col='Date', 
                                                value_cols='target',
                                                fill_missing_dates=True,
                                                freq='D',
                                                fillna_value=0
                                                
                                                )



    transformer = StaticCovariatesTransformer()
    ts_train = transformer.fit_transform(ts_train)
    ts_val = transformer.transform(ts_val)




    scaler = Scaler() # MinMaxScaler
    ts_train_prepared = scaler.fit_transform(ts_train)
    ts_val_prepared = scaler.transform(ts_val)


    def get_overall_rmse(prediction_series, val_series):
        return np.round(np.mean(rmse(actual_series=val_series, 
                                    pred_series=prediction_series, n_jobs=-1)),
                        2)


    backtests_baseline_model = Parallel(n_jobs=-1,
                                        verbose=5, 
                                        backend = 'multiprocessing',
                                        pre_dispatch='1.5*n_jobs')(
            delayed(_backtests_local_estimator)(
                _estimator=NaiveSeasonal(K=7),
                _ts_set=single_ts_set,
                _horizons=max_prediction_length,
            )
        for single_ts_set in ts_train
    )
        
    print(f'overall baseline rmse: {get_overall_rmse(backtests_baseline_model, ts_val)}')

    sample = np.random.randint(0,samplesize,)
    fig, ax = plt.subplots(figsize=(30, 10))
    ts_val[sample].plot(label='True value', color='black')
    backtests_baseline_model[sample][:8].plot(label='Baseline', color='purple')
    plt.show()




    return ts_train_prepared, ts_val_prepared, backtests_baseline_model

def createDatasetLightning(df,varLengths = True,max_prediction_length = 7):
    min_prediction_length = 1
    max_encoder_length = 45
    min_encoder_length = max_encoder_length //2


    df['Date'] = pd.to_datetime(df['Date'])
    training_cutoff = df["dayindex"].max() - max_prediction_length
    
    
    # Define the TimeSeriesDataSet
    train_dataset_ts = TimeSeriesDataSet(
        df[lambda x: x.dayindex <= training_cutoff],
        time_idx='dayindex',
        target='target',
        group_ids=['groupId'],

        min_encoder_length= min_encoder_length if (varLengths) else max_encoder_length ,
        min_prediction_length=min_prediction_length if (varLengths) else max_prediction_length,
        max_encoder_length=max_encoder_length,  # this should be set according to your needs
        max_prediction_length=max_prediction_length,
        static_categoricals=['groupId','staticCategorical1'],
        time_varying_known_categoricals=[],
        time_varying_known_reals=['dayindex',"day","day_of_week", "month"],
        time_varying_unknown_reals=['target'],
        categorical_encoders={"staticCategorical1": NaNLabelEncoder(add_nan=True)}
        
        
        ,
        target_normalizer=GroupNormalizer(
            groups=['groupId'], transformation='softplus'  # use "softplus" and not "log" as it can handle zero values
        ),
        add_relative_time_idx=True if (varLengths) else False,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )
    
    val_dataset_ts = TimeSeriesDataSet.from_dataset(
        train_dataset_ts, df, predict=True, stop_randomization=True
    )
    


    return train_dataset_ts, val_dataset_ts

def GetNaiveBaseline(df,max_prediction_length):
    max_time_idx_per_series = df.groupby('groupId')['dayindex'].max()
    start_per_series = max_time_idx_per_series - (max_prediction_length * 2)
    mid_per_series = max_time_idx_per_series - (max_prediction_length)
    
    baseline_actuals_data = df[df.apply(lambda row: row['dayindex'] > start_per_series[row['groupId']] and 
                                                        row['dayindex'] <= mid_per_series[row['groupId']], axis=1)][['groupId','dayindex','target']]
    baseline_pred_data = df[df.apply(lambda row: row['dayindex'] > mid_per_series[row['groupId']], axis=1)][['groupId','dayindex','target']]
    
    #align dayindex with last week
    baseline_pred_data['dayindex'] = baseline_pred_data['dayindex'] - 7
    
    
    merged_df = pd.merge(baseline_pred_data, baseline_actuals_data, on=['groupId', 'dayindex'], suffixes=('_current', '_previous'))
    
    
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    
    rmse = sqrt(mean_squared_error(merged_df['target_previous'],merged_df['target_current']))
    print("Overall RMSE:", rmse)
    
    return rmse, merged_df



def get_or_create_experiment(experiment_name):
    """
    Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

    This function checks if an experiment with the given name exists within MLflow.
    If it does, the function returns its ID. If not, it creates a new experiment
    with the provided name and returns its ID.

    Parameters:
    - experiment_name (str): Name of the MLflow experiment.

    Returns:
    - str: ID of the existing or newly created MLflow experiment.
    """

    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)

def champion_callback(study, frozen_trial):
    """
    Logging callback that will report when a new trial iteration improves upon existing
    best trial values.

    Note: This callback is not intended for use in distributed computing systems such as Spark
    or Ray due to the micro-batch iterative implementation for distributing trials to a cluster's
    workers or agents.
    The race conditions with file system state management for distributed trials will render
    inconsistent values with this callback.
    """

    winner = study.user_attrs.get("winner", None)

    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        if winner:
            improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
            print(
                f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                f"{improvement_percent: .4f}% improvement"
            )
        else:
            print(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")

def smape(actual, forecast):
    """
    Calculate SMAPE between two arrays.

    Args:
    actual (numpy.array): Array of actual values.
    forecast (numpy.array): Array of forecasted values.

    Returns:
    float: SMAPE score.
    """
    # Convert inputs to numpy arrays (if they aren't already)
    actual, forecast = np.array(actual), np.array(forecast)
    
    # Avoid division by zero: add a small epsilon where (actual + forecast) is zero
    epsilon = np.finfo(np.float64).eps
    denominator = np.abs(actual) + np.abs(forecast) + epsilon

    # Calculate SMAPE
    smape_value = np.mean(2.0 * np.abs(forecast - actual) / denominator)
    
    return smape_value * 100  # Return as percentage
