
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
