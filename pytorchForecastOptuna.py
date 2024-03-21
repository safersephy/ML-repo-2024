import optuna
from optuna.integration import PyTorchLightningPruningCallback
import mlflow
import numpy as np
from datetime import datetime
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping,Callback
from pytorch_forecasting import TemporalFusionTransformer, NHiTS  # Import other models as needed
from pytorch_forecasting.metrics import RMSE,QuantileLoss
from lightning.pytorch.tuner import Tuner
from dataFunctions import get_or_create_experiment,champion_callback

class MLflowLoggingCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        # Log val_loss
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            mlflow.log_metric("val_loss", val_loss.item(), step=trainer.current_epoch)

def hyperparameter_tuning_with_mlflow(model_class, param_space, train_dataset, val_dataset, experiment_name,samplesize, num_trials=50, max_epochs=50):
    

    experiment_id = get_or_create_experiment(experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)

    def objective(trial):
        with mlflow.start_run(nested=True):
            # Dynamic hyperparameter suggestion based on param_space
            hyperparams = {}
            for param_name, param_config in param_space.items():
                if param_config["type"] == "categorical":
                    hyperparams[param_name] = trial.suggest_categorical(param_name, param_config["values"])
                elif param_config["type"] == "float":
                    hyperparams[param_name] = trial.suggest_float(param_name, param_config["low"], param_config["high"], log=param_config.get("log", False))
                elif param_config["type"] == "int":
                    hyperparams[param_name] = trial.suggest_int(param_name, param_config["low"], param_config["high"])

            mlflow.log_params(hyperparams)

            # Handling optimizer and loss from hyperparameters
            optimizer = hyperparams.pop("optimizer", "AdamW")  
            loss_function = hyperparams.pop("loss", QuantileLoss())  


            # Initialize the model
            model = model_class.from_dataset(
                train_dataset,
                **hyperparams,
                reduce_on_plateau_patience=5,
                reduce_on_plateau_reduction=10,
                reduce_on_plateau_min_lr=1e-7,
                weight_decay=1e-2,
                optimizer=optimizer,
                loss=loss_function
                )

            # Early stopping callback
            prunercb = PyTorchLightningPruningCallback(trial, monitor="val_loss")
            early_stopping = EarlyStopping(monitor="val_loss", patience=10, verbose=True)

            # Initialize the PyTorch Lightning trainer
            trainer = Trainer(
                max_epochs=max_epochs,
                accelerator="gpu",
                gradient_clip_val=0.1,                                    
                callbacks=[early_stopping,prunercb,MLflowLoggingCallback()],
                limit_train_batches=50,  # Adjust as needed
            )

            # Train the model
            trainer.fit(model, train_dataloaders=train_dataset.to_dataloader(train=True, batch_size=128), val_dataloaders=val_dataset.to_dataloader(train=False, batch_size=128 * 10))

            val_loss = trainer.callback_metrics["val_loss"].item()
            mlflow.log_metric("val_loss", val_loss)

            return val_loss

    pruner = optuna.pruners.HyperbandPruner()
    run_name = f'attempt {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True):    
        study = optuna.create_study(direction="minimize", pruner=pruner)
        study.optimize(objective, n_trials=num_trials, callbacks=[champion_callback])

        print("Best trial:")
        best_trial = study.best_trial
        for key, value in best_trial.params.items():
            print(f"{key}: {value}")

        mlflow.log_params(best_trial.params)
        mlflow.log_metric("best_val_loss", best_trial.value)

    # Train and save the best model
    best_params = best_trial.params

    optimizer = best_params.pop("optimizer", "AdamW")
    loss_function = best_params.pop("loss", QuantileLoss())
    
    best_model = model_class.from_dataset(
        train_dataset,
        **best_params, 
        reduce_on_plateau_patience=5,
        reduce_on_plateau_reduction=10,
        reduce_on_plateau_min_lr=1e-7,
        weight_decay=1e-2,
        optimizer=optimizer, 
        loss=loss_function)

    early_stopping = EarlyStopping(monitor="val_loss", patience=10, verbose=True)

    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",        
        gradient_clip_val=0.1,
        callbacks=[early_stopping,MLflowLoggingCallback()],        
        ) 
    
    with mlflow.start_run(experiment_id=experiment_id, run_name=f'{experiment_name} - Full') as run:
        trainer.fit(best_model, train_dataloaders=train_dataset.to_dataloader(train=True,batch_size=128,num_workers=8), val_dataloaders=val_dataset.to_dataloader(train=False, batch_size=128 * 10,num_workers=8))



        trainer.save_checkpoint(f"models/best_{experiment_name}.ckpt")

        # Log the best model in MLflow
        mlflow.pytorch.log_model(best_model, "best_model {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        val_dataloader = val_dataset.to_dataloader(train=False, batch_size=samplesize, num_workers=8)
        predictions = best_model.predict(val_dataloader, return_y=True)

        rmse = RMSE()(predictions.output, predictions.y)
        mlflow.log_metric("rmse", rmse)
        print(rmse)

    return best_model

def findlrLightning(trainer,model,train_dataloader,val_dataloader):
    res = Tuner(trainer).lr_find(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        max_lr=10.0,
        min_lr=1e-6,
    )
    
    print(f"suggested learning rate: {res.suggestion()}")
    fig = res.plot(show=True, suggest=True)
    fig.show()




