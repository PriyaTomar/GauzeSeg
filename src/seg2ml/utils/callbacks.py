import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class EarlyStoppingAfterWarmup(EarlyStopping):
    def __init__(self, warmup_epochs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.warmup_epochs = warmup_epochs

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch < self.warmup_epochs:
            self.wait_count = 0
            if trainer.is_global_zero:
                print(f"Skipping early stopping check. Warm-up phase (epoch {trainer.current_epoch}) in progress.")
            return  
        super().on_validation_end(trainer, pl_module)


class MetricSmoothingCallback(pl.Callback):
    def __init__(self, monitor='valid/DiceScore', window_size=5):
        super().__init__()
        self.monitor = monitor  
        self.smoothing_window_size = window_size
        self.val_metric_history = []  
    
    def on_validation_epoch_end(self, trainer, pl_module):
        val_metric = trainer.callback_metrics.get(self.monitor)

        if val_metric is not None:
            self.val_metric_history.append(val_metric.item())

            # Apply smoothing moving average
            smoothed_val_metric = self.smooth_metrics(self.val_metric_history, self.smoothing_window_size)
            pl_module.log(f'smoothed_{self.monitor}', smoothed_val_metric, prog_bar=True, sync_dist= True)

    def smooth_metrics(self, metric_history, window_size):
        if len(metric_history) < window_size:
            # If not enough history, return the current metric
            return metric_history[-1]
        else:
            # Calculate moving average of the last `window_size` metrics
            smoothed_value = np.mean(metric_history[-window_size:])
            return smoothed_value

class ReduceLROnPlateauCallback(pl.Callback):
    def __init__(self, monitor, mode='min', factor=0.1, patience=5, verbose=True, min_lr=0.00001):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.min_lr = min_lr
        self.scheduler = None

    def on_fit_start(self, trainer, pl_module):
        #Called when fit begins to initialize the scheduler
        optimizer = trainer.optimizers[0]  # Assuming you have 1 optimizer
        self.scheduler = ReduceLROnPlateau(
            optimizer,
            mode=self.mode,
            factor=self.factor,
            patience=self.patience,
            verbose=self.verbose,
            min_lr=self.min_lr
        )

    def on_validation_end(self, trainer, pl_module):
        #Called at the end of the validation step to step the scheduler"""
        monitor_val = trainer.callback_metrics.get(self.monitor)
        if monitor_val is not None:
            self.scheduler.step(monitor_val)

    def on_train_epoch_start(self, trainer, pl_module):
        optimizer = trainer.optimizers[0]  
        current_lr = optimizer.param_groups[0]['lr']
        pl_module.log('learning_rate', current_lr)

        
class TrainLoaderSwitchCallback(pl.Callback):
    def __init__(self, warmup_epochs):
        self.warmup_epochs = warmup_epochs

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == self.warmup_epochs:
            print("switching to main training phase")
            trainer.datamodule.switch_training(phase="main")

    
    