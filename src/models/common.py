"""Pieces shared by the LightningModules of the three models."""


def build_optimizers(parameters, optimizer, scheduler=None):
    """Return value for configure_optimizers() from the Hydra partials of the model config.

    optimizer is a functools.partial (YAML with _partial_: true) that only lacks the parameters.
    scheduler is optional, also partial, and only lacks the optimizer. It is stepped once per epoch.
    """
    opt = optimizer(params=parameters)
    if scheduler is None:  # constant learning rate
        return opt
    return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler(opt), "interval": "epoch"}}
