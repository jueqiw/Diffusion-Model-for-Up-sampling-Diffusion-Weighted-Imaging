from torch.optim import lr_scheduler
import torch.nn.init as init
import math


def get_scheduler(optimizer, hyperparameters, iterations=-1):
    if "lr_policy" not in hyperparameters or hyperparameters["lr_policy"] == "constant":
        scheduler = None  # constant scheduler
    elif hyperparameters["lr_policy"] == "step":
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=hyperparameters["step_size"],
            gamma=hyperparameters["gamma"],
            last_epoch=iterations,
        )
    else:
        return NotImplementedError(
            "learning rate policy [%s] is not implemented", hyperparameters["lr_policy"]
        )
    return scheduler


def weights_init(init_type="gaussian"):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find("Conv") == 0 or classname.find("Linear") == 0) and hasattr(
            m, "weight"
        ):
            # print m.__class__.__name__
            if init_type == "gaussian":
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == "default":
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun
