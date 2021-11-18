class TrainerConfig:
    # model config
    in_channels = 3
    out_channels = 19

    # optimization parameters
    max_epochs = 10
    max_epochs = 2000
    batch_size = 64
    # learning_rate = 3e-4
    learning_rate = 1e-3
    weight_decay = 0.1
    num_workers = 4

    # checkpoint settings
    ckpt_path = None

    # logging settings
    print_step = 10

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)