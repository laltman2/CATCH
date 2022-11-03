class EarlyStopping(object):
    '''Stops training if validation loss doesn't improve

    Adapted from
    https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py

    Properties
    ----------
    patience: int
        How long to wait after last time validation loss improved.
    delta: float
        Minimum change that qualifies as an improvement.
        Default: 0
    '''
    def __init__(self, patience=100, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > (self.best_loss - self.delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
