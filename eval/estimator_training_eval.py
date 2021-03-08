import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os


def training_progress(configuration='test'):
    fig, ax = plt.subplots()
    ax.set_xlabel('Epochs')
    ax.set_ylabel('log(loss)')

    basedir = os.path.dirname(os.path.abspath(__file__)).split('eval')[0]
    path = basedir + 'cfg_estimator/{}_train_info.csv'.format(configuration)
    df = pd.read_csv(path)

    ax.plot(df.epochs, np.log(df.train_loss), label='train')
    ax.plot(df.epochs, np.log(df.test_loss), label='test')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig('{}_training_progress.png'.format(configuration))
    plt.show()

if __name__ == '__main__':
    training_progress('scale_float')
