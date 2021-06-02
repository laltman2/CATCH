import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os


def training_progress(configuration='test', savedir='./results/'):
    fig, ax = plt.subplots()
    ax.set_xlabel('Epochs')
    ax.set_ylabel('log(loss)')

    basedir = os.path.dirname(os.path.abspath(__file__)).split('eval')[0]
    path = basedir + 'cfg_estimator/{}_train_info.csv'.format(configuration)
    df = pd.read_csv(path)
    print(df)

    ax.plot(df.epochs, np.log(df.train_loss), label='train')
    ax.plot(df.epochs, np.log(df.test_loss), label='test')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(savedir+'{}_training_progress.png'.format(configuration))
    plt.show()

    if 'z_loss' in df.columns:
        fig, ax = plt.subplots()
        ax.set_xlabel('Epochs')
        ax.set_ylabel('log(loss)')
        ax.plot(df.epochs, np.log(df.z_loss), label='z train', color='red')
        ax.plot(df.epochs, np.log(df.z_testloss), label='z test', color='darkred')
        ax.plot(df.epochs, np.log(df.a_loss), label='a train', color='blue')
        ax.plot(df.epochs, np.log(df.a_testloss), label='a test', color='darkblue')
        ax.plot(df.epochs, np.log(df.n_loss), label='n train', color='green')
        ax.plot(df.epochs, np.log(df.n_testloss), label='n test', color='darkgreen')
        ax.legend()
        ax.grid(alpha=0.3)
        fig.savefig(savedir+'{}_params_training_progress.png'.format(configuration))
        plt.show()

    
if __name__ == '__main__':
    training_progress('test')
