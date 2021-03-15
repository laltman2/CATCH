import numpy as np
import json, cv2, os, ast
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from CATCH.Estimator_Torch import Estimator
from CATCH.training.Torch_DataLoader import makedata_inner

def estimator_accuracy(configuration='test', nframes=None):
    basedir = os.path.dirname(os.path.abspath(__file__)).split('eval')[0]
    path = basedir + 'cfg_estimator/{}.json'.format(configuration)

    with open(path, 'r') as f:
        config = json.load(f)

    #make eval dataset
    makedata_inner(config, settype='eval', nframes=nframes)

    imgpath_fmt = config['directory']+'/eval/images/image{}.png'
    parampath_fmt = config['directory']+'/eval/params/image{}.json'
    
    est = Estimator(configuration = configuration)

    if not nframes:
        nframes = config['eval']['nframes']

    df = pd.DataFrame(columns = ['img_num', 'scale', 'z_pred', 'a_pred', 'n_pred', 'z_true', 'a_true', 'n_true'])
    for n in range(nframes):
        img = cv2.imread(imgpath_fmt.format(str(n).zfill(4)))
        with open(parampath_fmt.format(str(n).zfill(4)), 'r') as f:
            params = ast.literal_eval(json.load(f)[0])
        scale = params['scale']
        results = est.predict(images = [img])[0]

        resultsdict = {'img_num':n, 'scale':params['scale'],
                       'z_pred':results['z_p'], 'a_pred': results['a_p'], 'n_pred':results['n_p'],
                       'z_true':params['z_p'], 'a_true': params['a_p'], 'n_true':params['n_p']}
        df = df.append(resultsdict, ignore_index=True)

    print(df)
    
    savepath = configuration + '_eval.csv'
    df.to_csv(savepath)

    z_rmse = np.sqrt(((df.z_pred - df.z_true) **2).mean(axis=0))
    a_rmse = np.sqrt(((df.a_pred - df.a_true) **2).mean(axis=0))
    n_rmse = np.sqrt(((df.n_pred - df.n_true) **2).mean(axis=0))


    matplotlib.use('TkAgg')
    fig, axes = plt.subplots(1,3, figsize=(10,5))
    names = ['z_p', 'a_p', 'n_p']
    for ax, name in list(zip(axes, names)):
        ax.grid(alpha=0.3)
        ax.set_xlabel(r'True ${}$'.format(name))
        ax.set_ylabel(r'Pred ${}$'.format(name))
    ax1, ax2, ax3 = axes
    ax1.scatter(df.z_true, df.z_pred, c='b', alpha=0.4)
    ax1.plot(df.z_true, df.z_true, c='r')
    ax1.annotate('RMSE = {} px'.format('%.3f'%z_rmse), xy=(0.05, 0.95), xycoords='axes fraction')
    
    ax2.scatter(df.a_true, df.a_pred, c='b', alpha=0.4)
    ax2.plot(df.a_true, df.a_true, c='r')
    ax2.annotate(r'RMSE = {}$\mu m$ '.format('%.3f'%a_rmse), xy=(0.05, 0.95), xycoords='axes fraction')
    
    ax3.scatter(df.n_true, df.n_pred, c='b', alpha=0.4)
    ax3.plot(df.n_true, df.n_true, c='r')
    ax3.annotate('RMSE = {}'.format('%.3f'%n_rmse), xy=(0.05, 0.95), xycoords='axes fraction')
    fig.tight_layout()

    figsavepath = configuration+'_eval.png'
    fig.savefig(figsavepath)
    
    plt.show()
        


if __name__ == '__main__':
    estimator_accuracy(configuration='test')
