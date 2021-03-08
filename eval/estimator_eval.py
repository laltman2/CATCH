import numpy as np
import json, cv2, os, ast
import pandas as pd
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

        results = est.predict(img_list = [img], scale_list = [params['scale']])[0]

        resultsdict = {'img_num':n, 'scale':params['scale'],
                       'z_pred':results['z_p'], 'a_pred': results['a_p'], 'n_pred':results['n_p'],
                       'z_true':params['z_p'], 'a_true': params['a_p'], 'n_true':params['n_p']}
        df = df.append(resultsdict, ignore_index=True)

    savepath = configuration + '_eval.csv'

    df.to_csv(savepath)
        


if __name__ == '__main__':
    estimator_accuracy(configuration='scale_float')
