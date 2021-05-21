from CATCH.training.YOLO_data_generator import makedata
from CATCH.Localizerv5_pip import Localizer
from CATCH.Estimator_Torch import Estimator
from CATCH.CATCHobject import CATCH
from CATCH.utilities.mtd import feature_extent
from pylorenzmie.theory import Sphere
import json, os, cv2, ast
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

def catch_accuracy(loc='yolov5_test', est='test', nframes=None, version=None, plot=False, weights='best'):
    basedir = os.path.dirname(os.path.abspath(__file__)).split('eval')[0]
    path = basedir + 'cfg_yolov5/{}.json'.format(loc)

    with open(path, 'r') as f:
        config = json.load(f)
    
    file_header = os.path.abspath(config['directory'])
    eval_dir = file_header + '/eval'
    
    mtd_config = config.copy()
    mtd_config['directory'] = eval_dir
    if not nframes:
        nframes = config['nframes_eval']
        
    mtd_config['nframes'] = nframes
    mtd_config['particle']['nspheres'] = [1,1]
    #mtd_config['overwrite'] = True
    
    makedata(config = mtd_config)

    localizer = Localizer(configuration=loc, version=version)
    estimator = Estimator(configuration=est, weights=weights)
    catch = CATCH(localizer=localizer, estimator=estimator)

    imgpath_fmt = config['directory']+'/eval/images/image{}.png'
    parampath_fmt = config['directory']+'/eval/params/image{}.json'

    df = pd.DataFrame(columns = ['img_num', 'x_true', 'y_true', 'ext_true', 'num_detections', 'x_pred', 'y_pred', 'ext_pred', 'conf', 'z_pred', 'a_pred', 'n_pred', 'z_true', 'a_true', 'n_true'])
    style = dict(fill=False, linewidth=3, edgecolor='r')
    for n in range(nframes):
        img = cv2.imread(imgpath_fmt.format(str(n).zfill(4)))
        with open(parampath_fmt.format(str(n).zfill(4)), 'r') as f:
            params = json.load(f)[0]
        sphere = Sphere()
        sphere.loads(params)
        params = ast.literal_eval(params)
        ext = 2. * feature_extent(sphere, mtd_config)

        results = catch.analyze([img])

        if plot:
            fig, ax = plt.subplots()
            ax.imshow(img, cmap='gray')
            for feature in results:
                corner, w, h = feature['bbox']
                ax.add_patch(Rectangle(xy=corner, width=w, height=h, **style))
            plt.show()
            
        resultsdict = {'img_num':n, 'x_true':params['x_p'], 'y_true':params['y_p'],  'z_true':params['z_p'], 'a_true': params['a_p'], 'n_true':params['n_p'], 'ext_true':ext, 'num_detections':len(results), 'x_pred':None, 'y_pred':None, 'z_pred': None, 'a_pred': None, 'n_pred': None, 'ext_pred':None, 'conf':None, 'edge':None}

        if len(results) == 1:
            r = results.iloc[0]
            p_ext = max(r['bbox'][1:])
            resultsdict['x_pred'] = r['x_p']
            resultsdict['y_pred'] = r['y_p']
            resultsdict['a_pred'] = r['a_p']
            resultsdict['z_pred'] = r['z_p']
            resultsdict['n_pred'] = r['n_p']
            resultsdict['conf'] = r['conf']
            resultsdict['ext_pred'] = p_ext
            resultsdict['edge'] = r['edge']
            
        df = df.append(resultsdict, ignore_index=True)

    print(df)

    if weights=='best':
        wstr = ''
    else:
        wstr=weights
    saveheader = 'est_{}{}_loc_{}'.format(est, wstr, loc)
    if version:
        saveheader += 'v{}'.format(version)
    savepath = saveheader + '_eval.csv'
    df.to_csv(savepath)

    truepos = df[df.num_detections==1]
    falseneg = df[df.num_detections==0]
    falsepos = df[df.num_detections>1]
    numfalsepos = int(falsepos.num_detections.sum() - len(falsepos))
    numtruepos = len(truepos) + len(falsepos)
    numfalseneg = len(falseneg)

    print('{} true positive detections, {} false positive (additional) detections, {} false negative detections'.format(numtruepos, numfalsepos, numfalseneg))

    z_rmse = np.sqrt(((truepos.z_pred - truepos.z_true) **2).mean(axis=0))
    a_rmse = np.sqrt(((truepos.a_pred - truepos.a_true) **2).mean(axis=0))
    n_rmse = np.sqrt(((truepos.n_pred - truepos.n_true) **2).mean(axis=0))


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

    figsavepath = saveheader+'_eval.png'
    fig.savefig(figsavepath)
    
    plt.show()
        
if __name__ == '__main__':
    catch_accuracy(est='longnsmooth',nframes=5000, version=2)
