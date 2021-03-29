from CATCH.training.YOLO_data_generator import makedata
from CATCH.Localizerv5_pip import Localizer
from CATCH.utilities.mtd import feature_extent
from pylorenzmie.theory import Sphere
import json, os, cv2, ast
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

def localizer_accuracy(configuration='yolov5_test', nframes=None, version=None, plot=False):
    basedir = os.path.dirname(os.path.abspath(__file__)).split('eval')[0]
    path = basedir + 'cfg_yolov5/{}.json'.format(configuration)

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

    localizer = Localizer(configuration=configuration, version=version)

    imgpath_fmt = config['directory']+'/eval/images/image{}.png'
    parampath_fmt = config['directory']+'/eval/params/image{}.json'

    df = pd.DataFrame(columns = ['img_num', 'x_true', 'y_true', 'ext_true', 'num_detections', 'x_pred', 'y_pred', 'ext_pred', 'conf'])
    style = dict(fill=False, linewidth=3, edgecolor='r')
    for n in range(nframes):
        img = cv2.imread(imgpath_fmt.format(str(n).zfill(4)))
        with open(parampath_fmt.format(str(n).zfill(4)), 'r') as f:
            params = json.load(f)[0]
        sphere = Sphere()
        sphere.loads(params)
        params = ast.literal_eval(params)
        ext = 2. * feature_extent(sphere, mtd_config)

        results = localizer.detect([img])[0]

        if plot:
            fig, ax = plt.subplots()
            ax.imshow(img, cmap='gray')
            for feature in results:
                corner, w, h = feature['bbox']
                ax.add_patch(Rectangle(xy=corner, width=w, height=h, **style))
            plt.show()
            
        resultsdict = {'img_num':n, 'x_true':params['x_p'], 'y_true':params['y_p'], 'ext_true':ext, 'num_detections':len(results), 'x_pred':None, 'y_pred':None, 'ext_pred':None, 'conf':None}

        if len(results) == 1 and not results[0]['edge']:
            r = results[0]
            p_ext = max(r['bbox'][1:])
            resultsdict['x_pred'] = r['x_p']
            resultsdict['y_pred'] = r['y_p']
            resultsdict['conf'] = r['conf']
            resultsdict['ext_pred'] = p_ext
            
        df = df.append(resultsdict, ignore_index=True)

    print(df)

    saveheader = configuration
    if version:
        saveheader += '_v{}'.format(version)
    savepath = saveheader + '_eval.csv'
    df.to_csv(savepath)

    truepos = df[df.num_detections==1]
    falseneg = df[df.num_detections==0]
    falsepos = df[df.num_detections>1]
    numfalsepos = int(falsepos.num_detections.sum() - len(falsepos))
    numtruepos = len(truepos) + len(falsepos)
    numfalseneg = len(falseneg)

    print('{} true positive detections, {} false positive (additional) detections, {} false negative detections'.format(numtruepos, numfalsepos, numfalseneg))

    inplane_err_sq = (truepos.x_true - truepos.x_pred)**2 + (truepos.y_true - truepos.y_pred)**2
    inplane_RMSE = np.sqrt(inplane_err_sq.sum()/len(inplane_err_sq))
    inplane_err = np.sqrt(inplane_err_sq)

    fig, ax = plt.subplots()
    loc = ax.scatter(truepos.x_true, truepos.y_true, c=np.log(inplane_err), cmap='Spectral')
    ax.set_xlabel(r'$x_p$ [px]')
    ax.set_xlabel(r'$y_p$ [px]')
    ax.grid(alpha=0.3)
    fig.colorbar(loc, label='log(In-plane error [px])')
    ax.annotate('In-plane RMSE: {}px'.format('%.1f'%inplane_RMSE), xy=(0.05, 0.95), xycoords='axes fraction', bbox=dict(facecolor='white', edgecolor='black',alpha=0.5))
    fig.tight_layout()
    fig.savefig(saveheader + '_inplane_err.png')
    plt.show()
    
    ext_err_sq = (truepos.ext_true - truepos.ext_pred)**2
    ext_RMSE = np.sqrt(ext_err_sq.sum()/len(ext_err_sq))
    ext_percent = np.sqrt(ext_err_sq)/truepos.ext_true
    print(ext_percent)
    ext_perror = ext_percent.mean()*100

    fig, ax = plt.subplots()
    ax.plot(truepos.ext_true, truepos.ext_true, c='r')
    ax.scatter(truepos.ext_true, truepos.ext_pred, alpha=0.3, c='b')
    ax.annotate('{}% Extent Error'.format('%.1f'%ext_perror), xy=(0.05, 0.95), xycoords='axes fraction', bbox=dict(facecolor='white', edgecolor='black',alpha=0.5))
    ax.set_xlabel('True feature size [px]')
    ax.set_ylabel('Predicted bounding box size [px]')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(saveheader + '_ext_err.png')
    plt.show()
    
        
if __name__ == '__main__':
    #localizer_accuracy(configuration='yolov5s', nframes = 5000)
    localizer_accuracy(version=2, nframes = 5000)
