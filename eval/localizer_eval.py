from CATCH.training.YOLO_data_generator import makedata
from CATCH.Localizerv5_pip import Localizer
from CATCH.utilities.mtd import feature_extent
from pylorenzmie.theory import Sphere
import json, os, cv2, ast
import pandas as pd

def localizer_accuracy(configuration='yolov5_test', nframes=None):
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
    makedata(config = mtd_config)

    localizer = Localizer(configuration=configuration)

    imgpath_fmt = config['directory']+'/eval/images/image{}.png'
    parampath_fmt = config['directory']+'/eval/params/image{}.json'

    df = pd.DataFrame(columns = ['img_num', 'x_true', 'y_true', 'ext_true', 'num_detections', 'x_pred', 'y_pred', 'ext_pred', 'conf'])
    for n in range(nframes):
        img = cv2.imread(imgpath_fmt.format(str(n).zfill(4)))
        with open(parampath_fmt.format(str(n).zfill(4)), 'r') as f:
            params = json.load(f)[0]
        sphere = Sphere()
        sphere.loads(params)
        params = ast.literal_eval(params)
        ext = 2. * feature_extent(sphere, mtd_config)

        results = localizer.detect([img])[0]
        
        resultsdict = {'img_num':n, 'x_true':params['x_p'], 'y_true':params['y_p'], 'ext_true':ext, 'num_detections':len(results), 'x_pred':None, 'y_pred':None, 'ext_pred':None, 'conf':None}

        if len(results) == 1:
            r = results[0]
            p_ext = max(r['bbox'][1:])
            resultsdict['x_pred'] = r['x_p']
            resultsdict['y_pred'] = r['y_p']
            resultsdict['conf'] = r['conf']
            resultsdict['ext_pred'] = p_ext
            
        df = df.append(resultsdict, ignore_index=True)
    print(df)

if __name__ == '__main__':
    localizer_accuracy()
