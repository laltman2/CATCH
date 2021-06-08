import timeit
import numpy as np
import pandas as pd

def CATCH_timer(est_config='overlaps', loc_config='small_noisy', numimgs=200, num_trials=10, loc_dataset='../datasets/yolov5_test/eval/', est_dataset='../datasets/overlaps/eval/', device='cuda'):

    modelstart = '''
from CATCH.Estimator_Torch import Estimator
from CATCH.Localizerv5_pip import Localizer
from CATCH.CATCHobject import CATCH
    
est = Estimator(configuration='{}', device='{}')
loc = Localizer(configuration='{}', device='{}')
catch = CATCH(localizer=loc, estimator = est)
'''.format(est_config, device, loc_config, device)
    
    imgload = '''
import cv2
img_list = []
for n in range({}):
    img = cv2.imread('{}images/image'+str(n).zfill(4) + '.png', cv2.IMREAD_GRAYSCALE)
    img_list.append(img)'''

    est_setup = modelstart + imgload.format(numimgs, est_dataset)
    loc_setup = modelstart + imgload.format(numimgs, loc_dataset)
    
    est_stmt = 'results = est.predict(img_list)'
    loc_stmt = 'results = loc.detect(img_list)'
    catch_stmt = 'results = catch.analyze(img_list)'

    strpairs = [(loc_setup, loc_stmt), (est_setup, est_stmt), (loc_setup, catch_stmt)]

    data = {'model': ['localizer', 'estimator', 'catch'],
            'device': [device]*3,
            'num_images': [numimgs]*3, 'num_trials': [num_trials]*3,
            'total_time':[], 'time/trial':[], 'time/image':[]}
    
    for (setup, stmt) in strpairs:
        ntot = timeit.timeit(stmt= stmt, setup= setup, number=num_trials)
        data['total_time'].append(ntot)
        n1 = ntot/num_trials
        data['time/trial'].append(n1)
        nind = n1/numimgs
        data['time/image'].append(nind)

    df = pd.DataFrame(data)

    return df
    
if __name__ == '__main__':
    print(CATCH_timer())
