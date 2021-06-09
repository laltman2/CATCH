import cv2, json
from matplotlib import pyplot as plt
import numpy as np
import os, os.path
from CATCH.CATCHobject import CATCH
from CATCH.Localizerv5_pip import Localizer
from CATCH.Estimator_Torch import Estimator
import pandas as pd

est = Estimator()
loc = Localizer()
catch = CATCH(estimator=est, localizer=loc)

savedict = []
path = '/path/to/normed/images' #specify folder of normalized images here
numimgs = len([f for f in os.listdir(path)if os.path.isfile(os.path.join(path, f))])

for i in range(numimgs):
    filepath = path +'/image' + str(i).zfill(4) + '.png'
    localim = cv2.imread(filepath)
    results = catch.analyze([localim])
    results['framepath'] = filepath
    if not results.empty:
        results['framenum'] = [i]*len(results['framenum'])
    savedict.append(results)
    print('Completed frame {}'.format(i), end='\r')

savedict = pd.concat(savedict)
print(savedict)
savedict.to_csv('results_file.csv', index=False) #specify name of results file here
print('saved ML')
