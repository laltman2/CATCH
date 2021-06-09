from pylorenzmie.analysis import Feature
from pylorenzmie.theory import LMHologram
from pylorenzmie.utilities import coordinates
from CATCH.CATCHobject import crop_frame
from CATCH.utilities.visualization import report, present
import cv2
import ast
import pandas as pd
import numpy as np

df = pd.read_csv('results_file.csv') #specify your ML preds file here
df['bbox'] = df['bbox'].apply(ast.literal_eval) #convert csv string to tuple

df = df[df['edge']==False] #remove features near the edge (optional)


refined = []
for index, row in df.iterrows():
    f = Feature(model=LMHologram(double_precision=False))
    
    # Instrument configuration
    ins = f.model.instrument
    ins.wavelength = 0.447     # [um]
    ins.magnification = 0.048  # [um/pixel]
    ins.n_m = 1.34

    #provide initial parameter estimates from ML
    p = f.particle
    p.properties = row

    #crop experimental data and give it to feature
    frame = cv2.imread(row.framepath, cv2.IMREAD_GRAYSCALE)
    crop = crop_frame(frame, [row])[0]
    
    f.data = crop/np.mean(crop)
    f.coordinates = coordinates(crop.shape, corner=row.bbox[0])

    #mask settings
    f.mask.percentpix = 0.2
    f.mask.distribution = 'radial'
    
    #fit
    result = f.optimize()
    report(result)

    #replace refined values in new df
    refrow = row.copy()
    newprops = pd.Series(f.particle.properties)
    refrow.update(newprops)
    refrow['redchi'] = result.redchi

    refined.append(refrow)

refined = pd.concat(refined)

refined.to_csv('refined_results_file.csv', index=False) #specify name of refined results file here
