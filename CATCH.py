from .Localizerv5_pip import Localizer
from .Estimator_Torch import Estimator
import numpy as np
import pandas as pd

class CATCH(object):

    def __init__(self, localizer=None, estimator=None):

        self.localizer = localizer or Localizer()
        self.estimator = estimator or Estimator()
        self.instrument = self.estimator.config['instrument']

    def crop_frame(self, image, detections):
        crops = []
        img_cols, img_rows = image.shape[:2]
        for detection in detections:
            _, _, w, h = detection['bbox']
            cropsize = np.max([w,h])
            xc, yc = detection['x_p'], detection['y_p']
            if cropsize % 2 == 0:
                right_top = left_bot = int(cropsize/2)
            else:
                right_top = int(np.ceil(cropsize/2.))
                left_bot = int(np.floor(cropsize/2.))
            xbot = xc - left_bot
            xtop = xc + right_top
            ybot = yc - left_bot
            ytop = yc + right_top
            if xbot < 0:
                xbot = 0
                xtop = cropsize
            if ybot < 0:
                ybot = 0
                ytop = cropsize
            if xtop > img_rows:
                xtop = img_rows
                xbot = img_rows - cropsize
            if ytop > img_cols:
                ytop = img_cols
                ybot = img_cols - cropsize
            crop = image[ybot:ytop, xbot:xtop]
            crops.append(crop)
        return crops
            
    def analyze(self, image_list=[]):

        list_of_detections = self.localizer.detect(image_list)
        
        #structure = list(map(len, detections))
        
        for i in range(len(image_list)):
            image = image_list[i]
            detections = list_of_detections[i]

            crops = self.crop_frame(image, detections)

            preds = self.estimator.predict(crops)
            
            
