#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from yolov5 import YOLOv5


class Localizer(YOLOv5):
    '''
    Attributes
    __________
    configuration : str | None
        Name of trained configuration
        default: None -- use standard hologram localizer
    version : str | int | None
        Version of trained model to use
        default: None -- use latest version
    threshold : float
        Confidence threshold for feature detection
        default: 0.5
    device : str or None
        Computational device: 'cpu' or '0' for gpu
        default: None -- choose automatically

    Methods
    _______
    detect(img_list)
    '''

    def __init__(self,
                 configuration=None,
                 version=None,
                 threshold=0.5,
                 device=None):
        self.configuration = configuration or 'yolov5_test'
        self.version = version or 2
        
        basedir = os.path.dirname(os.path.abspath(__file__))
        cfg_version = self.configuration + str(self.version)
        path = (basedir, 'cfg_yolov5', cfg_version, 'weights', 'best.pt')
        self.model_path = os.path.join(*path)        
        self.threshold = threshold
        self.device = device
        self.load_model()

    def detect(self, img_list=[]):
        '''Detect and localize features in an image

        Inputs
        ------
        img_list: list
           images to be analyzed
        thresh: float
           threshold confidence for detection

        Outputs
        -------
        predictions: list
            list of dicts
            len(predictions): number of detected features
            Each prediction consists of
            {'label': l, 
             'conf': c,
             'bbox': (x1, y1, w, h), 
             'x_p': x, 
             'y_p': y}
            l: str
            c: float between 0 and 1
            x1, y1: bottom left corner of bounding box
            w, h: width and height of bounding box
            x, y: centroid position
        '''
        size = np.max(np.array(img_list).shape[1:3])
        
        results = self.predict(img_list, size=size)
        predictions = []
        for image in results.pred:
            image = image.cpu().numpy()
            prediction = []
            image = [x for x in image if x[4] > self.threshold]
            for pred in image:
                x1, y1, x2, y2 = pred[:4]
                w, h = x2 - x1, y2 - y1
                x_p, y_p = (x1 + x2)/2., (y1 + y2)/2.
                bbox = [x1, y1, w, h]
                conf = pred[4]
                ilabel = int(pred[5])
                label = results.names[ilabel]
                prediction.append({'label': label,
                                   'conf': conf,
                                   'x_p': x_p,
                                   'y_p': y_p,
                                   'bbox': bbox})
            predictions.append(prediction)
        return predictions
            

if __name__ == '__main__':
    import cv2
    import matplotlib
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle

    # Create a Localizer
    localizer = Localizer()
    
    # Read hologram (not normalized)
    img_file = os.path.join('examples', 'test_image_large.png')
    test_img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)

    # Use Localizer to identify features in the hologram
    features = localizer.detect([test_img])[0]

    # Show and report results
    style = dict(fill=False, linewidth=3, edgecolor='r')
    report = 'Feature at ({0:.1f}, {1:.1f}) with {2:.2f} confidence'

    matplotlib.use('Qt5Agg')
    fig, ax = plt.subplots()
    ax.imshow(test_img, cmap='gray')
    for feature in features:
        (x1, y1, w, h) = feature['bbox']
        corner = (x1, y1)
        ax.add_patch(Rectangle(xy=corner, width=w, height=h, **style))
        print(report.format(feature['x_p'], feature['y_p'], feature['conf']))
    plt.show()
