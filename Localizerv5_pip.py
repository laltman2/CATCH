#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import os
import json
import numpy as np
from yolov5 import YOLOv5

class Localizer(YOLOv5):
    '''
    Attributes
    __________
    configuration : str
        Name of trained configuration
    threshold : float
        Confidence threshold for feature detection
        default: 0.5
    version : str
        Version of trained model to use
        default: ''
    device : str or None
        choose 'cpu' or '0' for gpu
        default: None (will choose automatically)

    Methods
    _______
    detect(img_list)
    '''

    def __init__(self,
                 configuration='holo',
                 threshold=0.5,
                 version='', device = None):
        self.configuration = configuration
        config_version = configuration+str(version)
        config_json = configuration + '.json'
        basedir = os.path.dirname(os.path.abspath(__file__))
        configdir = os.path.join(basedir, 'cfg_yolov5')
        weightspath = os.path.join(configdir, config_version,
                                   'weights', 'best.pt')
        cfgpath = os.path.join(configdir, config_json)

        #might not need cfg but keeping it in anyway
        with open(cfgpath, 'r') as f:
            cfg = json.load(f)

        self.model_path = weightspath
        self.device = device

        self.threshold = threshold

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
        n images => n lists of dicts
        per holo prediction:
             {'conf': 50%, 'bbox': (x_centroid, y_centroid, width, height)}
        '''
        size = np.max(np.array(img_list).shape[1:3])
        
        results = self.predict(img_list, size=size)
        predictions = []
        for image in results.pred:
            image = image.cpu().numpy()
            imagepreds = []
            image = [x for x in image if x[4] > self.threshold]
            for pred in image:
                x1, y1, x2, y2 = pred[:4]
                w = x2-x1
                h = y2-y1
                x_p = (x1+x2)/2.
                y_p = (y1+y2)/2.
                bbox = [x_p, y_p, w, h]
                conf = pred[4]
                ilabel = int(pred[5])
                label = results.names[ilabel]
                imagepreds.append({'label': label, 'conf': conf, 'bbox': bbox})
            predictions.append(imagepreds)
        return predictions
            

if __name__ == '__main__':
    import cv2
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle

    localizer = Localizer('yolov5_test', version=2)
    print('done')
    img_file = 'examples/test_image_large.png'
    test_img = cv2.imread(img_file)
    detection = localizer.detect([test_img])
    example = detection[0]
    fig, ax = plt.subplots()
    ax.imshow(test_img, cmap='gray')
    for feature in example:
        (x, y, w, h) = feature['bbox']
        conf = feature['conf']
        msg = 'Feature at ({0:.1f}, {1:.1f}) with {2:.2f} confidence'
        print(msg.format(x, y, conf))
        print(w*2, h*2)
        test_rect = Rectangle(xy=(x - w/2, y - h/2), width=w, height=h, fill=False, linewidth=3, edgecolor='r')
        ax.add_patch(test_rect)
    plt.show()
