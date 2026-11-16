#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Make Training Data'''
import json
from pylorenzmie.theory import Sphere
from CATCH.utilities.mtd import (feature_extent, make_model,
                                 particle_properties)
from CATCH.training.Classify import classify
import numpy as np
import cv2
import os

def format_yolo(sample, config):
    '''Returns a string of YOLO annotations'''
    (h, w) = config['shape']
    fmt = '{}' + 4 * ' {:.6f}' + '\n'
    annotation = ''
    for sphere in sample:
        stype = classify(sphere, config)
        diameter = 2. * feature_extent(sphere, config)
        x_p = sphere.x_p / w
        y_p = sphere.y_p / h
        w_p = diameter / w
        h_p = diameter / h
        annotation += fmt.format(stype, x_p, y_p, w_p, h_p)
    return annotation


def format_json(sample, config):
    '''Returns a string of JSON annotations'''
    annotation = [json.dumps(particle_properties(s), sort_keys=True)
                  for s in sample]
    return json.dumps(annotation, indent=4)


def make_value(range, decimals=3):
    '''Returns the value for a property'''
    if np.isscalar(range):
        value = range
    elif isinstance(range[0], list): #multiple ranges (ie excluded region(s))
        values = []
        p=[]
        for localrange in range:
            values.append(np.random.uniform(localrange[0], localrange[1]))
            p.append(localrange[1] - localrange[0])
        # choose a range with a probability in proportion to its width
        value = np.random.choice(values, size=1, p=np.array(p)/np.sum(p))[0]
    elif range[0] == range[1]:
        value = range[0]
    else:
        value = np.random.uniform(range[0], range[1])
    return np.around(value, decimals=decimals)


def make_sample(config, max_attempts=1000):
    '''Returns a list of Sphere objects

    Parameters
    ----------
    config : dict
        Configuration, with the ranges of the properties of the particles
        (``config['particle']``) and the instrument.
    max_attempts : int
        Number of positions to try for each sphere before giving up.
        Default: 1000

    Raises
    ------
    RuntimeError
        If no position separates a sphere from the others.
    '''
    particle = config['particle']
    nrange = particle['nspheres']
    mpp = config['instrument']['magnification']
    if nrange[0]==nrange[1]:
        nspheres = nrange[0]
    else:
        nspheres = np.random.randint(nrange[0], nrange[1])
    sample = []
    for index in range(nspheres):
        np.random.seed()
        sphere = Sphere()
        for prop in ('a_p', 'n_p', 'k_p', 'z_p'):
            setattr(sphere, prop, make_value(particle[prop]))
        # make sure that the separation between particles is large enough
        aval = sphere.a_p
        zval = sphere.z_p
        for _ in range(max_attempts):
            xval = make_value(particle['x_p'])
            yval = make_value(particle['y_p'])
            if all(np.sqrt((s.x_p - xval)**2 + (s.y_p - yval)**2 +
                           (s.z_p - zval)**2) >= (s.a_p + aval) / mpp
                   for s in sample):
                break
        else:
            raise RuntimeError(
                f'no position for sphere {index + 1} of {nspheres} '
                f'after {max_attempts} attempts: the range of x_p and y_p '
                'is too small to separate the spheres at their values of '
                'z_p and a_p')
        sphere.x_p = xval
        sphere.y_p = yval
        sample.append(sphere)
    return sample


def makedata(config):
    '''Make Training Data'''
    # set up pipeline for hologram calculation
    shape = config['shape']
    holo = make_model(shape, config['instrument'])

    # create directories and filenames
    directory = os.path.expanduser(config['directory'])
    imgtype = config['imgtype']

    nframes = config['nframes']
    start = 0
    tempnum = nframes
    for dir in ('images', 'labels', 'params'):
        path = os.path.join(directory, dir)
        if not os.path.exists(path):
            os.makedirs(path)
        already_files = len(os.listdir(path))
        if already_files < tempnum:  # fewer than the files desired
            tempnum = already_files
    if not config['overwrite']:
        start = tempnum
        if start >= nframes:
            return
    with open(directory + '/config.json', 'w') as f:
        json.dump(config, f)
    filetxtname = os.path.join(directory, 'filenames.txt')
    imgname = os.path.join(directory, 'images', 'image{:05d}.' + imgtype)
    jsonname = os.path.join(directory, 'params', 'image{:05d}.json')
    yoloname = os.path.join(directory, 'labels' , 'image{:05d}.txt')

    filenames = []
    for n in range(start, nframes):  # for each frame ...
        print(imgname.format(n))
        sample = make_sample(config)   # ... get params for particles
        # ... calculate hologram
        frame = np.random.normal(1., config['noise'], shape)
        if len(sample) > 0:
            holo.particle = sample
            frame += holo.hologram().reshape(shape) - 1.
        frame = np.clip(100 * frame, 0, 255).astype(np.uint8)
        # ... and save the results
        cv2.imwrite(imgname.format(n), frame)
        with open(jsonname.format(n), 'w') as fp:
            fp.write(format_json(sample, config))
        with open(yoloname.format(n), 'w') as fp:
            fp.write(format_yolo(sample, config))
        filenames.append(imgname.format(n))
    with open(filetxtname, 'w') as fp:
        fp.writelines(f'{name}\n' for name in filenames)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('configfile', type=str,
                        nargs='?', default='./darknet_train_config.json',
                        help='configuration file')
    args = parser.parse_args()

    with open(args.configfile, 'r') as f:
        config = json.load(f)

    makedata(config)
