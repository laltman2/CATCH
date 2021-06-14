#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Make Training Data'''
import json
from pylorenzmie.theory import (LMHologram, Sphere)
from pylorenzmie.utilities import coordinates
import numpy as np
import cupy as cp
import cv2
import os
import shutil


def feature_extent(sphere, config, nfringes=20, maxrange=300):
    '''Radius of holographic feature in pixels'''

    x = np.arange(0, maxrange)
    y = np.arange(0, maxrange)
    xv, yv = np.meshgrid(x, y)
    xv = xv.flatten()
    yv = yv.flatten()
    zv = np.zeros_like(xv)
    coordinates = np.stack((xv, yv, zv))
    h = LMHologram(coordinates=coordinates)
    h.instrument.properties = config['instrument']
    h.particle.a_p = sphere.a_p
    h.particle.n_p = sphere.n_p
    h.particle.z_p = sphere.z_p
    # roughly estimate radii of zero crossings
    b = h.hologram() - 1.
    ndx = np.where(np.diff(np.sign(b)))[0] + 1
    if len(ndx) <= nfringes:
        return maxrange
    else:
        return float(ndx[nfringes])


def format_yolo(sample, config):
    '''Returns a string of YOLO annotations'''
    (h, w) = config['shape']
    type = 0  # one class for now
    fmt = '{}' + 4 * ' {:.6f}' + '\n'
    annotation = ''
    for sphere in sample:
        diameter = 2. * feature_extent(sphere, config)
        x_p = sphere.x_p / w
        y_p = sphere.y_p / h
        w_p = diameter / w
        h_p = diameter / h
        annotation += fmt.format(type, x_p, y_p, w_p, h_p)
    return annotation


def format_json(sample, config):
    '''Returns a string of JSON annotations'''
    annotation = []
    for s in sample:
        annotation.append(s.dumps(sort_keys=True))
    return json.dumps(annotation, indent=4)


def make_value(range, decimals=3):
    '''Returns the value for a property'''
    if np.isscalar(range):
        value = range
    elif range[0] == range[1]:
        value = range[0]
    else:
        value = np.random.uniform(range[0], range[1])
    return np.around(value, decimals=decimals)


def make_sample(config):
    '''Returns an array of Sphere objects'''
    particle = config['particle']
    nrange = particle['nspheres']
    mpp = config['instrument']['magnification']
    if nrange[0] == nrange[1]:
        nspheres = nrange[0]
    else:
        nspheres = np.random.randint(nrange[0], nrange[1])
    sample = []
    for n in range(nspheres):
        sphere = Sphere()
        for prop in ('a_p', 'n_p', 'k_p', 'z_p'):
            setattr(sphere, prop, make_value(particle[prop]))
        # Making sure separation between particles is large enough##
        close = True
        aval = sphere.a_p
        zval = sphere.z_p
        while close:
            close = False
            xval = make_value(particle['x_p'])
            yval = make_value(particle['y_p'])
            for s in sample:
                xs, ys, zs = s.x_p, s.y_p, s.z_p
                atest = s.a_p
                dist = np.sqrt(
                    (xs - xval)**2 + (ys - yval)**2 + (zs - zval)**2)
                threshold = (atest + aval) / mpp
                if dist < threshold:
                    close = True
        setattr(sphere, 'x_p', xval)
        setattr(sphere, 'y_p', yval)
        sample.append(sphere)
    return sample

def add_overlaps(ext, num, config):
    particle = config['particle']
    outer_spheres = []
    for n in range(num):
        sphere = Sphere()
        for prop in ('a_p', 'n_p', 'k_p', 'z_p'):
            setattr(sphere, prop, make_value(particle[prop]))
        #place randomly x,y within an annulus with 1.5*ext < r < 2*ext
        r = np.random.uniform(1.5*ext, 2*ext)
        theta = np.random.uniform(0, 2*np.pi)
        xval = r * np.cos(theta) + ext
        yval = r * np.sin(theta) + ext
        setattr(sphere, 'x_p', xval)
        setattr(sphere, 'y_p', yval)
        outer_spheres.append(sphere)
    return outer_spheres
        

def mtd(configfile='mtd.json'):
    '''Make Training Data'''
    # read configuration
    with open(configfile, 'r') as f:
        config = json.load(f)

    # set up pipeline for hologram calculation
    shape = config['shape']
    holo = LMHologram(coordinates=coordinates(shape))
    holo.instrument.properties = config['instrument']

    # create directories and filenames
    directory = os.path.expanduser(config['directory'])
    imgtype = config['imgtype']
    for dir in ('images_labels', 'params'):
        if not os.path.exists(os.path.join(directory, dir)):
            os.makedirs(os.path.join(directory, dir))
    shutil.copy2(configfile, directory)
    filetxtname = os.path.join(directory, 'filenames.txt')
    imgname = os.path.join(
        directory, 'images_labels', 'image{:05d}.' + imgtype)
    jsonname = os.path.join(directory, 'params', 'image{:05d}.json')
    yoloname = os.path.join(directory, 'images_labels', 'image{:05d}.txt')

    filetxt = open(filetxtname, 'w')
    for n in range(config['nframes']):  # for each frame ...
        print(imgname.format(n))
        sample = make_sample(config)   # ... get params for particles
        # ... calculate hologram
        frame = np.random.normal(0, config['noise'], shape)
        if len(sample) > 0:
            holo.particle = sample
            frame += holo.hologram().reshape(shape)
        else:
            frame += 1.
        frame = np.clip(100 * frame, 0, 255).astype(np.uint8)
        # ... and save the results
        cv2.imwrite(imgname.format(n), frame)
        with open(jsonname.format(n), 'w') as fp:
            fp.write(format_json(sample, config))
        with open(yoloname.format(n), 'w') as fp:
            fp.write(format_yolo(sample, config))
        filetxt.write(imgname.format(n) + '\n')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('configfile', type=str,
                        nargs='?', default='mtd.json',
                        help='configuration file')
    args = parser.parse_args()

    mtd(args.configfile)
