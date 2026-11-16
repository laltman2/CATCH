#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Make Training Data'''
import json
from collections.abc import Mapping, Sequence
from typing import Any
from pylorenzmie.theory import (LMHologram, Sphere)
import numpy as np
import cv2
import os
import shutil


def make_model(shape: Sequence[int],
               instrument: Mapping[str, float]) -> Any:
    '''Returns a hologram calculator for an image of the given shape

    Parameters
    ----------
    shape : sequence of int
        ``(rows, columns)`` of the image, in pixels.
    instrument : mapping
        Properties of the instrument: ``wavelength`` and ``magnification``
        (micrometers per pixel), and ``n_m``, the refractive index of the
        medium.

    Returns
    -------
    model : pylorenzmie.theory.LMHologram
        Set ``model.particle`` and call ``model.hologram()``, which returns
        the flattened image.
    '''
    model = LMHologram()
    model.instrument.properties = dict(instrument)
    model.coordinates = model.meshgrid(tuple(shape))
    return model


def particle_properties(sphere: Sphere) -> dict[str, float]:
    '''Returns the properties of a sphere, as plain floats'''
    return {key: float(value) for key, value in sphere.properties.items()}


def feature_extent(sphere, config, nfringes=20, maxrange=300):
    '''Radius of holographic feature in pixels'''

    h = make_model((maxrange, maxrange), config['instrument'])
    h.particle = Sphere(a_p=sphere.a_p, n_p=sphere.n_p, z_p=sphere.z_p)
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
    annotation = [json.dumps(particle_properties(s), sort_keys=True)
                  for s in sample]
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
    if nrange[0] == nrange[1]:
        nspheres = nrange[0]
    else:
        nspheres = np.random.randint(nrange[0], nrange[1])
    sample = []
    for index in range(nspheres):
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

def add_overlaps(ext, num, config):
    particle = config['particle']
    outer_spheres = []
    for _ in range(num):
        sphere = Sphere()
        for prop in ('a_p', 'n_p', 'k_p', 'z_p'):
            setattr(sphere, prop, make_value(particle[prop]))
        #place randomly x,y within an annulus with 1.5*ext < r < 2*ext
        r = np.random.uniform(1.5*ext, 2*ext)
        theta = np.random.uniform(0, 2*np.pi)
        xval = r * np.cos(theta) + ext
        yval = r * np.sin(theta) + ext
        sphere.x_p = xval
        sphere.y_p = yval
        outer_spheres.append(sphere)
    return outer_spheres


def mtd(configfile='mtd.json'):
    '''Make Training Data'''
    # read configuration
    with open(configfile, 'r') as f:
        config = json.load(f)

    # set up pipeline for hologram calculation
    shape = config['shape']
    holo = make_model(shape, config['instrument'])

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

    filenames = []
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
        filenames.append(imgname.format(n))
    with open(filetxtname, 'w') as fp:
        fp.writelines(f'{name}\n' for name in filenames)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('configfile', type=str,
                        nargs='?', default='mtd.json',
                        help='configuration file')
    args = parser.parse_args()

    mtd(args.configfile)
