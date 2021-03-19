from __future__ import print_function
import json, shutil, os, cv2, glob, re
import numpy as np
import torch
from YOLO_data_generator import makedata
try:
    from pylorenzmie.theory.CudaLMHologram import CudaLMHologram as LMHologram
except ImportError:
    from pylorenzmie.theory.LMHologram import LMHologram
import yolov5
import gc

'''One-stop training of a new darknet model for classification and localization of features in frame

Steps to follow:
-edit darknet_train_config.json with appropriate params
(make sure you have available disk space)
-Run this file with python or nohup
(dataset generation + training will take at least few hours)

Weights files will save to cfg_darknet/backup every 100 epochs, then, after 1000, every 1000 epochs
'''


configfile='yolov5_train_config.json'

with open(configfile, 'r') as f:
    config = json.load(f)


#File names/numbers
file_header = os.path.abspath(config['directory'])
numtrain = config['nframes_train']
numtest = config['nframes_test']
numeval = config['nframes_eval']

classes = config['particle']['names']
numclasses = len(classes)

#Make test/train data
mtd_config = config.copy()
train_dir = file_header + '/train'
test_dir = file_header + '/test'
eval_dir = file_header + '/eval'

mtd_config['directory'] = train_dir
mtd_config['nframes'] = numtrain
print('Training set')
makedata(config = mtd_config)

mtd_config['directory'] = test_dir
mtd_config['nframes'] = numtest
print('Validation set')
makedata(config = mtd_config)

'''
#Make eval data
mtd_config['directory'] = eval_dir
mtd_config['nframes'] = numeval
print('Validation set')
makedata(config = mtd_config)
'''

basedir = os.getcwd().split('/training')[0]

#prepare config files
save_dir = os.path.abspath(config['training']['save_dir'])
save_name = config['training']['save_name']

save_header = save_dir +'/' + save_name

save_json = save_header+'.json'
with open(save_json, 'w') as f:
    json.dump(config, f)
print('Saved training config')


cfg_template = os.path.abspath('../cfg_yolov5/yolov5_cfg_template.yaml')

with open(cfg_template, 'r') as fr:
    cfg_lines = fr.readlines()

cfg_lines[12] = "train: " + train_dir + '/images/'+'\n'
cfg_lines[13] = "val: " + test_dir + '/images/'+'\n'
cfg_lines[16] = "nc: " + str(numclasses)+'\n'
cfg_lines[19] = "names: " + str(classes)


#create .yaml file
cfg_yaml = os.path.abspath(save_header +'.yaml')

with open(cfg_yaml, 'w') as fw:
    fw.writelines(cfg_lines)

    
batch = config['training']['batch']
epochs = config['training']['epochs']

img_size = np.max(config['shape'])

#model_size must be one of: ['s', 'm', 'l', 'x']
model_size = config['training']['model_size']

yolo_path = yolov5.__file__.split('yolov5')[0]
yolo_dir = os.path.dirname(os.path.realpath(yolo_path))


os.chdir(yolo_path)
print(yolo_path)

#check to see if training script is downloaded from github
#if not, download from url
if not os.path.exists('./y5_train.py'):
    import urllib.request
    url = 'https://raw.githubusercontent.com/fcakyon/yolov5-pip/main/scripts/train.py'
    urllib.request.urlretrieve(url, filename='./y5_train.py')


#gc.collect()

if config['training']['continue']:
    weights_path = save_header + '/weights/last.pt'
    if os.path.exists(weights_path):
        weights = weights_path
    else:
        print('No weights file found, starting a new model')
        weights = 'yolov5{}.pt'.format(model_size)
else:
    weights = 'yolov5{}.pt'.format(model_size)

num_workers = config['training']['num_workers']
    
cmd = 'python3 y5_train.py --img {} --batch {} --epochs {} --data {} --weights {} --project {} --name {} --workers {}'.format(img_size, batch, epochs, cfg_yaml, weights, save_dir, save_name, num_workers)

if config['training']['resume']:
    weights_path = save_header + '/weights/last.pt'
    if os.path.exists(weights_path):
        cmd = 'python3 y5_train.py --resume {}'.format(prev_weights_path)
    else:
        print('No weights file found, starting a new model')

print(cmd)

torch.cuda.empty_cache()

os.system(cmd)


if config['delete_files_after_training']:
    shutil.rmtree(file_header)
