import json, shutil, os, cv2, ast
from PIL import Image
import numpy as np
from pylorenzmie.utilities.mtd import make_value, make_sample, feature_extent
try:
    from pylorenzmie.theory.CudaLMHologram import CudaLMHologram as LMHologram
except ImportError:
    from pylorenzmie.theory.LMHologram import LMHologram
from pylorenzmie.theory.Instrument import coordinates
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.autograd import Variable


def format_json(sample, config, scale=1):
    '''Returns a string of JSON annotations'''
    annotation = []
    for s in sample:
        savestr = s.dumps(sort_keys=True)
        savedict = ast.literal_eval(savestr)
        savedict['scale'] = scale
        savestr = json.dumps(savedict)
        annotation.append(savestr)
    return json.dumps(annotation, indent=4)

def scale_int(s, config):
    shape = config['shape']
    ext = feature_extent(s, config)
    #introduce 1% noise to ext
    ext = np.random.normal(ext, 0.01*ext)
    extsize = ext*2
    shapesize = shape[0]
    if extsize <= shapesize:
        scale = 1
    else:
        scale = int(np.floor(extsize/shapesize) + 1)
    newshape = [i * scale for i in shape]
    holo = LMHologram(coordinates=coordinates(newshape))
    holo.instrument.properties = config['instrument']
    # ... calculate hologram
    frame = np.random.normal(0, config['noise'], newshape)
    holo.particle = s
    holo.particle.x_p += (scale-1)*100
    holo.particle.y_p += (scale-1)*100
    frame += holo.hologram().reshape(newshape)
    frame = np.clip(100 * frame, 0, 255).astype(np.uint8)
    #decimate
    frame = frame[::scale, ::scale]
    return frame, scale

def scale_float(s, config):
    shape = config['shape']
    ext = feature_extent(s, config)
    #introduce 1% noise to ext
    ext = np.random.normal(ext, 0.01*ext)
    extsize = ext*2
    shapesize = shape[0]
    scale = float(extsize)/float(shapesize)
    newshape = [int(extsize)]*2
    holo = LMHologram(coordinates=coordinates(newshape))
    holo.instrument.properties = config['instrument']
    # ... calculate hologram
    frame = np.random.normal(0, config['noise'], newshape)
    holo.particle = s
    holo.particle.x_p += (scale-1)*100.
    holo.particle.y_p += (scale-1)*100.
    frame += holo.hologram().reshape(newshape)
    frame = np.clip(100 * frame, 0, 255).astype(np.uint8)
    #reshape
    frame = cv2.resize(frame, tuple(shape))
    return frame, scale

def makedata_inner(config, settype='train', nframes=None):
    # create directories and filenames
    directory = os.path.abspath(os.path.expanduser(config['directory'])+settype)
    if nframes is None:
        nframes = config[settype]['nframes']
    start = 0
    tempnum = nframes
    for dir in ('images', 'params'):
        path = os.path.join(directory, dir)
        if not os.path.exists(path):
            os.makedirs(path)
        already_files = len(os.listdir(path))
        if already_files < tempnum:  #if there are fewer than the number of files desired
            tempnum = already_files
    if not config['overwrite']:
        start = tempnum
        if start >= nframes:
            return
    with open(directory + '/config.json', 'w') as f:
        json.dump(config, f)
    filetxtname = os.path.join(directory, 'filenames.txt')
    imgname = os.path.join(directory, 'images', 'image{:04d}.' + config['imgtype'])
    jsonname = os.path.join(directory, 'params', 'image{:04d}.json')
    filetxt = open(filetxtname, 'w')
    #always only one particle per stamp
    config['particle']['nspheres'] = [1,2]
    shape = config['shape']
    for n in range(start, nframes):  # for each frame ...
        print(imgname.format(n))
        sample = make_sample(config) # ... get params for particles
        s = sample[0]
        if config['scale_integer']:
            frame, scale = scale_int(s, config)
        else:
            frame, scale = scale_float(s, config)
    
        # ... and save the results
        cv2.imwrite(imgname.format(n), frame)
        with open(jsonname.format(n), 'w') as fp:
            fp.write(format_json(sample, config, scale))
        filetxt.write(imgname.format(n) + '\n')
    return

def makedata(config):
    makedata_inner(config, settype='train')
    makedata_inner(config, settype='test')
    makedata_inner(config, settype='eval')



def PILimage_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    imsize = image.size[0]
    loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    try:
        return image.cuda()
    except:
        return image



class EstimatorDataset(Dataset):

    def __init__(self, config, settype='train'):
        self.batch_size = config['training']['batchsize']
        self.settype = settype
        self.nframes = config[settype]['nframes']
        self.directory = os.path.join(config['directory'], self.settype)
        self.directory = os.path.abspath(self.directory)
        self.config = config

        #preprocessing steps
        self.img_transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(num_output_channels=1)])
        self.params_transform = None #need to rescale by max/min values

    def __len__(self):
        return self.nframes

    def __getitem__(self, idx):
        #idx: batch_number, between 0 and len()
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        imgname = os.path.join(self.directory, 'images', 'image{:04d}.' + self.config['imgtype']).format(idx)
        jsonname = os.path.join(self.directory, 'params', 'image{:04d}.json').format(idx)

        with open(jsonname, 'r') as fp:
            param_string = json.load(fp)[0]
            params = ast.literal_eval(param_string)

        image = cv2.imread(imgname)

        z_p = params['z_p']
        a_p = params['a_p']
        n_p = params['n_p']
        scale = params['scale']

        outputs = np.array([z_p, a_p, n_p])
        outputs = outputs.astype('float')

        if self.img_transform:
            image = self.img_transform(image)
        image = image.unsqueeze(0)

        scale = torch.IntTensor([scale])
        
        if self.params_transform:
            outputs = self.params_transform(outputs)

        return image, scale, outputs
        
        
        


if __name__ == '__main__':
    config = {
        "instrument": {
            "wavelength": 0.447,
            "magnification": 0.048,
            "n_m": 1.340
        },
        "particle": {
            "a_p": [0.2, 5.0],
            "n_p": [1.38, 2.5],
            "k_p": [0, 0],
            "x_p": [90, 110],
            "y_p": [90, 110],
            "z_p": [50, 600]
        },
        "training": {
            "epochs": 10000,
            "batchsize": 64,
            "savefile": "../keras_models/fully_trained_stamp"
        },
        "directory": "./test_data/",
        "imgtype": "png",
        "shape": [201, 201],
        "noise": 0.05,
        "train": {"nframes": 10},
        "test": {"nframes": 10},
        "eval": {"nframes": 10},
        "overwrite": True,
        "delete_files_after_training": False
    }
    

    makedata(config)
    
