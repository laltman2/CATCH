# Training custom Localizers and Estimators

All training occurs within CATCH.training. To train a custom model, modify the config files for either Localizer or Estimator and then run the corresponding training script. It is recommended to use a GPU for training if available. nohup is also useful for long training, especially if using ssh.

## Training a YOLOv5 Localizer

1. Modify the config file: yolov5_train_config.json

### Configuration options

* Instrument: microscope and medium properties for training data. These are unlikely to affect performance of the Localizer much.
  - wavelength: imaging laser wavelength [nanometers]
  - magnification: objective lens magnification [microns per pixel]
  - n_m: refractive index of the medium

* Particle: property ranges for generated particles in training data. Each two-element list gives minimum and maximum values.
  - nspheres: number of particles to generate in each frame
  - names: labels for particle types. If len(names) > 1, you must set the class definitions in Classify.py (see optional step 2)
  - a_p: radius [microns]
  - n_p: refractive index
  - k_p: extinction coefficient
  - x_p: in-plane horizontal position of particle center [pixels]
  - y_p: in-plane vertical position of particle center [pixels]
  - z_p: axial position of particle center [pixels]

* Training: options for initializing and training the model
  - batch: batch size of training data. Recommended: use the largest power of two that your system's memory will allow
  - epochs: number of epochs to train for
  - model_size: complexity of model to initialize. Options: 's', 'm', 'l', 'x'. Recommended: use smallest model that will train.
  - resume: only set this to true if your training was interrupted and closed unexpectedly.
  - continue: if true, initialize a previously trained model to continue training
  - save_dir: location to save trained model files. Recommended: keep this as cfg_yolo/
  - save_name: unique name of your model

* Directory: location to save training data
* imgtype: format to save training images
* shape: frame shape for training images
* noise: added Gaussian noise to frame [%]
* nframes_train: number of training frames
* nframes_test: number of validation frames
* nframes_eval: number of evaluation frames
* overwrite: option to write over training data if it already exists in the specified location
* delete_files_after_training: option to remove training data once training is complete


2. (Optional) Set class definitions: Classify.py

If your Localizer has only one class or label, skip this step.
If you are creating a multi-class Localizer, you will need to set definitions for which properties correspond to your classes.
Inside Classify.py is an example for a two-class model which distinguishes particles with refractive index above or below that of the medium:

```
if '+n_p' in names:
        #classify based on sign of (n_p - n_m)
        if sphere.n_p < config['instrument']['n_m']:
            return 0
        else:
            return 1
```

Here, a label of '+n_p' corresponds to the class 1 and the condition that particle.n_p > instrument.n_m. A label '-n_p' corresponds to class 0 and condition particle.n_p < instrument.n_m. In the config file, you would set particle.names = ['-n_p', '+n_p'].

3. Run training script:

```
python3 train_yolov5.py
```
