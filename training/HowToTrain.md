# Training custom Localizers and Estimators

All training occurs within CATCH.training. To train a custom model, modify the config files for either Localizer or Estimator and then run the corresponding training script. It is recommended to use a GPU for training if available. nohup is also useful for long training, especially if using ssh.

## Training a YOLOv5 Localizer

1. Modify the config file: yolov5_train_config.json

### Configuration options