try:
    import cupy as cp
except ImportError:
    print('falling back to CPU')
import torch
import torch.optim as optim
import torch.nn as nn
from torch_estimator_arch import TorchEstimator
import json
import os
import pandas as pd
import numpy as np
from Torch_DataLoader import makedata, EstimatorDataset
from EarlyStopping import EarlyStopping


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = TorchEstimator()
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = optim.RMSprop(model.parameters(), lr=1e-5)
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer


class EstimatorTraining(object):
    def __init__(self, config):
        self.config = config
        self.is_setup = False

        # stuff that gets set in do_setup():
        self.model = None
        self.device = None
        self.trainloader = None
        self.testloader = None
        self.weights_dir = None
        self.cfg_path = None
        self.train_info_path = None
        self.checkpoint_every = None

        # stuff that gets set in train() or subtrain:
        self.losses = None
        self.best_state_checkpoint = None
        self.best_loss = None
        self.stoppers = None
        self.epoch = 0

        # same for every model. make settable?
        self.optimizer = None
        self.criterion = nn.SmoothL1Loss()

        if self.config:
            self.do_setup()
            self.is_setup = True

    def do_setup(self):
        makedata(self.config)

        cfg = self.config['training']
        if cfg['continue']:
            print('Searching for model checkpoint')
            try:
                loadpath = cfg['savefile'] + '_checkpoints/best.pt'
                self.model, self.optimizer = load_checkpoint(loadpath)
                print('Checkpoint loaded')
            except Exception as ex:
                print(f'No checkpoint found: {ex}')
                print('Creating a new model')
        else:
            self.model = TorchEstimator()
            print('Creating a new model')

        self.model.train()

        self.optimizer = optim.RMSprop(self.model.parameters(), lr=1e-5)

        # use GPU if you have one
        if torch.cuda.device_count():
            print('Using GPU')
            self.device = 'cuda:0'
            self.model.to(self.device)
        else:
            print('No CUDA device detected, falling back to CPU')
            self.device = 'cpu'

        # define data loaders here
        load = torch.utils.data.DataLoader
        opts = {'batch_size': cfg['batchsize'],
                'pin_memory': True,
                'num_workers': cfg['num_workers']}
        train_set = EstimatorDataset(self.config, settype='train')
        self.trainloader = load(train_set, **opts)

        test_set = EstimatorDataset(self.config, settype='test')
        self.testloader = load(test_set, **opts)

        self.weights_dir = cfg['savefile'] + '_checkpoints/'
        if not os.path.isdir(self.weights_dir):
            os.mkdir(self.weights_dir)

        self.cfg_path = cfg['savefile'] + '.json'
        with open(self.cfg_path, 'w') as f:
            json.dump(self.config, f)
        self.train_info_path = cfg['savefile'] + '_train_info.csv'
        self.checkpoint_every = cfg['checkpoint_every']

    def loss_fn(self, outputs, labels):
        z1, a1, n1 = torch.transpose(outputs, 0, 1)
        z2, a2, n2 = torch.transpose(labels, 0, 1)

        loss1 = self.criterion(z1, z2)
        loss2 = self.criterion(a1, a2)
        loss3 = self.criterion(n1, n2)

        return loss1, loss2, loss3

    def _train_auto(self):
        cfg = self.config['training']
        patience = cfg['patience']
        opts = {'patience': patience, 'delta': 0}
        self.stoppers = {'z_stopper': EarlyStopping(**opts),
                         'a_stopper': EarlyStopping(**opts),
                         'n_stopper': EarlyStopping(**opts)}

        while not np.all([x.early_stop for x in self.stoppers.values()]):
            print(f'Epoch {self.epoch}')

            # training loop
            self.model.train()
            running_loss = 0.0
            rzloss = 0.0
            raloss = 0.0
            rnloss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                self.optimizer.zero_grad()

                images, scales, labels = data

                if self.device != 'cpu':
                    images = images.to(self.device)
                    scales = scales.to(self.device)
                    labels = labels.to(self.device)

                outputs = self.model(images, scales)

                paramloss = self.loss_fn(outputs, labels)
                zl, al, nl = paramloss
                rzloss += zl.item()
                raloss += al.item()
                rnloss += nl.item()

                loss = sum(list(paramloss))
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                total = len(self.trainloder)
                print(f'Batch {i}/{total}. Loss: {loss.item()}', end='\r')

            # evaluation loop
            self.model.eval()
            running_tloss = 0.0
            ztloss = 0.0
            atloss = 0.0
            ntloss = 0.0
            with torch.no_grad():
                for i, data in enumerate(self.testloader, 0):
                    images, scales, labels = data

                    if self.device != 'cpu':
                        images = images.to(self.device)
                        scales = scales.to(self.device)
                        labels = labels.to(self.device)

                    outputs = self.model(images, scales)

                    param_tloss = self.loss_fn(outputs, labels)
                    zl, al, nl = param_tloss
                    ztloss += zl.item()
                    atloss += al.item()
                    ntloss += nl.item()
                    tloss = sum(list(param_tloss))

                    running_tloss += tloss.item()

            # update best checkpoint
            if running_tloss < self.best_loss:
                self.best_loss = running_tloss
                checkpoint = {'state_dict': self.model.state_dict(),
                              'optimizer': self.optimizer.state_dict()}
                self.best_state_checkpoint = checkpoint

            # update losses
            ntrain = len(self.trainloader)
            ntest = len(self.testloader)
            epoch_loss = running_loss / ntrain
            epoch_tloss = running_tloss / ntest
            print(f'{epoch_loss=:.3f}, {epoch_tloss=:.3f}')

            self.losses['train_loss'].append(epoch_loss)
            self.losses['test_loss'].append(epoch_tloss)
            self.losses['z_loss'].append(rzloss/ntrain)
            self.losses['z_testloss'].append(ztloss/ntest)
            self.losses['a_loss'].append(raloss/ntrain)
            self.losses['a_testloss'].append(atloss/ntest)
            self.losses['n_loss'].append(rnloss/ntrain)
            self.losses['n_testloss'].append(ntloss/ntest)

            # early stopping
            self.stoppers['z_stopper'](ztloss/ntest)
            self.stoppers['a_stopper'](atloss/ntest)
            self.stoppers['n_stopper'](ntloss/ntest)

            if self.stoppers['z_stopper'].early_stop:
                self.model.freeze_z()
                # print('Finished training z_p')
            if self.stoppers['a_stopper'].early_stop:
                self.model.freeze_a()
                # print('Finished training a_p')
            if self.stoppers['n_stopper'].early_stop:
                self.model.freeze_n()
                # print('Finished training n_p')
            if np.any([x.early_stop for x in self.stoppers.values()]):
                self.model.freeze_shared()

            self.epoch += 1

            # checkpoint
            if self.epoch % self.checkpoint_every == 0:
                checkpoint = {'state_dict': self.model.state_dict(),
                              'optimizer': self.optimizer.state_dict()}
                num_state_checkpoint = checkpoint
                numpath = self.weights_dir + f'epoch{self.epoch}.pt'
                torch.save(num_state_checkpoint, numpath)
                data = {'epochs': np.arange(self.epoch),
                        **self.losses}
                df = pd.DataFrame(data)
                df.to_csv(self.train_info_path)
                print(f'Saved checkpoint at {self.epoch} epochs')

    def _train_manual(self, epochs):
        while self.epoch < epochs:
            print(f'Epoch {self.epoch} / {epochs}')

            self.model.train()
            running_loss = 0.0
            rzloss = 0.0
            raloss = 0.0
            rnloss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                self.optimizer.zero_grad()

                images, scales, labels = data

                if self.device != 'cpu':
                    images = images.to(self.device)
                    scales = scales.to(self.device)
                    labels = labels.to(self.device)

                outputs = self.model(images, scales)

                paramloss = self.loss_fn(outputs, labels)
                zl, al, nl = paramloss
                rzloss += zl.item()
                raloss += al.item()
                rnloss += nl.item()

                loss = sum(list(paramloss))
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                total = len(self.trainloader)
                print(f'Batch {i}/{total}. Loss: {loss.item()}', end='\r')

            # evaluation loop
            self.model.eval()
            running_tloss = 0.0
            ztloss = 0.0
            atloss = 0.0
            ntloss = 0.0
            with torch.no_grad():
                for i, data in enumerate(self.testloader, 0):
                    images, scales, labels = data

                    if self.device != 'cpu':
                        images = images.to(self.device)
                        scales = scales.to(self.device)
                        labels = labels.to(self.device)

                        outputs = self.model(images, scales)

                        param_tloss = self.loss_fn(outputs, labels)
                        zl, al, nl = param_tloss
                        ztloss += zl.item()
                        atloss += al.item()
                        ntloss += nl.item()
                        tloss = sum(list(param_tloss))

                        running_tloss += tloss.item()

            # update best checkpoint
            if running_tloss < self.best_loss:
                self.best_loss = running_tloss
                checkpoint = {'state_dict': self.model.state_dict(),
                              'optimizer': self.optimizer.state_dict()}
                self.best_state_checkpoint = checkpoint

            # update losses
            ntrain = len(self.trainloader)
            ntest = len(self.testloader)
            epoch_loss = running_loss / ntrain
            epoch_tloss = running_tloss / ntest
            print(f'{epoch_loss=:.3f}, {epoch_tloss=:.3f}')

            self.losses['train_loss'].append(epoch_loss)
            self.losses['test_loss'].append(epoch_tloss)
            self.losses['z_loss'].append(rzloss/ntrain)
            self.losses['z_testloss'].append(ztloss/ntest)
            self.losses['a_loss'].append(raloss/ntrain)
            self.losses['a_testloss'].append(atloss/ntest)
            self.losses['n_loss'].append(rnloss/ntrain)
            self.losses['n_testloss'].append(ntloss/ntest)

            self.epoch += 1

            # checkpoint
            if self.epoch % self.checkpoint_every == 0:
                checkpoint = {'state_dict': self.model.state_dict(),
                              'optimizer': self.optimizer.state_dict()}
                num_state_checkpoint = checkpoint
                numpath = self.weights_dir + f'epoch{self.epoch}.pt'
                torch.save(num_state_checkpoint, numpath)
                data = {'epochs': np.arange(self.epoch),
                        **self.losses}
                df = pd.DataFrame(data)
                df.to_csv(self.train_info_path)
                print(f'Saved checkpoint at {self.epoch} epochs')

    def train(self):
        if not self.is_setup:
            raise ValueError('Training not setup. Please load config')

        self.losses = {'train_loss': [], 'test_loss': [],
                       'z_loss': [], 'z_testloss': [],
                       'a_loss': [], 'a_testloss': [],
                       'n_loss': [], 'n_testloss': []}
        self.best_state_checkpoint = None
        self.best_loss = 1e10

        self.epoch = 0

        self.model.unfreeze_all()

        epochs = self.config['training']['epochs']
        if not epochs:
            # print('Training on auto')
            self._train_auto()
        else:
            # print('Training on manual for {} epochs'.format(epochs))
            self._train_manual(epochs)

        print('Finished training')
        last_state_checkpoint = {'state_dict': self.model.state_dict(),
                                 'optimizer': self.optimizer.state_dict()}

        bestpath = self.weights_dir + 'best.pt'
        lastpath = self.weights_dir + 'last.pt'
        torch.save(self.best_state_checkpoint, bestpath)
        torch.save(last_state_checkpoint, lastpath)

        data = {'epochs': np.arange(self.epoch), **self.losses}
        df = pd.DataFrame(data)
        df.to_csv(self.train_info_path)


if __name__ == '__main__':
    with open('torch_train_config.json', 'r') as f:
        config = json.load(f)
    est_train = EstimatorTraining(config)
    est_train.train()
