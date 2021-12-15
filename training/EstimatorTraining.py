import cupy as cp
import torch.optim as optim
import torch.nn as nn
from torch_estimator_arch import TorchEstimator
from Torch_DataLoader import makedata, EstimatorDataset
import torch, json, os
import pandas as pd
import numpy as np
from EarlyStopping import EarlyStopping

def load_checkpoint(filepath):
	checkpoint = torch.load(filepath)
	model = TorchEstimator()
	model.load_state_dict(checkpoint['state_dict'])
	optimizer = optim.RMSprop(net.parameters(), lr=1e-5)
	optimizer.load_state_dict(checkpoint['optimizer'])
	return model, optimizer


class EstimatorTraining(object):
	def __init__(self, config):
		self.config = config or None
		self.is_setup = False

		#stuff that gets set in do_setup():
		self.model = None
		self.device = None
		self.trainloader = None
		self.testloader = None
		self.weights_dir = None
		self.cfg_path = None
		self.train_info_path = None
		self.checkpoint_every = None

		#stuff that gets set in train() or subtrain:
		self.losses = None
		self.best_state_checkpoint = None
		self.best_loss = None
		self.stoppers = None
		self.epoch = 0

		#same for every model. make settable?
		self.optimizer = optim.RMSprop(net.parameters(), lr=1e-5)
		self.criterion = nn.SmoothL1Loss()

		if self.config:
			self.do_setup()
			self.is_setup = True

	@property
	def config(self):
		return self.config

	@config.setter
	def config(self, config):
		self.config = config
		self.do_setup()
		self.is_setup = True

	@property
	def model(self):
		return self.model

	def do_setup(self):
		makedata(config)

		if config['training']['continue']:
			print('Searching for model checkpoint')
		    try:
		        loadpath = config['training']['savefile'] + '_checkpoints/best.pt'
		        self.model, self.optimizer = load_checkpoint(loadpath)
		        print('Checkpoint loaded')
		    except:
		        print('No checkpoint found, creating a new model')
		else:
			self.model = TorchEstimator()
			print('Creating a new model')

        self.model.train()

        #use GPU if you have one
		if torch.cuda.device_count():
		    print('Using GPU')
		    self.device = 'cuda:0'
		    self.model.to(device)
		else:
		    print('No CUDA device detected, falling back to CPU')
		    self.device = 'cpu'

		#define data loaders here
		train_set = EstimatorDataset(self.config, settype='train')
		self.trainloader = torch.utils.data.DataLoader(train_set, batch_size=config['training']['batchsize'],
		                                         pin_memory=True, num_workers = config['training']['num_workers'])

		test_set = EstimatorDataset(self.config, settype='test')
		self.testloader = torch.utils.data.DataLoader(test_set, batch_size=config['training']['batchsize'],
		                                         pin_memory=True, num_workers = config['training']['num_workers'])

		self.weights_dir = self.config['training']['savefile'] + '_checkpoints/'
		if not os.path.isdir(self.weights_dir):
		    os.mkdir(self.weights_dir)

		self.cfg_path = self.config['training']['savefile'] +'.json'
		with open(cfg_path, 'w') as f:
		    json.dump(self.config, f)

		self.train_info_path = self.config['training']['savefile']+'_train_info.csv'


		self.checkpoint_every = self.config['training']['checkpoint_every']


	def loss_fn(self, outputs, labels):
	    z1, a1, n1 = torch.transpose(outputs,0,1)
	    z2, a2, n2 = torch.transpose(labels,0,1)

	    loss1 = self.criterion(z1, z2)
	    loss2 = self.criterion(a1, a2)
	    loss3 = self.criterion(n1, n2)

	    return loss1,loss2,loss3

	def _train_auto(self):
		self.stoppers = {'z_stopper': EarlyStopping(),
						'a_stopper': EarlyStopping(),
						'n_stopper': EarlyStopping()}

		while not np.all([x.early_stop for x in self.stoppers.values()]):
			print('Epoch {}'.format(self.epoch))

			#training loop
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
		        print('Batch {}/{}. Loss: {}'.format(i, len(self.trainloader), loss.item()), end='\r')

		    #evaluation loop
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

		    #update best checkpoint
		    if running_tloss < self.best_loss:
        		self.best_loss = running_tloss
        		self.best_state_checkpoint = {'state_dict': self.model.state_dict(),
                                 'optimizer' : self.optimizer.state_dict()}

            #update losses
            epoch_loss = running_loss / len(self.trainloader)
    		epoch_tloss = running_tloss / len(self.testloader)
    		print('Train loss: {}, test loss: {}'.format('%.3f'%epoch_loss, '%.3f'%epoch_tloss))

    		self.losses['train_loss'].append(epoch_loss)
    		self.losses['test_loss'].append(epoch_tloss)
    		self.losses['z_loss'].append(rzloss/len(self.trainloader))
    		self.losses['z_testloss'].append(ztloss/len(self.testloader))
    		self.losses['a_loss'].append(raloss/len(self.trainloader))
    		self.losses['a_testloss'].append(atloss/len(self.testloader))
    		self.losses['n_loss'].append(rnloss/len(self.trainloader))
    		self.losses['n_testloss'].append(ntloss/len(self.testloader))

    		#early stopping
    		self.stoppers['z_stopper'](ztloss/len(self.testloader))
    		self.stoppers['a_stopper'](atloss/len(self.testloader))
    		self.stoppers['n_stopper'](ntloss/len(self.testloader))

    		if self.stoppers['z_stopper'].early_stop:
    			self.model.freeze_z()
    		if self.stoppers['a_stopper'].early_stop:
    			self.model.freeze_a()
    		if self.stoppers['n_stopper'].early_stop:
    			self.model.freeze_n()
    		if np.any([x.early_stop for x in self.stoppers.values()]):
    			self.model.freeze_shared()

    		self.epoch += 1

    		#checkpoint
    		if self.epoch % self.checkpoint_every == 0:
    			num_state_checkpoint = {'state_dict': self.model.state_dict(),
                                 'optimizer' : self.optimizer.state_dict()}
		        numpath = self.weightsdir + 'epoch{}.pt'.format(self.epoch)
		        torch.save(num_state_checkpoint, numpath)

		        df = pd.DataFrame(data = {'epochs': np.arange(self.epoch), **self.losses})
		        df.to_csv(self.train_info_path)
		        print('Saved checkpoint at {} epochs'.format(self.epoch))

			

	def _train_maual(self, epochs):
		while self.epoch < epochs:
			print('Epoch {} / {}'.format(self.epoch, epochs))

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
		        print('Batch {}/{}. Loss: {}'.format(i, len(self.trainloader), loss.item()), end='\r')

		    #evaluation loop
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

		    #update best checkpoint
		    if running_tloss < self.best_loss:
        		self.best_loss = running_tloss
        		self.best_state_checkpoint = {'state_dict': self.model.state_dict(),
                                 'optimizer' : self.optimizer.state_dict()}

            #update losses
            epoch_loss = running_loss / len(self.trainloader)
    		epoch_tloss = running_tloss / len(self.testloader)
    		print('Train loss: {}, test loss: {}'.format('%.3f'%epoch_loss, '%.3f'%epoch_tloss))

    		self.losses['train_loss'].append(epoch_loss)
    		self.losses['test_loss'].append(epoch_tloss)
    		self.losses['z_loss'].append(rzloss/len(self.trainloader))
    		self.losses['z_testloss'].append(ztloss/len(self.testloader))
    		self.losses['a_loss'].append(raloss/len(self.trainloader))
    		self.losses['a_testloss'].append(atloss/len(self.testloader))
    		self.losses['n_loss'].append(rnloss/len(self.trainloader))
    		self.losses['n_testloss'].append(ntloss/len(self.testloader))

    		self.epoch += 1

    		#checkpoint
    		if self.epoch % self.checkpoint_every == 0:
    			num_state_checkpoint = {'state_dict': self.model.state_dict(),
                                 'optimizer' : self.optimizer.state_dict()}
		        numpath = self.weightsdir + 'epoch{}.pt'.format(self.epoch)
		        torch.save(num_state_checkpoint, numpath)

		        df = pd.DataFrame(data = {'epochs': np.arange(self.epoch), **self.losses})
		        df.to_csv(self.train_info_path)
		        print('Saved checkpoint at {} epochs'.format(self.epoch))


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
		if epochs is None:
			self._train_auto()
		else: 
			self._train_manual(epochs)

		print('Finished training')
		last_state_checkpoint = {'state_dict': self.model.state_dict(),
                         'optimizer' : self.optimizer.state_dict()}

        bestpath = self.weightsdir + 'best.pt'
		lastpath = self.weightsdir + 'last.pt'
		torch.save(self.best_state_checkpoint, bestpath)
		torch.save(last_state_checkpoint, lastpath)

		df = pd.DataFrame(data = {'epochs': np.arange(self.epoch), **self.losses})
		df.to_csv(self.train_info_path)



if __name__ == '__main__':
	with open('torch_train_config.json', 'r') as f:
    	config = json.load(f)

    est_train = EstimatorTraining(config)

    est_train.train()
