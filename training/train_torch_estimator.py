import cupy as cp
import torch.optim as optim
import torch.nn as nn
from torch_estimator_arch import TorchEstimator
from Torch_DataLoader import makedata, EstimatorDataset
import torch, json, os
import pandas as pd
import numpy as np

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = TorchEstimator()
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = optim.RMSprop(net.parameters(), lr=1e-5)
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer

with open('torch_train_config.json', 'r') as f:
    config = json.load(f)

#make data here
makedata(config)

#define data loaders here

net = TorchEstimator()
optimizer = optim.RMSprop(net.parameters(), lr=1e-5)

if config['training']['continue']:
    try:
        loadpath = config['training']['savefile'] + '_checkpoints/best.pt'
        net, optimizer = load_checkpoint(loadpath)
    except:
        print('No checkpoint found, creating a new model')

print(net)
net.train()

#use GPU if you have one
if torch.cuda.device_count():
    print('Using GPU')
    device = 'cuda:0'
    net.to(device)
else:
    print('No CUDA device detected, falling back to CPU')
    device = 'cpu'



#criterion = nn.MSELoss() 
criterion = nn.SmoothL1Loss()

epochs = config['training']['epochs']
checkpoint_every = config['training']['checkpoint_every']

#define loss function
def loss_fn(outputs, labels):
    z1, a1, n1 = torch.transpose(outputs,0,1)
    z2, a2, n2 = torch.transpose(labels,0,1)

    loss1 = criterion(z1, z2)
    loss2 = criterion(a1, a2)
    loss3 = criterion(n1, n2)

    return loss1,loss2,loss3

train_set = EstimatorDataset(config, settype='train')
trainloader = torch.utils.data.DataLoader(train_set, batch_size=config['training']['batchsize'],
                                         pin_memory=True, num_workers = config['training']['num_workers'])

test_set = EstimatorDataset(config, settype='test')
testloader = torch.utils.data.DataLoader(test_set, batch_size=config['training']['batchsize'],
                                         pin_memory=True, num_workers = config['training']['num_workers'])


weightsdir = config['training']['savefile'] + '_checkpoints/'
if not os.path.isdir(weightsdir):
    os.mkdir(weightsdir)

cfg_path = config['training']['savefile'] +'.json'
with open(cfg_path, 'w') as f:
    json.dump(config, f)

train_info_path = config['training']['savefile']+'_train_info.csv'

train_loss = []
test_loss = []
z_loss = []
z_tloss = []
a_loss = []
a_tloss = []
n_loss = []
n_tloss = []
best_state_checkpoint = None
best_loss = 1e10
for epoch in range(epochs):
    print('Epoch {}/{}'.format(epoch, epochs))

    net.train()
    running_loss = 0.0
    rzloss = 0.0
    raloss = 0.0
    rnloss = 0.0
    for i, data in enumerate(trainloader, 0):
        optimizer.zero_grad()

        images, scales, labels = data

        if device != 'cpu':
            images = images.to(device)
            scales = scales.to(device)
            labels = labels.to(device)

        outputs = net(images, scales)

        paramloss = loss_fn(outputs, labels)
        zl, al, nl = paramloss
        rzloss += zl.item()
        raloss += al.item()
        rnloss += nl.item()
        
        preloss = sum(list(paramloss))
        # add dense layer regularizers
        regloss = 0
        #regloss = torch.norm(net.dense1.weight, p=2) + torch.norm(net.densez.weight, p=2) + torch.norm(net.densea.weight, p=2) + torch.norm(net.densen.weight, p=2)
        loss = preloss + 0.01*regloss
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print('Batch {}/{}. Loss: {}'.format(i, len(trainloader), loss.item()), end='\r')

    net.eval()
    running_tloss = 0.0
    ztloss = 0.0
    atloss = 0.0
    ntloss = 0.0
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            images, scales, labels = data

            if device != 'cpu':
                images = images.to(device)
                scales = scales.to(device)
                labels = labels.to(device)

            outputs = net(images, scales)

            param_tloss = loss_fn(outputs, labels)
            zl, al, nl = param_tloss
            ztloss += zl.item()
            atloss += al.item()
            ntloss += nl.item()
            loss = sum(list(param_tloss))
            
            running_tloss += loss.item()
            
    if running_tloss < best_loss:
        best_loss = running_tloss
        best_state_checkpoint = {'state_dict': net.state_dict(),
                                 'optimizer' : optimizer.state_dict()}

    epoch_loss = running_loss / len(trainloader)
    epoch_tloss = running_tloss / len(testloader)
    train_loss.append(epoch_loss)
    test_loss.append(epoch_tloss)
    z_loss.append(rzloss/len(trainloader))
    a_loss.append(raloss/len(trainloader))
    n_loss.append(rnloss/len(trainloader))
    z_tloss.append(ztloss/len(testloader))
    a_tloss.append(atloss/len(testloader))
    n_tloss.append(ntloss/len(testloader))

    str_loss = '%.3f'%epoch_loss
    str_tloss = '%.3f'%epoch_tloss
    #print('preloss: {}'.format(preloss))
    print('Train loss: {}, test loss: {}'.format(str_loss, str_tloss))
    
    if (epoch+1) % checkpoint_every == 0:
        num_state_checkpoint = {'state_dict': net.state_dict(),
                                 'optimizer' : optimizer.state_dict()}
        numpath = weightsdir + 'epoch{}.pt'.format(epoch+1)
        torch.save(num_state_checkpoint, numpath)

        checkpointdata= {'epochs': np.arange(epoch+1), 'train_loss':train_loss, 'test_loss':test_loss, 'z_loss':z_loss, 'a_loss':a_loss, 'n_loss':n_loss, 'z_testloss':z_tloss, 'a_testloss':a_tloss, 'n_testloss':n_tloss}
        df = pd.DataFrame(data = checkpointdata)

        df.to_csv(train_info_path)
        print('Saved checkpoint')


last_state_checkpoint = {'state_dict': net.state_dict(),
                         'optimizer' : optimizer.state_dict()}
    
print('finished training')

bestpath = weightsdir + 'best.pt'
lastpath = weightsdir + 'last.pt'
torch.save(best_state_checkpoint, bestpath)
torch.save(last_state_checkpoint, lastpath)


df = pd.DataFrame(data = {'epochs': np.arange(epochs), 'train_loss':train_loss, 'test_loss':test_loss, 'z_loss':z_loss, 'a_loss':a_loss, 'n_loss':n_loss, 'z_testloss':z_tloss, 'a_testloss':a_tloss, 'n_testloss':n_tloss})

df.to_csv(train_info_path)
