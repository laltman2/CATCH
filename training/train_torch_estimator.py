import torch.optim as optim
import torch.nn as nn
from torch_estimator_arch import TorchEstimator
from Torch_DataLoader import makedata, EstimatorDataset
import torch
import json
import pandas as pd
import numpy as np

with open('torch_train_config.json', 'r') as f:
    config = json.load(f)

#make data here
makedata(config)

#define data loaders here

net = TorchEstimator()
net.train()
optimizer = optim.RMSprop(net.parameters())
criterion = nn.MSELoss() 

epochs = config['training']['epochs']

#define loss function
def loss_fn(outputs, labels):
    z1, a1, n1 = torch.transpose(outputs,0,1)
    z2, a2, n2 = torch.transpose(labels,0,1)

    loss1 = criterion(z1, z2)
    loss2 = criterion(a1, a2)
    loss3 = criterion(n1, n2)

    return loss1 + loss2 + loss3

train_set = EstimatorDataset(config, settype='train')
trainloader = torch.utils.data.DataLoader(train_set, batch_size=config['training']['batchsize'])

test_set = EstimatorDataset(config, settype='test')
testloader = torch.utils.data.DataLoader(test_set, batch_size=config['training']['batchsize'])


train_loss = []
test_loss = []
for epoch in range(epochs):
    print('Epoch {}/{}'.format(epoch, epochs))

    net.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        optimizer.zero_grad()

        images, scales, labels = data

        outputs = net(images, scales)
        
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    net.eval()
    running_tloss = 0.0
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            images, scales, labels = data

            outputs = net(images, scales)

            loss = loss_fn(outputs, labels)
            running_tloss += loss.item()

    epoch_loss = running_loss / len(trainloader)
    epoch_tloss = running_tloss / len(testloader)

    train_loss.append(epoch_loss)
    test_loss.append(epoch_tloss)

    print('Train loss: {}, test loss: {}'.format(epoch_loss, epoch_tloss))


print('finished training')

PATH = config['training']['savefile'] + '.pt'
torch.save(net.state_dict(), PATH)

cfg_path = config['training']['savefile'] +'.json'
with open(cfg_path, 'w') as f:
    json.dump(config, f)

df = pd.DataFrame(data = {'epochs': np.arange(epochs), 'train_loss':train_loss, 'test_loss':test_loss})
train_info_path = config['training']['savefile']+'_train_info.csv'
df.to_csv(train_info_path)
