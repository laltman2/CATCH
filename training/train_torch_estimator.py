import torch.optim as optim
import torch.nn as nn
from torch_estimator_arch import TorchEstimator
from Torch_DataLoader import makedata, EstimatorDataset
import torch
import json

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
trainloader = torch.utils.data.DataLoader(train_set, batch_size=2)


for epoch in range(epochs):
    print('Epoch {}/{}'.format(epoch, epochs))
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        optimizer.zero_grad()

        images, scales, labels = data

        outputs = net(images, scales)
        
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()



print('finished training')

PATH = config['training']['savefile'] + '.pt'
torch.save(net.state_dict(), PATH)
