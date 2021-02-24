import torch.optim as optim
import torch.nn as nn
from torch_entropy_arch import TorchEstimator
from Batch_Generator import makedata
import json

with open('torch_train_config.json', 'r') as f:
    config = json.load(config)

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
    z1, a1, n1 = outputs
    z2, a2, n2 = labels

    loss1 = criterion(z1, z2)
    loss2 = criterion(a1, a2)
    loss2 = criterion(n1, n2)

    return loss1 + loss2 + loss3

for epoch in range(epochs):
    for i, data in enumerate(trainloader, 0):
        optimizer.zero_grad()

        inputs, labels = data

        outputs = net(inputs)
        
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()


print('finished training')

PATH = config['training']['savefile']
torch.save(net.state_dict(), PATH)
