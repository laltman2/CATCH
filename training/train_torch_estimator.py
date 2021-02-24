import torch.optim as optim
import torch.nn as nn
from torch_entropy_arch import TorchEstimator
import json

with open('torch_train_config.json', 'r') as f:
    config = json.load(config)

#make data here

#define data loaders here

net = TorchEstimator()
net.train()
optimizer = optim.RMSprop(net.parameters())
criterion = nn.MSELoss() 

epochs = config['epochs']

#define loss function

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

PATH = config['save_path']
torch.save(net.state_dict(), PATH)
