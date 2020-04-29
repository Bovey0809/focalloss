from data import MyDataset
from network import MLP
from tqdm import tqdm
from tqdm.auto import trange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loss import FocalLoss
from loss import cross_entropy
from utils import visualize_3D
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLP()
print(model)
dataset = MyDataset()
trainloader = torch.utils.data.DataLoader(
    dataset, batch_size=128, shuffle=False)
criterion = cross_entropy
# criterion = FocalLoss(num_classes=1)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
# loop over the dataset multiple times

epochs = 50


def train():
    with tqdm(total=epochs) as pbar:
        for epoch in trange(epochs):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                model.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                for name, param in model.named_parameters():
                    writer.add_histogram(name, param.grad, i)
                running_loss += loss.item()
                pbar.set_description_str(
                    f'epoch:{epoch} iter:{i} Loss: {running_loss}')

    print('Finished Training')


if __name__ == "__main__":
    train()
