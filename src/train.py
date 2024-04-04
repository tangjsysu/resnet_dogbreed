# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import ImageFile
from preprocessor import dataloader
from model import resnet50
from config import Config
from tqdm import tqdm
config = Config()
from torch.utils.tensorboard import SummaryWriter

ImageFile.LOAD_TRUNCATED_IMAGES = True
writer = SummaryWriter(os.path.join("./tb_log/eval"), comment="resnet_dog")
if torch.cuda.is_available():
    device = 'cuda'

model = resnet50()
dataloader = dataloader(train_dir='../train_valid_test/train',test_dir='../train_valid_test/test',valid_dir='../train_valid_test/valid')


def train():
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    epochs = config.epochs
    valid_loss_min = np.Inf
    for epoch in tqdm(range(epochs)):
        train_loss = 0.0
        valid_loss = 0.0
        correct = 0
        total = 0
        model.train()
        for data, target in dataloader.loaders['train']:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
        model.eval()
        for data, target in dataloader.loaders['valid']:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        train_loss = train_loss / len(dataloader.loaders['train'].dataset)
        valid_loss = valid_loss / len(dataloader.loaders['valid'].dataset)
        accuracy = correct / total
        writer.add_scalar("Train/loss", train_loss, epoch)
        writer.add_scalar("Valid/loss", valid_loss, epoch)
        writer.add_scalar("Valid/accuracy", accuracy, epoch)  # 记录准确率
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tValidation Accuracy: {:.6f}'.format(epoch,
                                                                                                                 train_loss,
                                                                                                                 valid_loss,
                                                                                                                 accuracy))
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            torch.save(model.state_dict(), 'model.pt')
            valid_loss_min = valid_loss
    writer.close()

if __name__ == '__main__':
    train()