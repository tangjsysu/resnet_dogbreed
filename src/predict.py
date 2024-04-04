import torch
from preprocessor import dataloader
from model import resnet50
from config import Config
from tqdm import tqdm
config = Config()
dataloader = dataloader(train_dir='../train_valid_test/train',test_dir='../train_valid_test/test',valid_dir='../train_valid_test/valid')
if torch.cuda.is_available():
    device = 'cuda'

def predict():
    # Initialize the model
    model = resnet50()  # replace this with your actual model architecture

    # Load the state dictionary
    state_dict = torch.load('./model.pt')

    # Load the state dictionary into the model
    model.load_state_dict(state_dict)
    model = model.to(device)
    # Set the model to evaluation mode
    model.eval()
    total = 0
    correct = 0
    for data, target in tqdm(dataloader.loaders['test']):
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    test_accuracy = correct / total
    print(f'Test accuracy: {test_accuracy}')

if __name__ == '__main__':
    predict()