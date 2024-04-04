import torch
from model import resnet50
import pandas as pd
import os
from PIL import Image
from torchvision import transforms



class classifier():
    def __init__(self):
        self.model = resnet50()
        self.model.load_state_dict(torch.load('./model.pt'))
        self.transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, data_path):
        self.model.eval()
        data = Image.open(data_path).convert("RGB")
        data = self.transform(data).unsqueeze(0)
        output = self.model(data)
        _, predicted = torch.max(output, 1)
        return predicted

def get_subdirectories(directory_path):
    return [name for name in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, name))]

if __name__ == '__main__':
    labels = get_subdirectories('../train_valid_test/test')
    id2label = {i: label for i, label in enumerate(labels)}
    aclassifier = classifier()
    data_path = './img.png'
    predicted = aclassifier.predict(data_path)
    predict_label = id2label[predicted.item()]
    print("This is a", predict_label)
