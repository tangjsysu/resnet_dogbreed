from pylab import mpl
import cv2
import os
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import warnings
import numpy as np
from model import resnet50

from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

fontC = ImageFont.truetype('C:/Windows/Fonts/Arial.ttf', 25)

warnings.filterwarnings("ignore")

def drawText(image, addText, x1, y1):

    # mean = np.mean(image[:50, :50], axis=0)
    # color = 255 - np.mean(mean, axis=0).astype(np.int)
    # color = tuple(color)
    color = (255, 202, 97)
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    draw.text((x1, y1),
              addText.encode("utf-8").decode("utf-8"),
              color, font=fontC)
    imagex = np.array(img)

    return imagex


class Classifier(object):

    def __init__(self):

        self.model = resnet50()
        self.model.load_state_dict(torch.load('./model.pt'))
        self.model.eval()
        self.loader = transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.485, 0.456, 0.406),
                                                               (0.229, 0.224, 0.225))])

    def test(self, pt, id2label):

        image = Image.open(pt).convert("RGB")
        image_tensor = self.loader(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        inputs = Variable(image_tensor)

        output = self.model(inputs)
        softmax = nn.Softmax(dim=1)
        preds = softmax(output)
        top_preds = torch.topk(preds, 3)
        top_pre = top_preds[1][0].tolist()
        pred_breeds = [id2label[i] for i in top_pre]
        confidence = top_preds[0][0]

        result = self.DrawResultv2(pred_breeds, confidence, pt)

        return result

    def DrawResultv2(self, breeds, confidence, pt, size=512):
        title = "Predicted breed (confidence):\n"
        for breed, conf in zip(breeds, confidence):
            if conf > 0.005:
                title += "  - {} ({:.0f}%)\n".format(breed, conf*100)
        image = cv2.imdecode(np.fromfile(pt, dtype=np.uint8), 1)
        w = min(image.shape[:2])
        ratio = size / w
        image = cv2.resize(image, None, fx=ratio, fy=ratio)
        image = drawText(image, title, 0, 10)

        print(title)
        return image

def get_subdirectories(directory_path):
    return [name for name in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, name))]
if __name__ == '__main__':

    labels = get_subdirectories('../train_valid_test/test')
    id2label = {i: label for i, label in enumerate(labels)}

    pt = './img.png'
    net = Classifier()
    result = net.test(pt,id2label)
    cv2.imshow('a', result)
    cv2.waitKey(0)

    cv2.destroyAllWindows()