import torchvision.models as models
import torch.nn as nn


class resnet50(nn.Module):
    def __init__(self):
        super(resnet50, self).__init__()
        self.model = models.resnet50(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 133)

    def forward(self, x):
        outputs = self.model(x)
        return outputs