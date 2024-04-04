from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from config import Config


config = Config()

class dataloader():
    def __init__(self, train_dir, valid_dir, test_dir):
        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.test_dir = test_dir
        self.train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                                    transforms.RandomResizedCrop(224),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])])
        self.valid_transforms = transforms.Compose([transforms.Resize(256),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])])
        self.test_transforms = transforms.Compose([transforms.Resize(256),
                                                   transforms.CenterCrop(224),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                                        [0.229, 0.224, 0.225])])
        self.train_data = datasets.ImageFolder(self.train_dir, transform=self.train_transforms)
        self.valid_data = datasets.ImageFolder(self.valid_dir, transform=self.valid_transforms)
        self.test_data = datasets.ImageFolder(self.test_dir, transform=self.test_transforms)
        self.train_loader = DataLoader(self.train_data, batch_size=config.batch_size, shuffle=True)
        self.valid_loader = DataLoader(self.valid_data, batch_size=config.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_data, batch_size=config.batch_size)
        self.loaders = {'train': self.train_loader, 'valid': self.valid_loader, 'test': self.test_loader}

