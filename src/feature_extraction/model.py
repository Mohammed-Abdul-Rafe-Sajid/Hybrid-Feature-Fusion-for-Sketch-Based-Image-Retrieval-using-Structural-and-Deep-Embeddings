#helper file
import torch
from torchvision.models import resnet18, ResNet18_Weights

def get_model():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    return model