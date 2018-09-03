import torch
import torchvision.models as models
import torchvision.transforms
import torchvision.transforms as transforms
from PIL import Image
import sys
import numpy as np

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def image_loader(path):
    loader = transforms.Compose([transforms.Resize(256),
                    transforms.CenterCrop(224), transforms.ToTensor()])
    image = Image.open(path)
    image = loader(image).float()
    image = image.unsqueeze(0)
    image = torch.autograd.Variable(image, requires_grad=True)
    return image


model = models.resnet18(pretrained=True)
#model = torch.nn.DataParallel(model)

model.eval()

image_name = sys.argv[1]
image = image_loader(image_name)

y_predict = model(image)
print( np.argmax(y_predict.detach().numpy()))


