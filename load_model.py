import os
from collections import OrderedDict

import torch
from torch import nn
from torchvision.models import vgg16


def load_model(arch=vgg16, save_dir="Dec_1.pth", hidden_units=512, train_loader=None):

    if save_dir is not None and os.path.exists(save_dir):
        checkpoint = torch.load(save_dir)
        # Overwrite function input parameters
        input_size = checkpoint['input_size']
        hidden_units = checkpoint['fc2_size']
        output_size = checkpoint['output_size']
        state_dict = checkpoint['state_dict']
        class_to_idx = checkpoint['class_to_idx']
        classes = checkpoint['classes']
    else:
        input_size = 25088
        output_size = len(train_loader.dataset.classes)
        state_dict = None
        class_to_idx = train_loader.dataset.class_to_idx
        classes = train_loader.dataset.classes

    if arch == 'vgg16':
        model = vgg16(weights='VGG16_Weights.DEFAULT', progress=True, dropout=0.5)
    else:
        print("ERROR: unknown architecture")
        exit(1)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_units)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_units, output_size)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier
    model.class_to_idx = class_to_idx
    model.classes = classes
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return model
