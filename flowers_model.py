####
#### Based on:
#### - https://github.com/udacity/cd0281-Introduction-to-Neural-Networks-with-PyTorch/blob/master/deep-learning-with-pytorch/fc_model.py
#### - https://github.com/udacity/cd0281-Introduction-to-Neural-Networks-with-PyTorch/blob/master/deep-learning-with-pytorch/Part%207%20-%20Loading%20Image%20Data%20(Solution).ipynb
#### Also see https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py
####
import torch
from torch import nn, optim
from torchvision.models import vgg, get_model, get_model_weights, get_weight, list_models
from torchvision.models.vgg import vgg16, VGG

class FlowerModel(VGG):
    def __init__(self, weights='VGG16_Weights.DEFAULT', progress=True, dropout=0.5):
        # super().__init__(dropout=dropout)

        model = vgg16(weights, progress)

        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False

        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(25088, 500)),
                                ('relu', nn.ReLU()),
                                ('fc2', nn.Linear(500, 102)),
                                ('output', nn.LogSoftmax(dim=1))
                                ]))  
        model.classifier = classifier
        # return flower_model
        
    def validation(self, testloader, criterion, device):
        accuracy = 0
        test_loss = 0
        self.eval()
        for images, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = self.forward(images)
            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()
            ## Calculating the accuracy 
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        return test_loss, accuracy

    ###
    ### Based on https://github.com/udacity/cd0281-Introduction-to-Neural-Networks-with-PyTorch/blob/master/deep-learning-with-pytorch/Part%208%20-%20Transfer%20Learning%20(Solution).ipynb
    ###
    def train(self, trainloader, testloader, criterion, optimizer, epochs=5, print_every=40, device="cpu"):
        self.to(device)
        steps = 0
        running_loss = 0
        for epoch in range(epochs):
            for images, labels in trainloader:
                images, labels = images.to(device), labels.to(device)
                steps += 1
                print(steps)
                
                logps = self.forward(images)
                loss = criterion(logps, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if steps % print_every == 0:
                    self.eval()
                    with torch.no_grad():
                        test_loss, accuracy = self.validation(testloader, criterion, device)
                    print(f"Epoch {epoch+1}/{epochs}.. "
                        f"Train loss: {running_loss/print_every:.3f}.. "
                        f"Test loss: {test_loss/len(testloader):.3f}.. "
                        f"Test accuracy: {accuracy/len(testloader):.3f}")
                    running_loss = 0
                    self.train()
