####
#### Based on:
#### - https://github.com/udacity/cd0281-Introduction-to-Neural-Networks-with-PyTorch/blob/master/deep-learning-with-pytorch/Part%208%20-%20Transfer%20Learning%20(Solution).ipynb
#### - https://github.com/udacity/cd0281-Introduction-to-Neural-Networks-with-PyTorch/blob/master/deep-learning-with-pytorch/Part%207%20-%20Loading%20Image%20Data%20(Solution).ipynb
####
#### Also see https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py
####

import torch
from torch import nn, optim
from torchvision import datasets, transforms
import argparse
from load_model import load_model

def train(model, train_loader, validation_loader, criterion, optimizer, epochs=5, print_every=40, device="cpu"):
    model.to(device)
    model.train()
    steps = 0
    running_loss = 0
    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            steps += 1
            print(steps, len(images.data), len(labels.data))

            logps = model.forward(images)
            loss = criterion(logps, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                validation_loss, accuracy = evaluate(model, validation_loader, criterion, device)
                print(f"Epoch {epoch + 1}/{epochs}.. "
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"Validation loss: {validation_loss:.3f}.. "
                      f"Test accuracy: {accuracy:.3f}")
                running_loss = 0
                model.train()


def evaluate(model, dataloader, criterion, device):
    accuracy = 0
    test_loss = 0
    model.eval()
    model.to(device)
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            # print(images.shape)
            logps = model.forward(images)
            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()
            ## Calculating the accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    return test_loss / len(dataloader), accuracy / len(dataloader)

def save_checkpoint(filepath, model):
    checkpoint = {'class_to_idx': model.class_to_idx,
                  'classes': model.classes,
                  'input_size': model.classifier.fc1.in_features,
                  'fc2_size': model.classifier.fc2.in_features,
                  'output_size': model.classifier.fc2.out_features,
                  'state_dict': model.state_dict()}
    torch.save(checkpoint, filepath)


def init_data_loading(data_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    validation_transforms = transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    validation_data = datasets.ImageFolder(data_dir + '/valid', transform=validation_transforms)
    print("training size =", len(train_data.samples), "validation size =", len(validation_data.samples))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=64)
    return train_loader, validation_loader

def main(
        data_dir="args.data_directory",
        save_dir="checkpoints",
        arch="vgg16",
        learning_rate=.003,
        hidden_units=512,
        epochs=5,
        device="cpu",
        print_every = 40):

    train_loader, validation_loader = init_data_loading(data_dir)
    model = load_model(arch, save_dir, hidden_units, train_loader=train_loader)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train(model, train_loader, validation_loader, criterion, optimizer, epochs=epochs, print_every=print_every, device=device)

    if save_dir is not None:
        save_checkpoint(save_dir, model)

###
### Ref: https://docs.python.org/3/howto/argparse.html
###
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_directory", help="Directory containing train/{class}/example", default="flowers")
    parser.add_argument("--save_dir", help="checkpoint file", default=None)
    parser.add_argument("--arch", help="Example: vgg13", default="vgg16")
    parser.add_argument("--learning_rate", default=0.003, type=float)
    parser.add_argument("--hidden_units", default=512, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--print_every", default=40, type=int)
    parser.add_argument("--gpu")
    args = parser.parse_args()

    if args.gpu:
        device = "cuda:0"
    else:
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Calling main()")
    main(
        data_dir=args.data_directory,
        save_dir=args.save_dir,
        arch=args.arch,
        learning_rate=args.learning_rate,
        hidden_units=args.hidden_units,
        epochs=args.epochs,
        device=device,
        print_every=args.print_every
    )