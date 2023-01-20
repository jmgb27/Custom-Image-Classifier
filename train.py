# Imports here
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("dataset_folder", help="directory containing the training data")
parser.add_argument("--save_dir", help="directory to save checkpoints", default=None)
parser.add_argument("--arch", help="model architecture", default="vgg16",choices=['vgg16', 'resnet18'])
parser.add_argument("--learning_rate", type=float, help="learning rate", default=0.001)
parser.add_argument("--hidden_units", type=int, help="number of hidden units", default=4096)
parser.add_argument("--epochs", type=int, help="number of training epochs", default=1)
parser.add_argument("--gpu", help="flag to use GPU for training", action="store_true")
args = parser.parse_args()

if args.arch == 'resnet18':
    model = models.resnet18(pretrained=True)
elif args.arch == 'vgg16':
    model = models.vgg16(pretrained=True)

if args.gpu is True and torch.cuda.is_available() is True:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

data_dir = args.dataset_folder
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
data_transforms = {
'train': transforms.Compose([
transforms.RandomResizedCrop(224),
transforms.RandomHorizontalFlip(),
transforms.ToTensor(),
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
'valid': transforms.Compose([
transforms.Resize(256),
transforms.CenterCrop(224),
transforms.ToTensor(),
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
'test': transforms.Compose([
transforms.Resize(256),
transforms.CenterCrop(224),
transforms.ToTensor(),
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
}

# TODO: Load the datasets with ImageFolder
image_datasets = {
'train': datasets.ImageFolder(train_dir, data_transforms['train']),
'valid': datasets.ImageFolder(valid_dir, data_transforms['valid']),
'test': datasets.ImageFolder(test_dir, data_transforms['test'])
}

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {
x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True)
for x in ['train', 'valid', 'test']
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
class_names = image_datasets['train'].classes

print(f"Number of images in each dataset: {dataset_sizes}")
print(f"Classes: {class_names}")

# Freeze the parameters of the pre-trained network
for param in model.parameters():
    param.requiresGrad = False

# Define a new classifier
classifier = nn.Sequential(nn.Linear(25088, args.hidden_units),
                           nn.ReLU(),
                           nn.Dropout(p=0.5),
                           nn.Linear(args.hidden_units, 102),
                           nn.LogSoftmax(dim=1))

# Replace the pre-trained classifier with the new one
model.classifier = classifier
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)


# Train the classifier layers using backpropagation
epochs = args.epochs
steps = 0
running_loss = 0
print_every = 5

for epoch in range(epochs):
    for inputs, labels in dataloaders['train']:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in dataloaders['valid']:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logits = model(inputs)
                    valid_loss += criterion(logits, labels).item()

                    probs = torch.exp(logits)
                    top_probs, top_labels = probs.topk(1, dim=1)
                    equals = top_labels == labels.view(*top_labels.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {valid_loss/len(dataloaders['valid']):.3f}.. "
                  f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}")
            running_loss = 0
            model.train()

# TODO: Do validation on the test set
model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for inputs, labels in dataloaders['test']:
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs)
        loss = criterion(logits, labels)
        test_loss += loss.item()

        _, preds = torch.max(logits, 1)
        correct += torch.sum(preds == labels.data)

test_loss /= dataset_sizes['test']
accuracy = correct.double() / dataset_sizes['test']
print(f'Test Loss: {test_loss:.4f} Test Accuracy: {accuracy:.4f}')

# TODO: Save the checkpoint 
checkpoint = {'classifier': model.classifier,
              'model': model,
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict(),
              'class_to_idx': image_datasets['train'].class_to_idx}

torch.save(checkpoint, 'model.pth')