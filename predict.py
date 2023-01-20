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
import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

parser = argparse.ArgumentParser()
parser.add_argument("input", help="path to the input image")
parser.add_argument("checkpoint", help="path to the model/checkpoint file")
parser.add_argument("--top_k", help="number of top K most likely classes", type=int, default=1)
parser.add_argument("--category_names", help="path to the mapping of categories to real names")
parser.add_argument("--gpu", help="flag to use GPU for inference", action="store_true")
args = parser.parse_args()

if args.gpu is True and torch.cuda.is_available() is True:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer

model, optimizer = load_checkpoint(args.checkpoint)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    width, height = image.size
    if width < height:
        image.thumbnail((256, height))
    else:
        image.thumbnail((width, 256))
        

    width, height = image.size
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2
    image = image.crop((left, top, right, bottom))
    
    np_image = np.array(image)
    np_image = np_image / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

def predict(image_path, model, topk=args.top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    img = Image.open(image_path)
    img = process_image(img)
    img = torch.from_numpy(img).type(torch.FloatTensor)
    img = img.unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = img.to(device)
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        logits = model(img)
        probs = torch.exp(logits)
        top_probs, top_classes = probs.topk(topk)
        top_probs = top_probs.tolist()[0]
        top_classes = top_classes.tolist()[0]
        
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[x] for x in top_classes]
    return top_probs, top_classes

def show_predict_image(image_path, model, topk=args.top_k):
    probs, classes = predict(image_path, model, topk)
    image = Image.open(image_path)
    class_names = [cat_to_name[x] for x in classes]
    plt.imshow(image)
    plt.axis('off')
    plt.title(class_names[0])
    plt.show()
    y_pos = np.arange(len(class_names))
    plt.barh(y_pos, probs)
    plt.yticks(y_pos, class_names)
    plt.xlabel('Probability')
    plt.show()
    

print(predict(args.input,model))
show_predict_image(args.input,model)