# Implementation of flower identification classifier
# Written by: Mehrnaz Siavoshi
# Date: December 14, 2020
# Developed for the Udacity AI Programming with Python Nanodegree

# Imports

import argparse
import json
import PIL
import torch
import numpy as np
from math import ceil


# Inputs

def parser():
    parser = argparse.ArgumentParser(description = 'Predict flower name from an image')
    
    # Define the image
    parser.add_argument('--image', type = str, help = 'Image pathfile', required = True)
    
    # Classifier checkpoint
    parser.add_argument('--checkpoint', type = str, help = 'Checkpoint str file', required = True)
    
    # Number of matches
    parser.add_argument('--top_k', type = int, help = 'Number of top matches')
    
    # JSON categories
    parser.add_argument('--category_names', type = str, help = 'Map category to flower name')
    
    # GPU option
    parser.add_argument('--gpu', action = 'store_true', help = 'Use GPU if available')
    
    # Combine
    args = parser.parse_args()
    
    return args

# Load checkpoint code from notebook

def model_loader(path):
    checkpoint = torch.load('checkpoint.pth')
    structure = checkpoint['structure']
    
    model = models.vgg16(pretrained = True)
    model.name = 'vgg16'
    for param in model.parameters():
        param.requires_grad = False

    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

# Image processor from notebook

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    img = PIL.Image.open(image)
    
    # image dimensions
    original_width, original_height = img.size
    
    # set size based on dimensions
    if original_width < original_height:
        size = [256, 256**600]
    else:
        size = [256**600, 256]
    img.thumbnail(size)
    
    # crop to center
    center = original_width/4, original_height/4
    left, top, right, bottom = center[0]-(244/2), center[1]-(244/2), center[0]+(244/2), center[1]+(244/2)
    img = img.crop((left, top, right, bottom))

    numpy_img = np.array(img)/255 
    
    # normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    numpy_img = (numpy_img-mean)/std
    
    numpy_img = numpy_img.transpose(2,0,1)
    
    return numpy_img

# Predictor from notebook

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    model.to('cpu') # change to cuda if enabled
    model.eval();
    
    # change following line to cuda if enabled
    torch_image = torch.from_numpy(np.expand_dims(process_image(image_path),axis=0)).type(torch.FoatTensor).to('cpu') 
    
    # probabilites 
    log_probs = model.forward(torch_image)
    linear_probs = torch.exp(log_probs)
    
    # return top k probabilities, default 5
    top_probs, top_labels = linear_probs.topk(top_k)
    
    # detach
    top_probs = np.array(top_probs.detach())[0]
    top_labels = np.array(top_labels.detach())[0]
    
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]
    
    return top_probs, top_labels, top_flowers



# MAIN 

def main():
    args = parser()
    
    with open(args.category_names, 'r') as f: 
        cat_to_name = json.load(f)
        
    model = model_loader(args.checkpoint)
    
    np_img = process_image(args.image)
    
    top_probs, top_labels, top_flowers = predict(np_img, model, topk=5)
    
    for i, j in enumerate(zip(top_flowers, top_probs)):
        print('Rank{}: '.format(i+1), 'Flower: {}, Probability: {}%'.format(j[1], ceil(j[0])))
        
# Run

if __name__ == '__main__':
    main()