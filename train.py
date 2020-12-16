# Implementation of flower identification classifier
# Written by: Mehrnaz Siavoshi
# Date: December 15, 2020
# Developed for the Udacity AI Programming with Python Nanodegree

# Imports
import argparse
import torch
from collections import OrderedDict
from torch import nn
from torch import optim
from torchvision import datasets, transforms
import torchvision.models as models

# Inputs

def parser():
    parser = argparse.ArgumentParser(description = 'Predict flower name from an image - Trainer')

    # Directories
    parser.add_argument('data_dir', type = str, help = 'Data directory', default = './flowers/')
    parser.add_argument('--save_dir', type = str, help = 'Checkpoint directory', default = './checkpoint.pth')

    # Trainer settings
    parser.add_argument('--learning_rate', type = float, help = 'Define learning rate, default 0.001', default = 0.001)
    parser.add_argument('--epochs', type = int, help = 'Number of training epochs, default 8', default = 4)
    parser.add_argument('--arch', type = str, help = 'Architecture, default vgg16', default = 'vgg16')
    parser.add_argument('--hidden_units', type = int, help = 'Hidden units value, default 256', default = 256)
    parser.add_argument('--gpu', action = 'store_true', help = 'Use GPU? default True', default = True)

    # Combine
    args = parser.parse_args()

    return args

# Data transformations from notebook

def train_data_setup(train_dir):
    # Transform data
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    # Load data
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    return train_data

def test_data_setup(test_dir):
    # Transform data
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    # Load data
    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)
    return test_data

# Load data from notebook - run twice for train and test data

def load_data(data):
    loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle = True)
    return loader

# Load model and set parameters

def load_model(architecture = 'vgg16'):
    for param in model.parameters():
        param.requires_grad = False
    return model

def classifier_params(model, hidden_units):
    # Set default hidden_units
    if type(hidden_units) == type(None):
        hidden_units = 256

    in_feat = model.classifier[0].in_features

    classifier = nn.Sequential(OrderedDict([
                          ('f1', nn.Linear(in_feat, hidden_units, bias=True)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.5)), # 50% chance of drop
                          ('f2', nn.Linear(hidden_units, 102, bias=True)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    return classifier

# Model trainer

def trainer(model, trainloader, testloader, criteria, optimze, epochs, print_every, steps):
    # Load default epochs
    if type(epochs) == type(None):
        epochs = 4

    for e in range(epochs):
        counter = 0
        step = 0

        model.train()

        train_loss = 0.0
        train_accuracy = 0.0
        valid_loss = 0.0
        valid_accuracy = 0.0

        for ii, (inputs,labels) in enumerate(trainloader):
            step += 1
            inputs = inputs.to('cuda') # or 'cpu' if enabled
            labels = labels.to('cuda') # or 'cpu' if enabled

            # clean existing
            optimizer.zero_grad()

            # compute outputs
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            # only going to print every n iterations to track progress
            if step % print_every == 0:

                # total loss
                train_loss += loss.item() * inputs.size(0)

                # accuracy
                ret, predictions = torch.max(outputs.data, 1)
                correct_count = predictions.eq(labels.data.view_as(predictions))

                accuracy = torch.mean(correct_count.type(torch.FloatTensor))

                train_accuracy += accuracy.item() * inputs.size(0)

                print("Batch no: {:03d}, Loss on training: {:.4f}, Model accuracy: {:.4f}".format(i, loss.item(), accuracy.item()))

    return model

# Validate model

def validator(model, testloader):
    pred_correct = 0
    pred_total = 0

    with torch.no_grad():
        model.eval()
        for image in trainloader:
            images, labels = image
            images = images.to('cuda') # or 'cpu' if enabled
            labels = labels.to('cuda') # or 'cpu' if enabled
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            pred_correct += (predicted == labels).sum().item()
            pred_total += labels.size(0)

    print("The model accuracy is: %d%%" % (pred_correct * 100 / pred_total))


# Save a Checkpoint

def checkpoint_save(model, save_dir, train_data):
    model.class_to_idx = train_data.class_to_idx
    model.cuda # change to cpu if enabled

    torch.save({'architecture' :'vgg16',
            'classifier' : model.classifier,
             'state_dict':model.state_dict(),
             'class_to_idx':model.class_to_idx,},
             'checkpoint.pth')


# MAIN

def main():
    args = parser()

    # Define directories

    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    test_dir = data_dir + '/test'
    valid_dir = data_dir + '/valid'

    # Transform data

    train_data = train_data_setup(train_dir)
    test_data = test_data_setup(test_dir)
    valid_data = test_data_setup(valid_dir)

    trainloader = load_data(train_data)
    testloader = load_data(test_data)
    validloader = load_data(valid_data)

    # Load model

    model = load_model()
    model.classifier = classifier_params(model, hidden_units = args.hidden_units)

    # CPU/CUDA

    processor = check_gpu(gpu_arg=args.gpu)
    model.to(processor)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), args.learning_rate)

    print_every = 8
    steps = 0

    # Train, Validate, Save model
    trainedmodel = trainer(model, trainloader, testloader, criteria, optimze, args.epochs, print_every, steps)
    validator(model, testloader)
    checkpoint_save(model, args.save_dir, train_data)


# Run
if __name__ == '__main__':
    main()
