# importing libraries
import numpy as np
from PIL import Image
from PIL import ImageFile
import os
import argparse

# Pytorch 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets

ImageFile.LOAD_TRUNCATED_IMAGES=True

description = 'parse example'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-f', '--is_train', type=int, default=1, help='0: training, 1: testing(default:False)')
parser.add_argument('-e', '--n_epochs', type=int, default=10, help='total epochs(default: 10)')
args = parser.parse_args()

# Loading images
data_dir = 'E:/Master_Hansung_Visual_Intelligence_LAB/CNN_homework/dogImages'
train_dir = os.path.join(data_dir, 'train')
valid_dir = os.path.join(data_dir, 'valid')
test_dir = os.path.join(data_dir, 'test')
save_dir = 'E:/Master_Hansung_Visual_Intelligence_LAB/CNN_homework/dogImages/ckpt_model_transfer'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

# Define preprocessing
# Data Augmentation and Normalization on images
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])])
valid_transforms = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# ** Using the image datasets and the trainforms, define the dataloader
trainloader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=16, shuffle=False)
loader_transfer = {'train':trainloader, 'valid':validloader, 'test':testloader}
data_transfer = {'train':trainloader}

# Load pretraind model(VGG11)
model_transfer = models.vgg11(pretrained=True)
print(model_transfer)

# **Freezing the parameters
for param in model_transfer.features.parameters():
    param.requires_grad=False # Freeze: set requires_grad=False

# Changing the classifier layer
model_transfer.classifier[6] = nn.Linear(4096, 133, bias=True)

# Moving the model to GPU-RAM space
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using Device:", device)

model_transfer.to(device)
print("Model Network:", model_transfer)

# Loading the cost function
transfer_criterion = nn.CrossEntropyLoss()

# Loading Optimizer
transfer_optimizer = optim.SGD(model_transfer.parameters(), lr=1e-3, momentum=0.9)

# Trainer
def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    valid_loss_min = np.inf
    # Max Value (As the loss decreases and becomes less than this value it gets saved)
    for epoch in range(1, n_epochs + 1):
        # Initializing training variables
        train_loss = 0.0
        valid_loss = 0.0
        # Start training the model
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU's memory space (if available)
            if use_cuda==True: # 'cuda' if torch.cuda.is_available() else 'cpu'
                data, target = data.cuda(), target.cuda()
                model.to('cuda')
            else:
                data, target = data.cpu(), target.cpu()
                model.to('cpu')
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            # validate the model
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            accuracy = 0
            # move to GPU's memory space (if available)
            if use_cuda==True:
                data, target = data.cuda(), target.cuda()
            else:
                data, target = data.cpu(), target.cpu()
            ## Update the validation loss
            logps = model(data)
            loss = criterion(logps, target)
            valid_loss += ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

            # print both training and validation losses
            print('Epoch: {} \tTraining Loss: {:.4f} \tValidation Loss: {:.4f}'.format(epoch, train_loss, valid_loss))
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.4f} --> {:.4f}). Saving model ...'.format(valid_loss_min, valid_loss)) # Saving the model
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss

train(50, loader_transfer, model_transfer, transfer_optimizer, transfer_criterion, use_cuda=False, save_path=save_dir)
model_transfer.load_state_dict(torch.load(os.path.join(save_dir, 'model_transfer.pt')))

def test(loaders, model, criterion, use_cuda):
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval() # So that it doesn't change the model parameters during testing
    for batch_idx, (data, target) in enumerate(loaders['test']):
    # move to GPU's memory spave if available
        if use_cuda==True:
            data, target = data.cuda(), target.cuda()
        else:
            data, target = data.cpu(), target.cpu()
        # Passing the data to the model (Forward Pass)
        output = model(data)
        loss = criterion(output, target) # Test Loss
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # Output probabilities to the predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # Comparing the predicted class to output
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
    print('Test Loss: {:.6f}\n'.format(test_loss))
    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (100. * correct / total, correct, total))

test(loader_transfer, model_transfer, transfer_criterion, use_cuda=False)
model_transfer.load_state_dict(torch.load(os.path.join(save_dir, 'model_transfer.pt')))

def predict_breed_transfer(img_path, model, use_cuda):
    transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                    [0.229, 0.224, 0.225])])
    img = Image.open(img_path)
    img = transform(img)[:3, :, :].unsqueeze(0)
    if use_cuda==True:
        img = img.cuda()
        model.to('cuda')
    else:
        img = img.cpu()
        model.to('cpu')
    # Passing through the model
    model.eval()
    # Checking the name of class by passing the index
    class_names = [item[4:].replace("_", " ") for item in data_transfer['train'].dataset.classes]
    output = model_transfer(img)
    # Probabilities to class
    pred = output.data.max(1, keepdim=True)[1]
    print("input image: {}, \t predicted class: {}".format(img_path, class_names[pred.item()]))

predict_breed_transfer('E:/Master_Hansung_Visual_Intelligence_LAB/CNN_homework/dogImages/dog.jpeg', model_transfer, use_cuda=False)