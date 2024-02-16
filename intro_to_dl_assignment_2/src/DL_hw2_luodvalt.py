import torch
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
os.chdir('C:/Users/julia/OneDrive/Tiedostot/Opiskelu/DeepLearning/intro_to_dl_assignment_2/')

#--- hyperparameters ---
N_EPOCHS = 20
BATCH_SIZE_TRAIN = 100
BATCH_SIZE_TEST = 100
LR = 0.01
#--- fixed constants ---
NUM_CLASSES = 24
DATA_DIR = './data/sign_mnist_%s'



# --- Dataset initialization ---

# We transform image files' contents to tensors
# Plus, we can add random transformations to the training data if we like
# Think on what kind of transformations may be meaningful for this data.
# Eg., horizontal-flip is definitely a bad idea for sign language data.
# You can use another transformation here if you find a better one.
train_transform = transforms.Compose([
                                        #transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor()])
test_transform = transforms.Compose([transforms.ToTensor()])

train_set = datasets.ImageFolder(DATA_DIR % 'train', transform=train_transform)
dev_set   = datasets.ImageFolder(DATA_DIR % 'dev',   transform=test_transform)
test_set  = datasets.ImageFolder(DATA_DIR % 'test',  transform=test_transform)


# Create Pytorch data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
dev_loader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=BATCH_SIZE_TEST, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE_TEST, shuffle=False)

#--- model ---
class CNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2) # 28*28 -> 14*14
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2) # 14*14 -> 7*7
        # self.fc1 = nn.Linear(64*14*14, 196)
        self.fc1 = nn.Linear(128*7*7, 196)
        self.fc2 = nn.Linear(196, num_classes)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(x, dim=1)
        return x




#--- set up ---
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = CNN().to(device)

loss_function  = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
# optimizer = optim.SGD(model.parameters(), lr=0.05)                                # test accuracy 89.3%, 10 epochs, no regularization
# optimizer = optim.SGD(model.parameters(), lr=0.01)                                # test accuracy 59%, 8 epochs, no regularization
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))     # test accuracy 11.4%, 8 epochs, no regularization
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)                  # test accuracy 65%, 8 epochs, no regularization
# optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)                  # test accuracy 86.2%, 8 epochs, no regularization
# optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)                 # test accuracy 84.2%, 20 epochs, early stopping regularization (stopped after 6 epochs)
# optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.9)                 # test accuracy 77.6%, 20 epochs, early stopping regularization (stopped after 13 epochs)
# optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.9)                 # test accuracy 85.2%, 20 epochs, dropout and early stopping regularization (stopped after 19 epochs)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)                  # test accuracy %, 20 epochs, dropout and early stopping regularization (stopped after  epochs)
# optimizer = optim.SGD(model.parameters(), lr=0.01)                                # test accuracy %, 20 epochs, early stopping regularization (stopped after  epochs)
# optimizer = optim.SGD(model.parameters(), lr=0.01)                                # test accuracy %, 20 epochs, dropout regularization
optimizer_adagrad = torch.optim.Adagrad(model.parameters(), lr=LR)


#--- training ---
last_error = list()
for epoch in range(N_EPOCHS):
    train_loss = 0
    train_correct = 0
    total = 0

    for batch_num, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        
        loss = loss_function(outputs, target)
        loss.backward()
        optimizer.step()

        predicted = torch.zeros(target.size(dim=0))
        for row_num, data in enumerate(outputs):
            predicted[row_num] = torch.argmax(torch.exp(data)).item()

        train_loss += loss.item()
        train_correct += (target == predicted).sum()
        total += BATCH_SIZE_TRAIN


        print('Training: Epoch %d - Batch %d/%d: Loss: %.4f | Train Acc: %.3f%% (%d/%d)' % 
              (epoch + 1, batch_num + 1, len(train_loader), train_loss / (batch_num + 1), 
               100. * train_correct / total, train_correct, total))
    
    dev_loss = 0
    dev_correct = 0
    dev_total = 0
    with torch.no_grad():
        for batch_num, (data, target) in enumerate(dev_loader):
            data, target = data.to(device), target.to(device)

            outputs = model(data)
            loss = loss_function(outputs, target)

            predicted = torch.zeros(target.size(dim=0))
            for row_num, data in enumerate(outputs):
                predicted[row_num] = torch.argmax(torch.exp(data)).item()

            dev_loss += loss.item()
            dev_correct += (target == predicted).sum()
            dev_total += BATCH_SIZE_TEST
            dev_error = 1 - dev_correct / dev_total

            print('Validating: Batch %d/%d: Loss: %.4f | Dev Acc: %.3f%% (%d/%d)' % 
                (batch_num + 1, len(dev_loader), dev_loss / (batch_num + 1), 
                100. * dev_correct / dev_total, dev_correct, dev_total))
        
        if epoch > 1 and last_error[epoch-1] < dev_error and last_error[epoch-2] < dev_error:
            break
        else:
            last_error.append(dev_error)


#--- test ---
test_loss = 0
test_correct = 0
total = 0

with torch.no_grad():
    for batch_num, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)

        outputs = model.forward(data)
        loss = loss_function(outputs, target)

        predicted = torch.zeros(target.size(dim=0))
        for row_num, data in enumerate(outputs):
            predicted[row_num] = torch.argmax(torch.exp(data)).item()

        test_loss += loss.item()
        test_correct += (target == predicted).sum()
        total += BATCH_SIZE_TEST

        print('Evaluating: Batch %d/%d: Loss: %.4f | Test Acc: %.3f%% (%d/%d)' % 
              (batch_num + 1, len(test_loader), test_loss / (batch_num + 1), 
               100. * test_correct / total, test_correct, total))

