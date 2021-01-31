import torch
import os 
import time
import torchvision.models as models
from torchvision import transforms, datasets
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from customdatasets import MSI_images_zip


image_transforms = {
    'train': transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'test': transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
}
'''
(from a different script)
dirs = {'train': 'C:\\Users\\guysh\\Desktop\\image_recognition\\mynetimage_recognition\\images\\crc\\train',
        'test' : 'C:\\Users\\guysh\\Desktop\\image_recognition\\mynetimage_recognition\\images\\crc\\test'
       }
# Load the datasets with ImageFolder
image_datasets: dict = {x: datasets.ImageFolder(dirs[x],   transform=data_transforms[x]) for x in ['train', 'test']}
# load the data into batches
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True, drop_last = False ) for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes
'''
# Load the Data

dataset = 'CRC-DX'

# Set train and valid directory paths

train_directory = 'C:\\Users\\guysh\\Desktop\\image_recognition\\images\\crc\\trail\\train\\'
valid_directory = 'C:\\Users\\guysh\\Desktop\\image_recognition\\images\\crc\\trail\\val\\'

data_directory = 'C:\\Users\\guysh\\Desktop\\image_recognition\\images\\crc\\trail\\'

# Batch size
bs = 32

# Number of classes

num_classes = 2
#num_classes = len(os.listdir(valid_directory))  #10#2#257
print(num_classes)

# Load Data from folders
data = {
    'train': MSI_images_zip(csv_file = train_directory+"train_csv.csv", zip_file = train_directory+'train.zip', transform=image_transforms['train']),
    'valid': MSI_images_zip(csv_file = valid_directory+"val_csv.csv", zip_file = valid_directory+'val.zip', transform=image_transforms['valid'])
}


#data = {
#    'train': cancer_images(csv_file = data_directory+"train.csv", rootdir = train_directory, transform=image_transforms['train']),
#    'valid': cancer_images(csv_file = data_directory+"val.csv", rootdir = valid_directory, transform=image_transforms['valid'])
#}

#data = {
#    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
#    'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid'])
#}

# Get a mapping of the indices to the class names, in order to see the output classes of the test images.
#idx_to_class = {v: k for k, v in data['train'].class_to_idx.items()}
#print(idx_to_class)

# Size of Data, to be used for calculating Average Loss and Accuracy
train_data_size = len(data['train'])
valid_data_size = len(data['valid'])

# Create iterators for the Data loaded using DataLoader module
train_data_loader = DataLoader(data['train'], batch_size=bs, shuffle=True)
valid_data_loader = DataLoader(data['valid'], batch_size=bs, shuffle=True)


resnet18 = models.resnet18(pretrained= True)

#freeze all parameters in the network (keep if i want to optimize only the classifier layer.
for param in resnet18.parameters():
    param.requires_grad = True

loss_func = nn.NLLLoss()
#the model parameters are registered here in the optimizer (change learning rate)
optimizer = optim.Adam(resnet18.parameters(), lr = 1e-3)

#replace the classifier layer (last layer, linear), with a new linear layer (unfrozen by default) with 2 claasses.
resnet18.fc = nn.Linear(512, 2)


def train_and_validate(model, loss_criterion, optimizer, epochs):
    '''
    Function to train and validate
    Parameters
        :param model: Model to train and validate
        :param loss_criterion: Loss Criterion to minimize
        :param optimizer: Optimizer for computing gradients
        :param epochs: Number of epochs (default=25)

    Returns
        model: Trained Model with best validation accuracy
        history: (dict object): Having training loss, accuracy and validation loss, accuracy
    '''

    start = time.time()
    history = []
    best_acc = 0.0

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))

        # Set to training mode
        model.train()

        # Loss and Accuracy within the epoch
        train_loss = 0.0
        train_acc = 0.0

        valid_loss = 0.0
        valid_acc = 0.0

        for i, (inputs, labels) in enumerate(train_data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Clean existing gradients
            optimizer.zero_grad()

            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)

            # Compute loss
            loss = loss_criterion(outputs, labels)

            # Backpropagate the gradients (backwards pass, calculate gradients for each parameter)
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Compute the total loss for the batch and add it to train_loss
            train_loss += loss.item() * inputs.size(0)

            # Compute the accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            # Compute total accuracy in the whole batch and add to train_acc
            train_acc += acc.item() * inputs.size(0)

            # print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))

        # Validation - No gradient tracking needed
        with torch.no_grad():

            # Set to evaluation mode
            model.eval()

            # Validation loop
            for j, (inputs, labels) in enumerate(valid_data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass - compute outputs on input data using the model
                outputs = model(inputs)

                # Compute loss
                loss = loss_criterion(outputs, labels)

                # Compute the total loss for the batch and add it to valid_loss
                valid_loss += loss.item() * inputs.size(0)

                # Calculate validation accuracy
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                # Compute total accuracy in the whole batch and add to valid_acc
                valid_acc += acc.item() * inputs.size(0)

                # print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))

        # Find average training loss and training accuracy
        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size

        # Find average training loss and training accuracy
        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc / valid_data_size

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        epoch_end = time.time()

        print(
            "Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation : Loss : {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
                epoch + 1, avg_train_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100,
                epoch_end - epoch_start))

        # Save if the model has best accuracy till now
        # torch.save(model, dataset+'_model_'+str(epoch)+'.pt')

    return model, history


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(str(device))

#choose number of epochs
num_epochs = 5
#train the model
trained_model, history = train_and_validate(resnet18, loss_func, optimizer, num_epochs)
#save the training history
torch.save(history, dataset+'_history.pt')

history = np.array(history)
plt.plot(history[:,0:2])
plt.legend(['Tr Loss', 'Val Loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.ylim(0,1)
plt.savefig(dataset+'_loss_curve.png')
plt.show()

plt.plot(history[:,2:4])
plt.legend(['Tr Accuracy', 'Val Accuracy'])
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.ylim(0,1)
plt.savefig(dataset+'_accuracy_curve.png')
plt.show()


def predict(model, test_image_name):
    '''
    Function to predict the class of a single test image
    Parameters
        :param model: Model to test
        :param test_image_name: Test image

    '''

    transform = image_transforms['test']

    test_image = Image.open(test_image_name)
    plt.imshow(test_image)

    test_image_tensor = transform(test_image)

    if torch.cuda.is_available():
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cuda()
    else:
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224)

    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        out = model(test_image_tensor)
        ps = torch.exp(out)
        topk, topclass = ps.topk(3, dim=1)
        for i in range(3):
            print("Predcition", i + 1, ":", idx_to_class[topclass.numpy()[0][i]], ", Score: ", topk.numpy()[0][i])



#predict(trained_model, 'caltec256subset/test/009_0098.jpg')