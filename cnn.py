# REFEFERENCES
# The code is partly adapted from pytorch tutorials, including https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

# ---- hyper-parameters ----
MAX_EPOCH = 100
MAX_LOSS = 5000
batch_size = 16
# we tried these hyper-parameters
# num_of_kernels = [2, 4, 8, 16]
# learning_rates = [1000, 0.1, 0.01, 0.0001]
# conv_layer_numbers = [1, 2, 4]
conv_layer_numbers = [2]
num_of_kernels = [16]
learning_rates = [0.1]


# --- imports ---
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import hw3utils

# ---- options ----

DEVICE_ID = 'cuda' if torch.cuda.is_available() else 'cpu'  # set to 'cpu' for cpu, 'cuda' /'cuda:0' or similar for gpu.
LOG_DIR = 'checkpoints'
VISUALIZE = False  # set True to visualize input, prediction and the output from the last batch
LOAD_CHKPT = False

torch.multiprocessing.set_start_method('spawn', force=True)


# ---- utility functions -----
def get_loaders(batch_size, device):
    data_root = 'ceng483-f22-hw3-dataset'
    train_set = hw3utils.HW3ImageFolder(root=os.path.join(data_root, 'train'), device=device)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_set = hw3utils.HW3ImageFolder(root=os.path.join(data_root, 'val'), device=device)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


# ---- ConvNet -----
class Net(nn.Module):
    def __init__(self, conv_layer_number=2, num_of_kernel=2, tanh=False, batch=False):
        super(Net, self).__init__()
        self.layer_number = conv_layer_number
        self.tanh = tanh
        self.batch = batch
        self.activation = nn.ReLU()

        padding = 1
        if conv_layer_number == 1:
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=padding)
            if self.batch:
                self.batchnorm = nn.BatchNorm2d(3)
        if conv_layer_number == 2:
            self.conv1 = nn.Conv2d(1, num_of_kernel, 3, padding=padding)
            self.conv2 = nn.Conv2d(num_of_kernel, 3, 3, padding=padding)
            if self.batch:
                self.batchnorm = nn.BatchNorm2d(num_of_kernel)
                self.batchnorm_2 = nn.BatchNorm2d(3)
        if conv_layer_number == 4:
            self.conv1 = nn.Conv2d(1, num_of_kernel, 3, padding=padding)
            self.conv2 = nn.Conv2d(num_of_kernel, num_of_kernel, 3, padding=padding)
            self.conv3 = nn.Conv2d(num_of_kernel, num_of_kernel, 3, padding=padding)
            self.conv4 = nn.Conv2d(num_of_kernel, 3, 3, padding=padding)
            if self.batch:
                self.batchnorm = nn.BatchNorm2d(num_of_kernel)
                self.batchnorm_2 = nn.BatchNorm2d(num_of_kernel)
                self.batchnorm_3 = nn.BatchNorm2d(num_of_kernel)
                self.batchnorm_4 = nn.BatchNorm2d(3)

    def forward(self, grayscale_image):
        x = self.conv1(grayscale_image)
        if self.batch:
            x = self.batchnorm(x)
        if self.layer_number == 2:
            x = self.activation(x)
            x = self.conv2(x)
            if self.batch:
                x = self.batchnorm_2(x)
        elif self.layer_number == 4:
            if self.batch:
                x = self.activation(x)
                x = self.conv2(x)
                x = self.batchnorm_2(x)
                x = self.activation(x)
                x = self.conv3(x)
                x = self.batchnorm_3(x)
                x = self.activation(x)
                x = self.conv4(x)
                x = self.batchnorm_4(x)
            else:
                x = self.activation(x)
                x = self.conv2(x)
                x = self.activation(x)
                x = self.conv3(x)
                x = self.activation(x)
                x = self.conv4(x)
        if self.tanh:
            tanh = nn.Tanh()
            x = tanh(x)
        return x

    # ---- training code -----
    def run(self, **parameters):
        training_losses = []  # for plotting training losses
        validation_losses = []  # for plotting validation losses
        accuracies = []  # for plotting validation accuracies
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        train_loader, val_loader = get_loaders(batch_size, device)

        if LOAD_CHKPT:
            print('loading the model from the checkpoint')
            self.load_state_dict(os.path.join(LOG_DIR, 'checkpoint.pt'))

        print('training begins')
        last_validation_loss = MAX_LOSS
        for epoch in range(MAX_EPOCH):
            running_loss = 0.0  # training loss of the network
            training_loss = 0.0
            for iteri, data in enumerate(train_loader, 0):
                inputs, targets = data  # inputs: low-resolution images, targets: high-resolution images.

                optimizer.zero_grad()  # zero the parameter gradients

                # do forward, backward, SGD step
                preds = self(inputs)
                loss = criterion(preds, targets)
                loss.backward()
                optimizer.step()

                # print loss
                running_loss += loss.item()
                training_loss += loss.item()
                print_n = 100  # feel free to change this constant
                if iteri % print_n == (print_n - 1):  # print every print_n mini-batches
                    print('[%d, %5d] network-loss: %.3f' %
                          (epoch + 1, iteri + 1, running_loss / 100))
                    running_loss = 0.0

                if (iteri == 0) and VISUALIZE:
                    hw3utils.visualize_batch(inputs, preds, targets)
            training_losses.append(training_loss/len(train_loader))

            name = f"kr{num_of_kernel}-conv{conv_layer_number}-lr{learning_rate}"
            if self.batch:
                name += "-batch"
            if self.tanh:
                name += "-tanh"

            print(f"Saving the model, end of epoch {epoch + 1}")
            if not os.path.exists(LOG_DIR):
                os.makedirs(LOG_DIR)
            torch.save(self.state_dict(), os.path.join(LOG_DIR, name+'.pt'))
            hw3utils.visualize_batch(inputs, preds, targets, os.path.join(LOG_DIR, name+'.png'))

            validation_loss = 0.0  # validation loss
            acc = 0.0
            for inputs, targets in val_loader:
                predictions = self(inputs)
                validation_loss += criterion(predictions, targets).item()
                cur_acc = (np.abs(targets.detach().numpy() - predictions.detach().numpy()) < 12/128).sum() / 19200
                acc += cur_acc/len(inputs)
            validation_losses.append(validation_loss/len(val_loader))
            accuracies.append(acc/len(val_loader))
            if (epoch + 1) % 6 == 0:  # checking validation loss change in every 6 epoch
                print("Calculating the validation loss")
                print(f'Validation loss: {validation_loss / 100:.4f}')

                if validation_loss < 0.5 or last_validation_loss <= validation_loss + 0.0001:
                    break

                last_validation_loss = validation_loss

        print('Finished Training')
        plot_array(training_losses, "training_losses", validation_losses, "validation_losses")
        plot_array(accuracies, "accuracies")


def plot_array(array, label1, array2=None, label2=None):
    plt.close()
    plt.plot(array, label=label1)
    if array2 is not None:
        plt.plot(array2, label=label2)
    plt.legend()
    plt.show()


device = torch.device(DEVICE_ID)
print('device: ' + str(device))

for num_of_kernel in num_of_kernels:
    for learning_rate in learning_rates:
        for conv_layer_number in conv_layer_numbers:
            net = Net(conv_layer_number=conv_layer_number, num_of_kernel=num_of_kernel, tanh=False, batch=True).to(
                device=device)
            net.run(learning_rate=learning_rate)

