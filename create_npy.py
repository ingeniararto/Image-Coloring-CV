import os

import numpy
import torch
import hw3utils
from cnn import Net

IMAGE_NUMBER = 100

def get_loaders(batch_size, device):
    data_root = 'ceng483-f22-hw3-dataset'
    val_set = hw3utils.HW3ImageFolder(root=os.path.join(data_root, 'val'), device=device , is_training=False)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_set = hw3utils.HW3ImageFolder(root=os.path.join(data_root, 'test'), device=device , is_training=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return val_loader, test_loader


DEVICE_ID = 'cuda' if torch.cuda.is_available() else 'cpu'  # set to 'cpu' for cpu, 'cuda' /'cuda:0' or similar for gpu.
device = torch.device(DEVICE_ID)

# get first 100 images from loader
val_loader, test_loader = get_loaders(IMAGE_NUMBER, device)

# BEST CONFIGURATIONS
kernel = 16
conv = 2
lr = 0.1
tanh = False
batch = True

model = Net(conv, kernel, tanh, batch)

checkpoint = f"checkpoints/kr{kernel}-conv{conv}-lr{lr}{'-batch' if batch else ''}.pt"
model.load_state_dict(torch.load(checkpoint))

model = model.to(device)  # Set model to gpu
model.eval()

dataloader_iter = iter(test_loader)
inputs, targets = next(dataloader_iter)

# Get predictions
with torch.no_grad():
    predictions = model(inputs)

# Save predictions
predictions = predictions.detach().cpu().numpy()
numpy.save('estimations_test.npy', predictions)
