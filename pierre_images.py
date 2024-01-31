import os

import h5py
import numpy as np
import ray
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

torch.autograd.set_detect_anomaly(True)
BASE_PATH = os.getcwd()

h5f = h5py.File(f'{BASE_PATH}/datasets/data_train_images.h5', 'r')
train_images = h5f['data_train_images'][:]
train_images = np.array([train_images[i][-1] for i in range(len(train_images))])
train_images = np.expand_dims(train_images, 1)
print(train_images.shape)
h5f.close()

h5f = h5py.File(f'{BASE_PATH}/datasets/data_train_labels.h5', 'r')
train_labels = h5f['data_train_labels'][:]
h5f.close()

def load_dataset(batch_size, device):
    X_train, X_valid, y_train, y_valid = train_test_split(train_images,
                                                          train_labels, test_size=0.15,
                                                          random_state=12345)
    TX_train = torch.tensor(X_train, dtype=torch.float).to(device)
    Ty_train = torch.tensor(y_train, dtype=torch.long).to(device)

    TX_test = torch.tensor(X_valid, dtype=torch.float).to(device)
    Ty_test = torch.tensor(y_valid, dtype=torch.long).to(device)

    t_dataset_train = TensorDataset(TX_train, Ty_train)
    t_dataset_test = TensorDataset(TX_test, Ty_test)

    train_batch = DataLoader(t_dataset_train, batch_size=batch_size, shuffle=True)
    validation_batch = DataLoader(t_dataset_test, batch_size=batch_size, shuffle=True)

    return train_batch, validation_batch

def convBlock(inSize, outSize, pooling=False):
    layers = [
        torch.nn.Conv2d(inSize, outSize, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(outSize),
        torch.nn.ReLU()
    ]
    if pooling:
        layers.append(torch.nn.MaxPool2d(kernel_size=2))

    return torch.nn.Sequential(*layers)

class ModelIm1(torch.nn.Module):
    def __init__(self, inChannel, nns=[(512, 0.5)]):
        super(ModelIm1, self).__init__()
        # 1 * 200 * 200
        self.conv1 = torch.nn.Sequential(
            convBlock(inChannel, 16),
            convBlock(16, 32, pooling=True)
        )  # 32 * 100 * 100
        self.residual1 = torch.nn.Sequential(
            convBlock(32, 32),
            convBlock(32, 32),
        )  # 32 * 100 * 100
        self.conv2 = torch.nn.Sequential(
            convBlock(32, 64, True),
            convBlock(64, 128, True)
        )  # 128 * 25 * 25
        self.residual2 = torch.nn.Sequential(
            convBlock(128, 128),
            convBlock(128, 128),
        )  # 128 * 25 *25
        self.conv3 = torch.nn.Sequential(
            convBlock(128, 256, True),
            convBlock(256, 512, True)
        )  # 512 * 6 * 6
        self.residual3 = torch.nn.Sequential(
            convBlock(512, 512),
            convBlock(512, 512),
        )  # 512 * 6 * 6
        self.classifier = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Flatten(),

            torch.nn.Linear(512 * 3 * 3, nns[0][0]),
            torch.nn.ReLU(),
            torch.nn.Dropout(nns[0][1]),

            torch.nn.Linear(nns[0][0], nns[1][0]),
            torch.nn.ReLU(),
            torch.nn.Dropout(nns[1][1]),

            torch.nn.Linear(nns[-1][0], 6)
        )

    def forward(self, x):
        # print(x.shape)
        logits = self.conv1(x)
        # print(logits.shape)
        logits = logits + self.residual1(logits)
        # print(logits.shape)
        logits = self.conv2(logits)
        # print(logits.shape)
        logits = logits + self.residual2(logits)
        # print(logits.shape)
        logits = self.conv3(logits)
        # print(logits.shape)
        logits = logits + self.residual3(logits)
        # print(logits.shape)
        logits = self.classifier(logits)
        # print(logits.shape)
        return logits

def train_model(config):
    # print(config)
    chk = None
    if 'checkpoint' in config:
        chk = config['checkpoint']
        config = chk['config']
    # Initialisation
    running_losses = []
    validation_losses = []
    accs = []

    device = config['device']
    SHOW_BAR = config['show_bar']

    model = config['model'](inChannel=1, nns=config['nns']).to(device)

    loss_fn = config['loss_fn']()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)

    start_epoch = 0

    if chk is not None:
        model.load_state_dict(chk['model_state_dict'])
        optimizer.load_state_dict(chk['optimizer_state_dict'])
        start_epoch = chk['epoch']

    # Get the DataLoaders
    train_batch, validation_batch = load_dataset(batch_size=config['batch_size'], device=device)

    # Used for a better display using tqdm
    MAX_EPOCH = config['max_epoch']
    if SHOW_BAR:
        bar = tqdm(total=MAX_EPOCH, initial=start_epoch)

    # Training for every epoch
    try:
        for epoch in range(start_epoch, MAX_EPOCH):
            running_loss = 0.
            model.train()

            # Computes the running loss during training
            for _, data in enumerate(train_batch):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()

                optimizer.step()

                running_loss += loss.item()

            validation_loss = 0.
            accuracy = 0
            validation_total = 0
            model.eval()

            # Test the model and evaluates the performances
            for _, data in enumerate(validation_batch):
                # No computation of the gradient for better performances
                with torch.no_grad():
                    inputs, labels = data
                    outputs = model(inputs).squeeze()
                    # print(torch.argmax(outputs, dim=1), labels.squeeze())
                    loss = loss_fn(outputs, labels.squeeze())
                    validation_loss += loss.item()
                    validation_total += labels.size(0)
                    accuracy += (torch.argmax(outputs, dim=1) == labels).sum().item()

            running_loss /= len(train_batch)
            validation_loss /= len(validation_batch)
            running_losses.append(running_loss)
            validation_losses.append(validation_loss)
            accs.append(accuracy / validation_total)

            ray.train.report({"loss": validation_loss, "accuracy": accuracy / validation_total},
                             )

            # Handles a proper display
            if SHOW_BAR:
                bar.update(1)
                bar.set_postfix(
                    str=f"Running loss: {running_loss:.3f} - Validation loss: {validation_loss:.3f} | Accuracy: {accuracy / validation_total:.3f}.")
    except KeyboardInterrupt:
        print(f"Saving model at {epoch} steps.")
    if SHOW_BAR:
        bar.close()

    return running_losses, validation_losses, accs, epoch, model, optimizer
