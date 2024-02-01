import os

import h5py
import numpy as np
import ray
import torch
import torchvision.transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

torch.autograd.set_detect_anomaly(True)
BASE_PATH = os.getcwd()

h5f = h5py.File(f'{BASE_PATH}/datasets/data_train_images.h5', 'r')
train_images = h5f['data_train_images'][:]
h5f.close()

# train_images = np.array([train_images[i][-1] for i in range(len(train_images))])
train_images = np.expand_dims(train_images, 2)

print(train_images.shape)

h5f = h5py.File(f'{BASE_PATH}/datasets/data_train_labels.h5', 'r')
train_labels = h5f['data_train_labels'][:]
h5f.close()

X_train, X_valid, y_train, y_valid = train_test_split(train_images,
                                                      train_labels, test_size=0.15,
                                                      random_state=12345)

default_tr = torchvision.transforms.Compose([
    torch.nn.AvgPool2d(kernel_size=5)
])
augments = torchvision.transforms.Compose([
    torch.nn.AvgPool2d(kernel_size=5),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(25),
    torchvision.transforms.RandomAffine(degrees=0, translate=(0.01, 0.2), shear=(0.01, 0.04), scale=(0.8, 0.9)),
    # torchvision.transforms.RandomAutocontrast(),
])

## top
h5f = h5py.File(f'{BASE_PATH}/datasets/data_test_images.h5', 'r')
test_images = h5f['data_test_images'][:]
h5f.close()

#test_images = np.array([test_images[i][-1] for i in range(len(test_images))])
test_images = np.expand_dims(test_images, 2)

h5f = h5py.File(f'{BASE_PATH}/datasets/data_test_labels.h5', 'r')
test_labels = h5f['data_test_labels'][:]
h5f.close()

##


class CustomTensorDataset(Dataset):
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)
        #print(x.shape)
        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


def load_dataset(batch_size, device):
    TX_train = torch.tensor(X_train, dtype=torch.float).to(device)
    Ty_train = torch.tensor(y_train, dtype=torch.long).to(device)

    TX_valid = torch.tensor(X_valid, dtype=torch.float).to(device)
    Ty_valid = torch.tensor(y_valid, dtype=torch.long).to(device)

    TX_test = torch.tensor(test_images, dtype=torch.float).to(device)
    Ty_test = torch.tensor(test_labels, dtype=torch.long).to(device)

    t_dataset_train = CustomTensorDataset((TX_train, Ty_train), transform=default_tr)
    t_dataset_generalization = CustomTensorDataset((TX_train, Ty_train), transform=augments)
    t_dataset_valid = CustomTensorDataset((TX_valid, Ty_valid), transform=default_tr)
    t_dataset_test = CustomTensorDataset((TX_test, Ty_test), transform=default_tr)

    train_batch = DataLoader(t_dataset_train, batch_size=batch_size, shuffle=True)
    train_generalization_batch = DataLoader(t_dataset_generalization, batch_size=batch_size, shuffle=True)
    validation_batch = DataLoader(t_dataset_valid, batch_size=batch_size, shuffle=True)
    test_batch = DataLoader(t_dataset_test, batch_size=batch_size, shuffle=True)

    return train_batch, validation_batch, train_generalization_batch, test_batch


def convBlock(inSize, outSize, pooling=False, kernel_size=3):
    layers = [
        torch.nn.Conv2d(inSize, outSize, kernel_size=kernel_size, padding=1),
        torch.nn.BatchNorm2d(outSize),
        torch.nn.ReLU()
    ]
    if pooling:
        layers.append(torch.nn.MaxPool2d(kernel_size=2))

    return torch.nn.Sequential(*layers)


class ModelIm1(torch.nn.Module):
    def __init__(self, inChannel=1, nns=[(512, 0.5)]):
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
        self.classifier = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Flatten(),

            torch.nn.Linear(128 * 12 * 12, nns[0][0]),
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
        logits = self.classifier(logits)
        # print(logits.shape)
        return logits


class ModelIm2(torch.nn.Module):
    def __init__(self, inChannel=10, nns=[(512, 0.5)]):
        super(ModelIm2, self).__init__()
        # 10 * 200 * 200

        self.base = torchvision.models.resnet152()
        self.base.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base.fc = torch.nn.Sequential(torch.nn.Linear(self.base.fc.in_features, 300))

        self.lstm = torch.nn.LSTM(300, 256, 3)

        self.classifier = torch.nn.Sequential(

            torch.nn.Linear(256, nns[0][0]),
            torch.nn.ReLU(),
            torch.nn.Dropout(nns[0][1]),

            # torch.nn.Linear(nns[0][0], nns[1][0]),
            # torch.nn.ReLU(),
            # torch.nn.Dropout(nns[1][1]),

            torch.nn.Linear(nns[-1][0], 6)
        )

    def forward(self, X):
        # batch * 10 * 1 * 50 * 50
        hidden = None
        #print(X.shape)
        for i in range(X.size(1)):
            x = X[:, i, :, :, :]
            #print(x.shape)
            x = self.base(x)
            out, hidden = self.lstm(x.unsqueeze(0), hidden)

        x = self.classifier(out[-1, :, :])
        return x


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
    test_accs = []

    device = config['device']
    SHOW_BAR = config['show_bar']

    model = config['model'](nns=config['nns']).to(device)

    loss_fn = config['loss_fn']()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)

    start_epoch = 0

    if chk is not None:
        model.load_state_dict(chk['model_state_dict'])
        optimizer.load_state_dict(chk['optimizer_state_dict'])
        start_epoch = chk['epoch']

    # Get the DataLoaders
    train_batch, validation_batch, train_generalization_batch, test_batch = load_dataset(
        batch_size=config['batch_size'], device=device)

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
            trainer = train_generalization_batch if epoch >= config['min_generalization_epoch'] and epoch % config[
                'generalization_step'] < config['generalization_duration'] else train_batch
            for _, data in enumerate(trainer):
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

            test_loss = 0
            test_acc = 0
            test_total = 0
            for _, data in enumerate(test_batch):
                # No computation of the gradient for better performances
                with torch.no_grad():
                    inputs, labels = data
                    outputs = model(inputs).squeeze()
                    # print(torch.argmax(outputs, dim=1), labels.squeeze())
                    loss = loss_fn(outputs, labels.squeeze())
                    test_loss += loss.item()
                    test_total += labels.size(0)
                    test_acc += (torch.argmax(outputs, dim=1) == labels).sum().item()

            running_loss /= len(train_batch)
            validation_loss /= len(validation_batch)
            test_loss /= len(test_batch)
            running_losses.append(running_loss)
            validation_losses.append(validation_loss)
            accs.append(accuracy / validation_total)
            test_accs.append(test_acc / test_total)

            ray.train.report({"loss": validation_loss, 'test_loss': test_loss, "accuracy": accuracy / validation_total,
                              "test_accuracy": test_acc / test_total})

            # Handles a proper display
            if SHOW_BAR:
                bar.update(1)
                bar.set_postfix(
                    str=f"Running loss: {running_loss:.4f} - Validation loss: {validation_loss:.4f} | Accuracy: {accuracy / validation_total:.4f} | Test acc: {test_acc / test_total:.4f}")
    except KeyboardInterrupt:
        print(f"Saving model at {epoch} steps.")
    if SHOW_BAR:
        bar.close()

    return running_losses, validation_losses, accs, test_accs, epoch, model, optimizer
