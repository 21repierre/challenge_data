import os
import tempfile

import h5py
import numpy as np
import ray
import torch
from ray.air import session
from ray.train import Checkpoint
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

BASE_PATH = os.getcwd()

def load_dataset(batch_size, device):
    h5f = h5py.File(f'{BASE_PATH}/datasets/data_train_landmarks.h5', 'r')
    train_landmarks = h5f['data_train_landmarks'][:]
    h5f.close()

    h5f = h5py.File(f'{BASE_PATH}/datasets/data_train_labels.h5', 'r')
    train_labels = h5f['data_train_labels'][:]
    h5f.close()

    X_train, X_valid, y_train, y_valid = train_test_split(train_landmarks.reshape((len(train_landmarks), 10, 3 * 478)),
                                                          train_labels, test_size=0.2,
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

def train_model(config):
    # print(config)
    # Initialisation
    running_losses = []
    validation_losses = []
    accs = []

    device = config['device']
    SHOW_BAR = config['show_bar']
    # model = config['model'](n1=config['n1'], n2=config['n2'], n3=config['n3'], n4=config['n4'],
    #                        p1=config['p1'], p2=config['p2'], p3=config['p3'], p4=config['p4']).to(device)
    model = config['model'](hidden=int(config['hidden']), lstm_layers=config['lstm_layers'], nns=config['nns']).to(
        device)

    loss_fn = config['loss_fn']()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)

    # We load the training at a specific checkpoint if wanted
    checkpoint = session.get_checkpoint()
    if checkpoint:
        checkpoint_state = checkpoint.to_dict()
        start_epoch = checkpoint_state["epoch"]
        model.load_state_dict(checkpoint_state["net_state_dict"])
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    # Get the DataLoaders
    train_batch, validation_batch = load_dataset(batch_size=config['batch_size'], device=device)

    # Used for a better display using tqdm
    MAX_EPOCH = config['max_epoch']
    if SHOW_BAR:
        bar = tqdm(total=MAX_EPOCH)
        bar.update(start_epoch)

    # Training for every epoch
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

        # Creates a checkpoint if necessary
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save((model.state_dict(), optimizer.state_dict()), path)
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            ray.train.report({"loss": validation_loss, "accuracy": accuracy / validation_total},
                             checkpoint=checkpoint, )

        # Handles a proper display
        if SHOW_BAR:
            bar.update(1)
            bar.set_postfix(
                str=f"Running loss: {running_loss:.3f} - Validation loss: {validation_loss:.3f} | Accuracy: {accuracy / validation_total:.3f}.")
    if SHOW_BAR:
        bar.close()

    return running_losses, validation_losses, accs, model

class Model1(torch.nn.Module):

    def __init__(self, hidden=20, lstm_layers=2, nns=[(20, 0.5)]):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=1434, hidden_size=hidden, num_layers=lstm_layers, batch_first=True)
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(10, hidden, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(2),
        )
        self.linear_after = torch.nn.Sequential(torch.nn.Linear(hidden * (hidden // 2), nns[0][0]), torch.nn.ReLU())
        self.stack = torch.nn.Sequential(*np.array([
            [torch.nn.Linear(nns[i][0], nns[i + 1][0]), torch.nn.ReLU(), torch.nn.Dropout(nns[i][1])] for i in
            range(len(nns) - 1)
        ]).flatten())
        self.pred = torch.nn.Linear(nns[-1][0], 6)

    def forward(self, x: torch.Tensor):
        # print(x.shape, "LSTM")
        x, (h_t) = self.lstm(x)
        # print(x)
        # print(x.shape)
        x = self.conv(x)
        # print(x.shape)
        # x = torch.sum(x, dim=1)
        # print(x.shape)
        x = torch.nn.Flatten()(x)
        # print(x.shape)

        x = self.linear_after(x)
        for l in self.stack:
            # print(x.shape, l)
            x = l(x)
        # print(x.shape)
        return self.pred(x)
