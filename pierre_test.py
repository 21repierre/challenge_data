import os
import time

import h5py
import numpy as np
import ray
import torch
from matplotlib import pyplot as plt
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from pierre import train_model, load_dataset, Model1

device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {device} device")

def ttrain():
    ray.init()
    config = {
        "hidden": tune.choice(np.arange(5, 512, 1)),
        # "lstm_layers": tune.choice(np.arange(1, 20, 1)),
        "nns": [(tune.choice(np.arange(1, 400, 1)), tune.uniform(0.3, 0.6)) for _ in range(5)],
        "lr": 5e-6,
        "batch_size": tune.choice([16, 32, 64, 128, 256]),
        'model': tune.choice([Model1]),
        'loss_fn': torch.nn.CrossEntropyLoss,
        'loader': load_dataset,
        'max_epoch': 1000,
        'device': device,
        'show_bar': False,
    }
    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=200,
        grace_period=10,
        reduction_factor=4,
    )

    # Handles the results
    result = tune.run(
        train_model,
        # resources_per_trial=tune.PlacementGroupFactory(
        #     [{'CPU': 1}, {'GPU': 1}] * 4
        # ),
        resources_per_trial={'CPU': 3, 'GPU': 0.8},
        config=config,
        num_samples=20,
        scheduler=scheduler,
        progress_reporter=CLIReporter(max_progress_rows=10, print_intermediate_tables=False)
    )

    # Find the best config on the accuracy parameter
    best_trial = result.get_best_trial("accuracy", "max", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")
    ray.shutdown()

def mcpred():
    files = [os.path.join('models', f) for f in os.listdir('models')]  # add path to each file
    files.sort(key=lambda x: os.path.getmtime(x))
    checkpoint = torch.load(f'{files[-1]}')

    model: torch.nn.Module = checkpoint['config']['model'](hidden=int(checkpoint['config']['hidden']),
                                                           lstm_layers=checkpoint['config']['lstm_layers'],
                                                           nns=checkpoint['config']['nns']).to(
        device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    BASE_PATH = os.getcwd()
    h5f = h5py.File(f'{BASE_PATH}/datasets/data_test_landmarks.h5', 'r')
    test_landmarks = h5f['data_test_landmarks'][:]
    test_landmarks = np.array([test_landmarks[i][-1] for i in range(len(test_landmarks))])
    h5f.close()
    TX_test = torch.tensor(test_landmarks, dtype=torch.float).to(device)
    result = torch.argmax(model(TX_test), dim=1)
    print(result)
    f = open('datasets/submission.csv', 'w')
    f.write("Id,Expression\n")
    for i, expr in enumerate(result):
        f.write(f"{i},{expr}\n")
    f.close()

def mctrain():
    files = [os.path.join('models', f) for f in os.listdir('models')]  # add path to each file
    files.sort(key=lambda x: os.path.getmtime(x))
    checkpoint = torch.load(f'{files[-1]}')
    # print(checkpoint)
    checkpoint['config']['max_epoch'] += 5000
    rl2, vl2, accs2, epoch, m, optimizer = train_model(config={
        'checkpoint': checkpoint,
    })
    rl = checkpoint['train_losses']
    vl = checkpoint['validation_losses']
    accs = checkpoint['accuracies']
    rl.extend(rl2)
    vl.extend(vl2)
    accs.extend(accs2)
    torch.save({
        'model_state_dict': m.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': checkpoint['config'],
        'epoch': epoch,
        'train_losses': rl,
        'validation_losses': vl,
        'accuracies': accs
    }, f'{files[-1].replace(".pt", "").split("-")[0]}-{round(time.time())}.pt')
    print("END")
    print(m)
    print(np.max(accs))

    fig = plt.figure(figsize=(10, 5))
    fig.add_subplot(2, 1, 1)
    plt.plot(rl, label='Train loss')
    plt.plot(vl, label='Validation loss')
    plt.legend()

    fig.add_subplot(2, 1, 2)
    plt.plot(accs, label='Accuracy')
    plt.legend()
    plt.show()

def mtrain():
    MAX_EPOCH = 2000
    config = {
        'hidden': 254, 'lstm_layers': 17,
        'nns': [(267, 0.43877970104996844), (327, 0.3030090602140453), (318, 0.5996603425801295),
                (220, 0.4478912598162003)], 'lr': 1e-06, 'batch_size': 128,
        'model': Model1,
        'loss_fn': torch.nn.CrossEntropyLoss,
        'loader': load_dataset,
        'max_epoch': MAX_EPOCH,
        'device': device,
        'show_bar': True,
    }
    rl, vl, accs, epoch, m, optimizer = train_model(config=config)
    torch.save({
        'model_state_dict': m.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'epoch': epoch,
        'train_losses': rl,
        'validation_losses': vl,
        'accuracies': accs
    }, f'./models/{round(time.time())}.pt')
    print("END")
    print(m)
    print(np.max(accs))

    fig = plt.figure(figsize=(10, 5))
    fig.add_subplot(2, 1, 1)
    plt.plot(rl, label='Train loss')
    plt.plot(vl, label='Validation loss')
    plt.legend()

    fig.add_subplot(2, 1, 2)
    plt.plot(accs, label='Accuracy')
    plt.legend()
    plt.show()

import warnings

warnings.filterwarnings("ignore")

# mcpred()
# mtrain()
# mctrain()
ttrain()
