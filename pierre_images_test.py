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

from pierre_images import train_model, load_dataset, ModelIm1

device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {device} device")

def ttrain():
    ray.init()
    config = {
        "nns": [(tune.choice(np.arange(32, 2048, 2)), tune.uniform(0.3, 0.6)) for _ in range(2)],
        "lr": 1e-4,
        "batch_size": tune.choice([16, 32, 64]),
        'model': tune.choice([ModelIm1]),
        'loss_fn': torch.nn.CrossEntropyLoss,
        'loader': load_dataset,
        'max_epoch': 200,
        'device': device,
        'show_bar': False,
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=50,
        grace_period=5,
        reduction_factor=4,
    )

    # Handles the results
    result = tune.run(
        train_model,
        resources_per_trial={'CPU': 3, 'GPU': 1},
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
    files = [os.path.join('models_images', f) for f in os.listdir('models_images')]  # add path to each file
    files.sort(key=lambda x: os.path.getmtime(x))
    checkpoint = torch.load(f'{files[-1]}')

    model: torch.nn.Module = checkpoint['config']['model'](nns=checkpoint['config']['nns']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    BASE_PATH = os.getcwd()
    h5f = h5py.File(f'{BASE_PATH}/datasets/data_test_images.h5', 'r')
    test_images = h5f['data_test_images'][:]
    test_images = np.array([test_images[i][-1] for i in range(len(test_images))])
    test_images = np.expand_dims(test_images, 1)
    h5f.close()
    TX_test = torch.tensor(test_images, dtype=torch.float).to(device)
    result = torch.argmax(model(TX_test), dim=1)
    print(result)
    f = open(f'datasets/submission_{time.time()}.csv', 'w')
    f.write("Id,Expression\n")
    for i, expr in enumerate(result):
        f.write(f"{i},{expr}\n")
    f.close()

def mctrain():
    files = [os.path.join('models_images', f) for f in os.listdir('models_images')]  # add path to each file
    files.sort(key=lambda x: os.path.getmtime(x))
    checkpoint = torch.load(f'{files[-1]}')
    # print(checkpoint)
    # checkpoint['config']['max_epoch'] += 5000
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
        'nns': [(512, 0.5)],
        'lr': 5e-6, 'batch_size': 64,
        'model': ModelIm1,
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
    }, f'./models_images/{round(time.time())}.pt')
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
