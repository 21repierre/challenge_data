import numpy as np
import ray
import torch
from matplotlib import pyplot as plt
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from pierre import train_model, load_dataset, Model1

device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {device} device")


def ttrain():
    ray.init()
    config = {
        "hidden": tune.choice(np.arange(5, 100, 1)),
        "lstm_layers": tune.choice(np.arange(1, 10, 1)),
        "nns": [tune.choice(np.arange(1, 100, 1)) for _ in range(10)],
        "lr": tune.loguniform(1e-5, 1e-2),
        "batch_size": tune.choice([256, 512, 1024, 2048]),
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
        max_t=100,
        grace_period=10,
        reduction_factor=2,
    )

    # Handles the results
    result = tune.run(
        train_model,
        resources_per_trial=tune.PlacementGroupFactory(
            [{'CPU': 1}] * 4
        ),
        config=config,
        num_samples=20,
        scheduler=scheduler,

    )

    # Find the best config on the accuracy parameter
    best_trial = result.get_best_trial("accuracy", "max", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")
    ray.shutdown()


def mtrain():
    rl, vl, accs, m = train_model({
        'hidden': 45, 'lstm_layers': 3, 'nns': [74, 58, 25, 68, 37, 61, 97, 80, 73, 97],
        'lr': 0.004808064043725182,
        'batch_size': 256,
        'model': Model1,
        'loss_fn': torch.nn.CrossEntropyLoss,
        'loader': load_dataset,
        'max_epoch': 200,
        'device': device,
        'show_bar': True,
    })
    print("END")
    print(m)
    fig = plt.figure(figsize=(10, 5))
    fig.add_subplot(2, 1, 1)
    plt.plot(rl, label='Train loss')
    plt.plot(vl, label='Validation loss')
    plt.legend()

    fig.add_subplot(2, 1, 2)
    plt.plot(accs, label='Accuracy')
    plt.legend()
    plt.show()


mtrain()
#ttrain()
