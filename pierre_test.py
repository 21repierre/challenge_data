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
        "hidden": tune.choice(np.arange(5, 256, 1)),
        "lstm_layers": tune.choice(np.arange(1, 20, 1)),
        "nns": [(tune.choice(np.arange(1, 400, 1)), tune.uniform(0.3, 0.6)) for _ in range(3)],
        "lr": tune.loguniform(1e-3, 9e-1),
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
        max_t=300,
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

def mtrain():
    rl, vl, accs, m = train_model({
        'hidden': 74, 'lstm_layers': 4,
        'nns': [(46, 0.21042692833450838), (90, 0.6353813394558738), (33, 0.335049036350751)],
        'lr': 0.6214794529627559,
        'batch_size': 256,
        'model': Model1,
        'loss_fn': torch.nn.CrossEntropyLoss,
        'loader': load_dataset,
        'max_epoch': 1000,
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
# ttrain()
