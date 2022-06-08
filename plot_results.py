import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec


if __name__ == '__main__':
    import wandb
    api = wandb.Api()
    run = api.run("pydqn/sdg_clf/1wbmr3oy")
    history = run.history()
    print(run.summary)
    run_type = "train"

    gs = GridSpec(2, 6)
    gs.update(wspace=0.75)
    fig = plt.figure(figsize=(18.2, 10.8))

    f1 = fig.add_subplot(gs[0, :3])
    loss = fig.add_subplot(gs[0, 3:])
    accuracy = fig.add_subplot(gs[1, :2])
    precision = fig.add_subplot(gs[1, 2:4])
    recall = fig.add_subplot(gs[1, 4:])

    plots = [f1, loss, accuracy, precision, recall]
    metrics = ["f1", "loss", "accuracy", "precision", "recall"]
    for plot, metric in zip(plots, metrics):
        plot.plot(range(10),history["train_"+metric][np.arange(10)*2] if run_type == "train" else history["val_"+metric][1+(np.arange(10)*2)], linewidth=3)
        plot.set_title(metric.capitalize(), fontsize=18)
        plot.set_xlabel("Epochs", fontsize=13)
        plot.set_xticks(range(10), range(10))
    gs.tight_layout(fig, pad=1.2)
    plt.savefig(f"{run.name}_metrics_{run_type}.png")
    plt.show()
    # fig, ax = plt.subplots(1, 5, figsize=(18.2, 3.6))
    # for row in range(1):
    #     for col in range(5):
    #         ax[col].plot(range(10), history["train_"+metrics[col]][np.arange(10)*2] if run_type == "train" else history["test_"+metrics[col]][1+(np.arange(10)*2)], linewidth=3)
    #         ax[col].set_title(metrics[col].capitalize())
    #         ax[col].set_xticks(range(10), range(10))
    # plt.tight_layout()
    # plt.savefig("testplot.png")
    # plt.show()
