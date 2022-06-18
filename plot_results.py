import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
import numpy as np
from matplotlib.gridspec import GridSpec

## Uses the Weights and Biases API to construct plots from training and validation.


if __name__ == '__main__':
    import wandb

    runs = {"roberta-base": "1wbmr3oy",
            "roberta-large": "xjy3r7fc",
            "albert-large": "3dho439j",
            "deberta-large": "2eo0xhps"}
    api = wandb.Api()
    for model, run_id in runs.items():
        run = api.run(f"pydqn/sdg_clf/{run_id}")
        history = run.history()
        print(model)
        arg = history["val_f1"].argmax()
        print(history.loc[arg, :])

        for run_type in ["train", "val"]:

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
                plot.plot(range(10), history["train_" + metric][np.arange(10)] if run_type == "train" \
                          else history["val_" + metric][np.arange(10)], linewidth=3)
                plot.set_title(metric.capitalize(), fontsize=18)
                plot.set_xlabel("Epochs", fontsize=13)
                plot.set_xticks(range(10), range(10))
            gs.tight_layout(fig, pad=1.2)
            plt.savefig(f"{run.name}_metrics_{run_type}.png")
            plt.show()

