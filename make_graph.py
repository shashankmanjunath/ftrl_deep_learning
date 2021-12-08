import os

from fire import Fire

import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("ggplot")


def main(experiment_name):
    run_dir = "ftrl_dl_data"
    run_types = ["test_acc", "test_loss"]
    run_type_map = {
        "test_acc": "Test Accuracy",
        "test_loss": "Test Loss"
    }

    for run_type in run_types:
        runs = [x for x in os.listdir(run_dir) if run_type in x and x.endswith(".csv")]
        opt_data = []

        plt.figure()
        for run in runs:
            opt_type = run.split("-")[1]
            opt_data = pd.read_csv(os.path.join(run_dir, run))

            step = opt_data["Step"].to_list()
            value = opt_data["Value"].to_list()

            if opt_type == "sgd":
                opt_type = "SGD+M"
            elif "mda" in opt_type:
                opt_type = "MDA"
            elif "adam" in opt_type:
                opt_type = "Adam"
            else:
                opt_type = opt_type.upper()

            plt.plot(step[50:], value[50:], alpha=0.7, label=opt_type)

        plt.title(f"{experiment_name} {run_type_map[run_type]}")
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.legend()

        exp_name_save = experiment_name.replace(" ", "_")
        plt.savefig(os.path.join(run_dir, f"{exp_name_save}_{run_type}_closeup.png"), dpi=300)


if __name__ == "__main__":
    Fire(main)
