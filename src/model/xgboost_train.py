import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score
import pickle
import click

# non-interactive backend for matplotlib
mpl.use("Agg")


class XGBoostModel(object):
    def __init__(
        self,
        input_data_dir="data/processed",
        objective="multi:softprob",
        seed=42,
    ):
        """
        input_data_dir (str): path containing training and testing data
        objective (str): learning objective for model;'multi:softprob', 'multi:softmax'
        seed (int): random seed number
        """
        self.model = xgb.XGBClassifier(
            objective=objective, seed=seed, use_label_encoder=False
        )
        self.input_data_dir = Path(input_data_dir)
        self.name = "XGBoost"
        self.d = self.read_input_data()
        self.acc = 0

    def get_params(self):
        return self.clf.get_params()

    def read_input_data(self):
        """
        read the input training & testing data from input directory
        returns:
            d (dict): dictionary containing 'X_train','X_test','y_train','y_test','y_encoder'
        """
        d = {}

        for path in self.input_data_dir.iterdir():
            if not path.stem.startswith("."):
                with open(path, "rb") as f:
                    d[path.stem] = pickle.load(f)
        return d

    def fit(self, verbose=False):
        """
        fit model to the training data
        """
        self.model.fit(
            self.d["X_train"],
            self.d["y_train"],
            eval_set=[
                (self.d["X_train"], self.d["y_train"]),
                (self.d["X_test"], self.d["y_test"]),
            ],
            early_stopping_rounds=10,
            eval_metric="mlogloss",
            verbose=True,
        )
        # update model accuracy
        self.get_accuracy()
        if verbose:
            print(f"Trained {self.name} model, with accuracy {self.acc:.3f}")

    def eval(self):
        results = self.model.evals_result()
        return results

    def predict(self, X):
        y_pred = self.model.predict(X)
        return y_pred

    def get_accuracy(self, test_set=True, verbose=False):
        """
        test_set(bool): if set to true model will be evaluated on test set,
        else model will be evaluated on training set
        """
        if test_set == True:
            X = self.d["X_test"]
            y = self.d["y_test"]
        else:
            X = self.d["X_train"]
            y = self.d["y_train"]

        y_pred = self.predict(X)
        self.acc = accuracy_score(y, y_pred)
        if verbose:
            print(f"Accuracy: {self.acc*100:.2f}")
        return self.acc

    def save(self, fname, rename=True):
        fname = Path(fname)
        if rename:
            model_name = f"{fname.stem}_acc_{self.acc:.3f}.pkl"
            fname = fname.parent / model_name

        with open(fname, "wb") as ofile:
            pickle.dump(self.model, ofile, pickle.HIGHEST_PROTOCOL)

    def load(self, fname):
        fname = Path(fname)
        with open(fname, "rb") as ifile:
            self.model = pickle.load(ifile)
        # update accuracy for new model
        self.get_accuracy()


def plot_logloss(results, path="data/figures/"):
    path = Path(path)
    # plot log loss
    epochs = len(results["validation_0"]["mlogloss"])
    x_axis = range(0, epochs)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.plot(x_axis, results["validation_0"]["mlogloss"], label="Train")
    ax.plot(x_axis, results["validation_1"]["mlogloss"], label="Test")
    ax.legend()
    plt.ylabel("Log Loss")
    plt.title("XGBoost Log Loss")

    plt.savefig(path / "XGBoost_logloss.png")


@click.command()
@click.option("-o", "--outdir", default="models/model.pkl", type=click.Path())
@click.option("-r", "rename", default=False, type=bool)
@click.option("-i", "--input_data_dir", default="data/processed/", type=click.Path())
def main(input_data_dir, outdir, rename):
    model = XGBoostModel(input_data_dir=input_data_dir)
    print(f"Creating {model.name} model")
    print(f"Reading data from {input_data_dir}")
    d = model.read_input_data()
    print(f"Training {model.name} model\n")
    model.fit(verbose=True)
    print(f"Generating model evaluation figures...")
    results = model.eval()
    plot_logloss(results)
    print(f"Saving {model.name} to {outdir}")
    model.save(outdir, rename=rename)
    print(f"Model Saved!")


if __name__ == "__main__":
    main()