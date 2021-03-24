from os import name
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import click
import pickle


class Dataset:
    def __init__(
        self,
        source,
        train_dir="data/raw/train.csv",
        test_dir="data/raw/test.csv",
        seed=42,
    ):
        self.df = self.read_raw_data(source)
        self.train_dir = Path(train_dir)
        self.test_dir = Path(test_dir)
        self.seed = seed

    def read_raw_data(self, path):
        """read data from csv into dataframe"""
        df = pd.read_csv(path)
        return df

    def get_features(
        self,
        features=[
            "bill_length_mm",
            "bill_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
        ],
    ):
        """
        returns dataframe containing desired features for predicting target labels
        """
        X = self.df[features]
        return X

    def get_label(self, label="species", encoding=False, return_encoder=False):
        """
        returns the desired label for prediction
        label (str): dataframe column name
        encoding (bool): set to True for to encode string labels to integers
        return_encoder (bool): set to True to return LabelEncoder object
        """
        y = self.df[label]

        if encoding:
            label_encoder = LabelEncoder()
            label_encoder = label_encoder.fit(y)
            y = label_encoder.transform(y)
            if return_encoder:
                return pd.DataFrame(y), label_encoder

        return y

    def _get_na_indices(self, df):
        """gets the indices of the dataframe containing missing values"""
        indices = df[df.isna().any(axis=1)].index
        return indices

    def _drop_na_rows(self, df, max_na=1):
        """
        drops rows containing more na values than 'max_na'
        """
        df = df[df.isna().sum(axis=1) < max_na]
        return df

    def handle_missing_vals(self, X, y):
        """
        drop rows with that have multiple missing values;
        handle missing features by replacing them with 0;
        since xgboost has built-in nan handling
        """
        X = self._drop_na_rows(X.copy())
        na_indices = self._get_na_indices(X)
        X.iloc[na_indices] = 0
        # update the target label series
        y = y.copy().iloc[X.index]
        # reset indices for features and labels
        X.reset_index(drop=True)
        y.reset_index(drop=True)
        return X, y

    def split_data(self, X, y, testsize=0.3):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=self.seed, stratify=y, test_size=testsize
        )
        return X_train, X_test, y_train, y_test

    def save_artifact(self, artifact, filename):
        with open(filename, "wb") as f:
            pickle.dump(artifact, f)

    def save_train_test_sets(self, sets, outdir):
        """
        takes a list of train/test sets and saves them as pickle files
        sets (list): [X_train, X_test, y_train, y_test]
        outdir (str): path to save the artifacts in
        """
        outdir = Path(outdir)
        filenames = ["X_train.pkl", "X_test.pkl", "y_train.pkl", "y_test.pkl"]
        for set, filename in zip(sets, filenames):
            filename = outdir / filename
            self.save_artifact(set, filename)


@click.command()
@click.option("-ts", "--testsize", default=0.3, type=float)
@click.option("-o", "--outdir", default="data/processed/", type=click.Path())
@click.option("-f", "--filename", default="data/raw/palmer.csv", type=click.Path())
def main(filename, outdir, testsize):
    print(f"Reading data from {filename}")
    dataset = Dataset(source=filename)
    X = dataset.get_features()

    print("Encoding labals...")
    y, y_encoder = dataset.get_label(encoding=True, return_encoder=True)

    print("Handling missing values...")
    X, y = dataset.handle_missing_vals(X, y)

    print("Splitting training and testing data...")
    X_train, X_test, y_train, y_test = dataset.split_data(X, y, testsize)

    print(f"Saving data to {outdir}")
    dataset.save_train_test_sets(sets=[X_train, X_test, y_train, y_test], outdir=outdir)
    dataset.save_artifact(y_encoder, Path(outdir) / "y_encoder.pkl")

    print("Preprocessing and cleaning complete!")


if __name__ == "__main__":
    main()