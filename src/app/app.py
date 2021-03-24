import pandas as pd
import streamlit as st
import pickle
from pathlib import Path
from skimage import io
import math


def read_artifact(fname):
    fname = Path(fname)
    with open(fname, "rb") as f:
        artifact = pickle.load(f)

    return artifact


def read_model(path):
    model = read_artifact(path)
    return model


def get_paths(dir):
    dir = Path(dir)
    paths = [path for path in dir.iterdir() if not path.stem.startswith(".")]
    return paths


def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier + 0.5) / multiplier


def order_of_magnitude(num):
    """get the order of magnitude of the number"""
    ndigits = int(math.log10(num)) + 1
    return ndigits


def round_magnitude(num):
    """round based on the order of magnitude of the number"""
    ndigits = order_of_magnitude(num)
    return round_half_up(int(num), -ndigits + 2)


def get_step_val(val1, val2):
    mean = (val1 + val2) / 2
    ndigits = order_of_magnitude(mean)
    return 10 ** (ndigits - 2)


@st.cache()
def get_input_range(series):
    """
    returns a range to be used for streamlit input,
        based on the standard deviation of the input data

    series (pd.Series): series containing float or int
    """
    std = series.std()
    max_val = series.max()
    min_val = series.min()
    # update max and min value based on standard deviation
    max_val = max_val + std
    min_val = min_val - std
    # round based on order of magnitude of the number
    min_val = round_magnitude(min_val)
    max_val = round_magnitude(max_val)
    return min_val, max_val


def get_model_dict(dir):
    paths = get_paths(dir)
    models = {path.stem: path for path in paths}
    return models


def generate_model_selector_field(models):
    st.subheader("Select a model")
    selected_model = st.selectbox("", list(models.keys()))
    # get the path to the selected model
    selected_model = models[selected_model]
    return selected_model


@st.cache()
def generate_all_field_ranges(X):
    field_dict = {}

    for col in X:
        min_val, max_val = get_input_range(X[col])
        field_dict[col] = (min_val, max_val, X[col].dtype)

    return field_dict


def generate_all_numeric_fields(field_dict):
    input_dict = {}
    for key, val in field_dict.items():
        min_val = val[0]
        max_val = val[1]
        step_val = get_step_val(min_val, max_val)
        # field title
        st.subheader(f"{key}")
        # store user input for each field as key value pairs
        input_dict[key] = [
            st.number_input("", float(min_val), float(max_val), step=float(step_val))
        ]
    return input_dict


def encode_prediction(col):
    y_encoder = read_artifact("data/processed/y_encoder.pkl")
    prediction = y_encoder.inverse_transform(col)
    return prediction


def predict_penguin(input_dict, model_path):
    model = read_model(model_path)
    df = pd.DataFrame(input_dict)
    pred = model.predict(df)
    pred = encode_prediction(pred)[0]

    return pred


def get_penguin_image(penguin=None, im_dir="src/app/images/"):
    im_dir = Path(im_dir)
    if penguin is None:
        im_path = im_dir / "penguins.png"
    else:
        im_path = im_dir / f"{penguin.lower()}.png"

    image = io.imread(im_path)
    return image


def get_title_image(path="src/app/images/palmer_penguin.png"):
    path = Path(path)
    image = io.imread(path)
    return image


def main():
    st.title("Palmer Penguin Predictor")
    title_image = get_title_image()
    st.image(title_image)
    st.write(
        """The objective of this model is to predict the 
    species of penguin given the 'bill length', 'bill depth',
     'flipper lenght' & 'body mass'. 
     \nSimply enter your penguin's measurements, and click predict!
     
     Note: I have enforced some limits based on the distribution of the 
     training data that was supplied; each measurement limit is determined by
     determining the maximum and minimum measurements and adding and subtracting 
     the standard deviation respectively.
     
For more information about this dataset check out the [Palmer Penguins Dataset](https://github.com/allisonhorst/palmerpenguins)"""
    )

    X = read_artifact("data/processed/X_train.pkl")
    field_dict = generate_all_field_ranges(X)
    input_dict = generate_all_numeric_fields(field_dict)
    model_dir = Path("models/")
    models = get_model_dict(model_dir)
    model_path = generate_model_selector_field(models)

    p_title = "Click Predict"

    penguin_title = st.empty()
    penguin_title.title(p_title)

    pred = None
    predict_button = st.empty()
    penguin = st.empty()
    penguin_image = get_penguin_image(penguin=pred)

    penguin.image(penguin_image)
    # if button is clicked predict penguin class
    if predict_button.button("Predict") and model_path is not None:
        pred = predict_penguin(input_dict, model_path)
        penguin_image = get_penguin_image(penguin=pred)

        penguin.image(penguin_image)
        penguin_title.title(f"{pred.upper()}!")


if __name__ == "__main__":
    main()
