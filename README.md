# Palmer Penguin Streamlit Application

![Palmer Penguins Streamlit App](https://github.com/AH-Merii/palmer_penguins_streamlit_app/blob/445f4b1ee62b38e655e2d8c9d4b7f06879480a01/src/app/images/palmer_penguin.png)

# About the Dataset
Data were collected and made available by [Dr. Kristen
Gorman](https://www.uaf.edu/cfos/people/faculty/detail/kristen-gorman.php)
and the [Palmer Station, Antarctica LTER](https://pal.lternet.edu/), a
member of the [Long Term Ecological Research
Network](https://lternet.edu/).

# About
This is a simple web application used to predict the species of the penguin from the palmer penguin dataset. The possible species are one of three:
* Chinstrap
* Gentoo
* Adeli

The fields used to predict the penguin species are:
* bill length (mm)
* bill depth (mm)
* flipper length (mm)
* body mass (g)

The model used to predict the species is an **XGBoost** model.

# Installation
Make sure that you have installed the necessary python libraries specified in `environment.yml` or the `requirements.txt`.
In my case I used miniconda3 to create a new virtual env.<br>
`conda env create -n palmer_env -f environment.yml`<br>
`conda activate palmer_env`<br>

This repo makes use of the Makefile. In order to make the installation process as seamless as possible make sure that you have `make` installed on your system.
### Windows
Installing make on windows can easily be done using [chocolatey](https://chocolatey.org/install)<br>
`choco install make`<br>
#### or 
by installing [GnuWin](http://gnuwin32.sourceforge.net/install.html)
### Linux
`apt install build-essential`

## Building the Makefile
After installing make simply run 
`make all`

## Streamlit
After running make, the dataset should automatically start downloading. The dataset will automatically be cleaned and split and an xgboost model will be trained and stored in the `models/` directory.

After the build is complete, you should see a prompt similar to the image below; Simply copy the address into your browser and you should be greeted with the application.<br>

<img src="https://github.com/AH-Merii/palmer_penguins_streamlit_app/blob/main/src/app/images/streamlit_prompt.png" alt="streamlit" width="400"/>

Using the application is extremely simple. Simply input your penguin dimensions, and the prediction will show up in the form of a picture!

<img src="https://github.com/AH-Merii/palmer_penguins_streamlit_app/blob/main/src/app/images/penguins.png" alt="penguins" width="1200"/>


| CHINSTRAP      | GENTOO      | ADELIE      |
|------------|-------------|-------------|
| <img src="https://github.com/AH-Merii/palmer_penguins_streamlit_app/blob/main/src/app/images/chinstrap.png" width="400"> | <img src="https://github.com/AH-Merii/palmer_penguins_streamlit_app/blob/main/src/app/images/gentoo.png" width="400"> | <img src="https://github.com/AH-Merii/palmer_penguins_streamlit_app/blob/main/src/app/images/adelie.png" width="400"> | 

## Model Performance
The data was split in a stratified manner, using a 70/30 split for the training/testing set respectively.

The algorithm used was XGBoost; the final accuracy of the model was 98% on the testing set.

The plot below shows the mean log loss over the training epochs.

<img src="https://github.com/AH-Merii/palmer_penguins_streamlit_app/blob/main/src/app/images/XGBoost_logloss.png" alt="penguins" width="1200"/>

