# Palmer Penguin Streamlit Application
![Palmer Penguins Streamlit App](https://github.com/AH-Merii/palmer_penguins_streamlit_app/blob/445f4b1ee62b38e655e2d8c9d4b7f06879480a01/src/app/images/palmer_penguin.png)

# Installation
Make sure that you have installed the necessary python libraries specified in `environment.yml` or the `requirements.txt`.

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

After the build is complete, you should see a prompt similar to the image below; Simply copy the address into your browser and you should be greeted with the application.
<img src="https://github.com/AH-Merii/palmer_penguins_streamlit_app/blob/main/src/app/images/streamlit_prompt.png" alt="streamlit" width="400"/>

Using the application is extremely simple. Simply input your penguin dimensions, and the prediction will show up in the form of a picture!

<img src="https://github.com/AH-Merii/palmer_penguins_streamlit_app/blob/main/src/app/images/penguins.png" alt="penguins" width="1200"/>


| CHINSTRAP      | GENTOO      | ADELIE      |
|------------|-------------|-------------|
| <img src="https://github.com/AH-Merii/palmer_penguins_streamlit_app/blob/main/src/app/images/chinstrap.png" width="400"> | <img src="https://github.com/AH-Merii/palmer_penguins_streamlit_app/blob/main/src/app/images/gentoo.png" width="400"> | <img src="https://github.com/AH-Merii/palmer_penguins_streamlit_app/blob/main/src/app/images/adelie.png" width="400"> | 
