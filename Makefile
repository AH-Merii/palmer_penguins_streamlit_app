# dummy targets
.PHONY: all clean dataset streamlit

all: data/raw/palmer.csv dataset src/model/xgboost_train.py streamlit

# url to provided dataset on github
PALMER_REPO_URL = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv"

clean:
	rm -f data/raw/*.csv
	rm -f data/processed/*.pkl
	rm -f data/figures/*.png
	rm -f models/*.pkl

# download the dataset from the github repo
data/raw/palmer.csv:
	python src/data/download.py\
	 --url $(PALMER_REPO_URL)\
	  -f $@

dataset: data/raw/palmer.csv
	python src/data/dataset.py\
	 --testsize 0.3\
	 -f $<\
	 -o data/processed

src/model/xgboost_train.py : data/processed/
	python $@\
	 -i	$<\
	 -o models/model.pkl\
	 -r True

streamlit: 
	streamlit run src/app/app.py