#HousingCostPredictor.py
# First Exercise Chapter in Hands on with ML

# Import files
import os
import csv
import tarfile
from six.moves import urllib
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from zlib import crc32
from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer




DOWNLOAD_ROOT = "https://github.com/ageron/handson-ml/tree/master/"
HOUSING_PATH = os.path.join("datasets","housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL,housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url,tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path,"housing.csv")
    return pd.read_csv(csv_path)



#fetch_housing_data()
housing = load_housing_data()

housing["income_cat"] = np.ceil(housing["median_income"]/1.5)

split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state = 42)
for train_index,test_index in split.split(housing,housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat",axis=1,inplace=True)

#Make a copy of housing

housing = strat_train_set.copy()
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]

#looking for correlations
corr_matrix = housing.corr()
#print(corr_matrix["median_house_value"].sort_values(ascending=False))



## Prepping shit for ML
imputer = SimpleImputer(missing_values=np.nan,strategy = "median")
housing = strat_train_set.drop("median_house_value",axis=1)
housing_labels = strat_train_set["median_house_value"].copy()


housing_num = housing.drop("ocean_proximity",axis=1)
imputer.fit(housing_num)
X= imputer.transform(housing_num)
housing_tr = pd.DataFrame(X,columns=housing_num.columns)


#file = open("testfile.txt","w")
#for i in range(16511):
#    file.write(str(housing_tr["total_bedrooms"][i]) + "\n")
    #file.write()
#file.close()

