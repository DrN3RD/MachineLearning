#load_housing_data
import os
import tarfile
from six.moves import urllib
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


#Start Download routine for SC housing price data
   
DOWNLOAD_ROOT = "https://github.com/ageron/handson-ml/tree/master/"

HOUSING_PATH = os.path.join("datasets","housing")

HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url = HOUSING_URL, housing_path = HOUSING_PATH):

    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)

    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path = housing_path)
    housing_tgz.close()

def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

def split_train_test(data,test_ratio):

    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

def  test_set_check(identifier,test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio*2*32

def split_train_test_by_id(data,test_ratio,id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set],data.loc[in_test_set]



housing = load_housing_data() #fetch_housing_data()
housing["income_cat"] = np.ceil(housing["median_income"]/1.5)
housing["income_cat"].where(housing["income_cat"]<5,5.0,inplace=True)



split = StratifiedShuffleSplit(n_splits =1, test_size=0.2,random_state=42)

for train_index,test_index in split.split(housing,housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

    #Fixes the income cat category, so as to revert the manipulated labels to its original state.
for set_ in (strat_train_set,strat_test_set):
    set_.drop("income_cat",axis=1,inplace=True)

housing = strat_train_set.copy() # copy of original file so that we can manipulate it consequence free


corr_matrix = housing.corr()
attributes = ["median_house_value",
              "median_income",
              "total_rooms",
              "housing_median_age"]










#----------------GARBAGE COLLECTOR---------------------------#
#scatter_matrix(housing[attributes],figsize=(12,8))

#housing.plot(kind="scatter",x="median_income",y="median_house_value",alpha=0.1)
#plt.show()
#print(corr_matrix["median_house_value"].sort_values(ascending=False))


#print(strat_test_set["income_cat"].value_counts() /len(strat_test_set))

#housing.plot(kind="scatter", x="longitude", y="latitude",alpha=0.4,s=housing["population"]/100, label = "population", figsize=(10,7),
            # c="median_house_value", cmap=plt.get_cmap("jet"),colorbar = True)
#plt.legend()
#plt.show()

#housing["income_cat"].hist()
#plt.show()
#train_set,test_set = train_test_split(housing, test_size = 0.2, random_state = 42)


#housing_with_id = housing.reset_index() # adds an index column
#train_set,test_set = split_train_test_by_id(housing_with_id,0.2,"index")




#train_set, test_set = split_train_test(housing,0.2)






#print(housing.head()) #Gives the columns and values of some datasets

#print(housing.info()) #Gives info on the dataset

#print(housing["ocean_proximity"].value_counts()) # Gives more information on specified category

#print(housing.describe()) #Gives a summary of the numerical attributes

#housing.hist(bins=50,figsize = (20,15)) #Creates a histogram using 50 values = bins, and a figure size of 20 by 15
#plt.show() # plots the aforementioned command.

#print(len(train_set))
#print(len(test_set))