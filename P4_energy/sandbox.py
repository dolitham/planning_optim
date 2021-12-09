import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from tools_preprocess import preprocess
from tools_models import *
pd.set_option('mode.chained_assignment', None)
np.set_printoptions(linewidth=100)
pd.set_option("display.max_columns", 100)
pd.set_option('display.width', 2000)
pd.set_option('display.max_rows', 100)

mode, directory = 'rb', 'pickle_storage/'
data = load(open(directory + 'data', mode)).reset_index(drop=True)
data = data[data['ENERGYSTARScore'].isnull() == False]
x_cat_columns = load(open(directory + 'x_cat', mode))
x_num_columns = load(open(directory + 'x_num_no_ESS', mode))
#x_num_columns = load(open(directory + 'x_num_w_ESS', mode))
y_num_columns = load(open(directory + 'y_num', mode))

cat_columns_to_remove = ['Neighborhood', 'ThirdLargestPropertyUseType', 'BuildingType']
x_cat_columns = [col for col in x_cat_columns if col not in cat_columns_to_remove]

#y_name = 'SiteEnergyUse(kBtu)'
y_name = 'GHGEmissions(MetricTonsCO2e)'
model_name = 'Bagging'
x_preprocess_name = 'scaledDum'

accept_memorized = False

## GHG
model_id = model_name + '++' + x_preprocess_name + '__' + y_name
print(model_id)


x = data[x_num_columns + x_cat_columns]
y = data[y_name]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=9)
x_train, x_test = preprocess(x_train, x_test, x_preprocess_name)
ft2 = data['PropertyGFATotal'][y_test.index]


try:
    models_directory = 'pickle_storage/models/'
    model = load(open(models_directory + model_id, 'rb'))
    print('found previously trained model', model_id)
except:
    print('model not found, training')
    model, _ = dispatch_and_print_model_search(x_train, y_train, model_name, model_id, accept_memorized=False)


model_id += 'WITHOUT ENERGY STAR SCORE'
predictions = model.predict(x_test)
if 'log' in y_name:
    predictions, y_test = exp(predictions), exp(y_test)

if '/ft2' in y_name:
    predictions, y_test = predictions * ft2, y_test * ft2

plt.figure(figsize=(6, 6))
plt.title(model_id)
sns.scatterplot(y_test, predictions)
plt.ylabel('predictions')
plt.show()


plt.figure(figsize=(10, 10))
plt.title(model_id + ' (zoomed)')
sns.scatterplot(y_test, predictions)
plt.ylabel('predictions')
if 'Site' in y_name:
    plt.xlim((-1000,110000000))
    plt.ylim((-1000,110000000))
else:
    plt.xlim((-20, 2000))
    plt.ylim((-20, 2000))
plt.show()

print('RMSE', rmse(predictions, y_test))
print('RMSE / std', rmse(predictions, y_test) / y_train.std())