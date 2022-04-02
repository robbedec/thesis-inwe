import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from src.analysis.enums import Measurements

data_file = '/home/robbedec/repos/ugent/thesis-inwe/src/analysis/csv/meei_measurements_with_synkinetic.csv'

# Data processing
df_data = pd.read_csv(data_file)
# Wanted fields: Measurements and the category
df_data = df_data[df_data.columns.intersection([e.name for e in Measurements] + ['category'])]

y = df_data['category']
x = df_data.drop(['category'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.25, random_state=27)

# Approx 25% with 11 classes.
clf = MLPClassifier(hidden_layer_sizes=(100,100), max_iter=2000, alpha=0.0001,
                     solver='adam', verbose=10, random_state=21)
#clf = RandomForestClassifier(max_depth=200, random_state=0)

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

acc = accuracy_score(y_test, y_pred)

print(acc)