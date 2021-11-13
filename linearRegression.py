import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
# x_train, x_test, y_train, y_test = train_test_split(df.)

df['MEDV'] = boston.target


print(df)