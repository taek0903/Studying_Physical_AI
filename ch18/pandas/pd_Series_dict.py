# pd_Series_dict.py

import pandas as pd

data = {'a':10, 'b':20, 'c':30}
series = pd.Series(data)
print(series)
print(type(series))