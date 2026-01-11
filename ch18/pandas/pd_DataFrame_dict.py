# pd_DataFrame_list.py

import pandas as pd

# dict 데이터 활용 생성
data ={
    'ID': [1,2,3],
    'Name': ['Alice', 'Bob','Charlie'],
    'Age': [30,35,25]
}

df = pd.DataFrame(data, columns=['ID','Name',"Age"])
print(df)
print(type(df))