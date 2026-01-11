# pd_data.py

import pandas as pd

# 데이터 분석
# CSV 파일 데이터 읽기
path=r'D:\rokey\python\ch18\pandas\data.csv'
df_csv = pd.read_csv(path)
print(type(df_csv))
print(df_csv)

print('-------------------------')
import openpyxl

# path = r'D:\rokey\python\ch18\pandas\data..xlsx'
# df_xl=pd.read_excel(path)
# print(type(df_xl))
# print(df_xl)

print('-------------------------')
# print(df_csv.head())
print(df_csv.head(3))

print('-------------------------')
# print(df_csv.tail())
print(df_csv.tail(3))

print('-------------------------')
df_csv.info()

print('-------------------------')
print(df_csv.describe())

print('-------------------------')
print(df_csv.sample(frac=0.5))

# 데이터 조작
print('-------------------------')
