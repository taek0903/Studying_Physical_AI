# pd_data1.py

import pandas as pd

# 데이터 조작
print('-------------------------')
# dic 데이터 활용 생성
data ={
    'ID': [1,2,3],
    'Name': ['Alice', 'Bob','Charlie'],
    'Age': [30,35,25]
}

# df=pd.DataFrame(data, index=[1,2,3])
df = pd.DataFrame(data)
print(df)

print('열 선택-------------------')
print(df['Name'])
print(df['Age'])

print('조건 필터링---------------')
# 나이가 30 초과인 데이터 조회
print(df['Age'])
print(df['Age']>30)
print(type(df['Age']>30))
# 데이터프레임 대괄호 안에 
# 불리언 시리즈를 넣으면 
# True인 행만 선택됨
filtered=df[df['Age'] > 30]
print(type(filtered))
print(filtered)

print('데이터 정렬---------------')
# sorted_df=df.sort_values(by='Age')
sorted_df=df.sort_values(by='Age',ascending=False)
print(sorted_df)

print('데이터 추가 및 삭제-------')
df['Salary'] = [5000,6000,7000]
print(df)

print('행 추가/수정-------------')
# df.loc[3] = [4, 'David', 40, 8000]
# print(df)
df.loc[len(df)] = [4, 'David', 40, 8000]
print(df)

print('행 삭제-----------------')
df = df.drop(1)
print(df)
# 행 번호 주의
# print(len(df))
# df.loc[len(df)] = [5, 'Eva', 50, 9000]
# print(df)

print('-----------------------')
# df.info()

# index 재정렬
# df1=df.reset_index(drop=True)
# print(df1)

print('데이터 연결-------------')
data2 = {
    'ID': [5,6],
    'Name': ['Eve','Frank'],
    'Age' : [28,33]
}
df2=pd.DataFrame(data2)
print(df2)
# concated = pd.concat([df, df2])
concated = pd.concat([df, df2], ignore_index=True)
print(concated)

print('데이터 병합-------------')
data3={
    'ID':[1,2,3,4,5,6],
    'Department':['HR',
                  "Engineering",
                  "Sales",
                  "R&D",
                  "Finace",
                  "Planning"]
}
df3 = pd.DataFrame(data3)
merged = pd.merge(concated, df3)
print(merged)

# 데이터 처리
print('결측치 확인-------------')
# print(merged.isnull())
print(merged.isnull().sum())

meanVal =  merged['Salary'].mean()
print(meanVal)
merged['Salary']=merged['Salary'].fillna(meanVal)
print(merged)

print('중복 데이터 처리--------')
data1 = {
    'ID':[1,3],
    'Name' : ['Alice', 'Charlie'],
    'Age' : [30,25],
    'Salary' : [5000,7000],
    'Department' : ['HR','Sales']
}

df1 = pd.DataFrame(data1)
df1=pd.concat([merged,df1])
print(df1)

print(df1.duplicated())

df1_1 = df1.drop_duplicates()
print(df1_1)