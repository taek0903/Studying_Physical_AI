print('6--------------------')
import pandas as pd
path = r'D:\rokey\python\ch18\data_hw18.csv'
df=pd.read_csv(path)
Age_mean = df["Age"].mean()
Age_max = df["Age"].max()
Age_min = df["Age"].min()
print(f'Age 평균:{Age_mean}, 최댓값:{Age_max}, 최솟값:{Age_min}')
Salary_mean = df["Salary"].mean()
Salaery_max = df["Salary"].max()
Salary_min = df["Salary"].min()
print(f'Salary 평균:{Salary_mean}, 최댓값:{Salaery_max}, 최솟값:{Salary_min}')

print('7--------------------')
filtered = df[df["Age"]>=30]
print(filtered)

print('8--------------------')
import numpy as np
arr = np.array([1,2,3,4,5,6,7,8,9,10])
arr_sq = arr ** 2
print(arr_sq)
arr_mean = arr_sq.mean()
arr_max = arr_sq.max()
arr_min = arr_sq.min()
print(f'평균:{arr_mean}, 최댓값:{arr_max}, 최솟값:{arr_min}')

print('9--------------------')
random=np.random.randint(1,20,(3,4))
print(random)
row_max=np.max(random,axis=1)
print(f'각행의 최댓값: {row_max}')

print('10-------------------')
import matplotlib.pyplot as plt
x=[1,2,3,4,5]
y=[2,4,6,8,10]
plt.plot(x,y)
plt.title('Line plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid()
plt.show()