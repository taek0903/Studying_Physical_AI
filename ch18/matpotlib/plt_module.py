# plt_module.py

import matplotlib.pyplot as plt

# 한글 폰트 설정
plt.rcParams["font.family"] = "Malgun Gothic"   # Windows 사용자

# 선그래프
# x=[1,2,3,4]
# y=[10,20,25,30]
# plt.plot(x,y)
# plt.title("Line Plot")
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# plt.show()

# 막대 그래프
# categories = ['A','B','C','D']
# values = [3,7,8,5]
# plt.bar(categories, values, color = 'red')
# plt.title('막대 그래프')
# plt.show()

# Histogram
# data = [1,2,2,3,3,3,4,4,4,4]
# plt.hist(data,bins=4,color='skyblue',edgecolor='black')
# plt.title('히스토그램')
# plt.show()

# 산점도(Scatter Plot)
# x=[5,7,8,7,2,17,2,9,4,11]
# y=[99,86,87,88,100,86,103,87,94,78]
# plt.scatter(x,y, color='green')
# plt.title('Scatter Plot')
# plt.show()

# Pie Chart
# sizes = [15,20,45,10]
# labels = ["Group A","Group B","Group C","Group D"]
# plt.pie(sizes, labels=labels, autopct='%1.1f%%', 
#         startangle=90, counterclock=False)
# plt.show()

# Box Plot
# data = [7,8,5,6,8,9,6,7,5,8]
# data = [7,8,5,6,8,9,6,7,5,8,11,4]
# data = [7,8,5,6,8,9,6,7,5,8,50]     # 50(outliner, 이상치)
# plt.boxplot(data)
# plt.title('Box Plot')
# plt.show()

# 이상치(outliner): 대부분의 데이터 패턴에서 크게 벗어난 값
# 이상치 생성 이유 : 측정오류, 데이터 처리 오류. 실제 특이한 사건

# x = [1,2,3,4]
# y = [10,20,25,30]
# plt.plot(x,y,color='red',linestyle='--', marker='o')
# plt.xlim(0,5)
# plt.ylim(0,40)
# plt.xticks(range(1,5))
# plt.yticks(0,41,10)
# plt.show()

# x = [1,2,3,4]
# y = [10,20,25,30]
# x1= [1,2,3,4]
# y1= [3,5,9,7]
# plt.plot(x,y, "k^", label="Data 1")
# plt.plot(x1,y1, label="Data 2")
# plt.legend(loc="upper left")
# plt.savefig()
# plt.show()

# Subplot 활용 1
x=[1,2,3,4]
y=[10,20,25,30]
categories = ['A','B','C','D']
values = [3,7,8,5]
data = [1,2,2,3,3,3,4,4,4,4]

fig, axs = plt.subplots(2,2)
axs[0,0].plot(x,y)
axs[0,1].bar(categories, values)
axs[1,0].scatter(x,y)
axs[1,1].hist(data)
plt.tight_layout()
plt.show()