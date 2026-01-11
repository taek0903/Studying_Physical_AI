# lmplot.py

import seaborn as sns
import matplotlib.pyplot as plt

# 1. 샘플 데이터셋 로드
iris = sns.load_dataset("iris")

# 3. 그래프 표시
sns.lmplot(data=iris,
           x="sepal_length",
           y="sepal_width",
           hue="species",
           height=6)
plt.title("Liner Regression Plot")
plt.show()