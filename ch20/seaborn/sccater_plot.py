# sccater_plot.py

import seaborn as sns
import matplotlib.pyplot as plt

# 1. 샘플 데이터 셋 로드
iris = sns.load_dataset("iris")

# 2. 기본 스타일 설정
sns.set_theme(style='dark')

# 3. 그래프 표시
sns.scatterplot(data=iris,
                x="petal_length",
                y='petal_width',
                hue="species",
                style='species')
plt.title("Scateter Plot Example")
plt.show()