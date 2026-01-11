# generator1.py

# 리스트 사용 예제
import time

def longtime_job():
    print("job start")
    time.sleep(1)   # 1초 지연
    return 'done'

list_job=[longtime_job() for i in range(5)]
# print(list_job)
print(list_job[0])