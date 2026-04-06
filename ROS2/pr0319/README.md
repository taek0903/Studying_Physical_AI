# ROS 2 Communication in Jupyter Notebook

## 프로젝트 소개
이 프로젝트는 대화형 환경인 Jupyter Notebook에서 ROS 2 통신(Publisher/Subscriber)을 안정적으로 구현하는 방법을 다룹니다. 터미널 기반의 일반적인 파이썬 스크립트와 달리, Jupyter 환경 특성상 발생하는 이벤트 루프 충돌(Blocking) 문제를 해결하고 정상적인 메시지 송수신을 수행합니다.

## 트러블슈팅: `rclpy.spin()`의 한계
일반적인 ROS 2 파이썬 노드에서는 `rclpy.spin(node)`를 사용하여 콜백(Callback) 처리를 대기합니다. 하지만 이 메서드를 Jupyter Notebook 셀 안에서 실행하면 다음과 같은 치명적인 문제가 발생합니다.

* **문제 현상:** 셀의 실행 상태가 `[*]`로 영원히 멈춰버리며, 더 이상 다른 코드를 실행할 수 없게 됩니다.
* **원인 분석:** `rclpy.spin()`은 무한 루프를 돌며 메인 스레드를 완전히 점유(Blocking)합니다. Jupyter 역시 내부적으로 커널(IPython)의 비동기 이벤트 루프를 사용하므로, `spin()`이 실행되는 순간 커널의 제어권을 빼앗겨 충돌이나 타임아웃이 발생하게 됩니다.

## 💡 해결 방법: `for` 반복문과 `spin_once()`의 조합
Jupyter 커널을 멈추지 않으면서 ROS 2 메시지를 처리하기 위해, 메인 스레드를 무한히 대기시키는 `spin()` 대신 **`rclpy.spin_once()`**를 활용합니다.

`spin_once()`는 통신 큐에 대기 중인 콜백을 단 한 번만 처리한 뒤 바로 다음 줄의 코드로 제어권을 반환합니다. 이를 `for` 문이나 `while` 문과 결합하면, Jupyter 셀의 통제권을 유지하면서 원하는 시간이나 횟수만큼만 유연하게 노드를 실행할 수 있습니다.

### 💻 핵심 코드 비교

**❌ 기존 방식 (Jupyter에서 멈춤 현상 발생)**
'''
python
import rclpy
# 노드 초기화 및 생성 로직 (생략)

# 실행 순간 Jupyter 셀이 영구적으로 멈춤
rclpy.spin(node)

import rclpy
# 노드 초기화 및 생성 로직 (생략)

# 원하는 횟수만큼만 콜백을 처리하여 메인 스레드 점유 방지
for _ in range(100):
    rclpy.spin_once(node, timeout_sec=0.1)
    # 필요시 이 루프 안에서 추가적인 데이터 처리나 시각화 코드 삽입 가능
'''
### 실행방법
1. 터미널을 열고 ROS 2 환경을 소싱합니다.
bash
source /opt/ros/humble/setup.bash
2. 해당 워크스페이스에서 Jupyter Notebook을 실행합니다.
3. .ipynb 파일을 열고 셀을 순서대로 실행합니다.
4. 셀이 멈추지 않고 실시간으로 ROS 2 메시지를 발행 및 구독하는 것을 확인합니다.