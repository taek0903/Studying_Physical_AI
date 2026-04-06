# 🌡️ ROS 2 Temperature Control System (my_robot_system)

## 📌 프로젝트 개요
이 프로젝트는 온도 센서 데이터를 기반으로 쿨러(Cooler)와 스위치(Switch)를 자동 제어하는 ROS 2 종합 통신 시스템입니다. ROS 2의 3대 핵심 통신 방식인 **토픽(Topic), 서비스(Service), 액션(Action)**을 단일 패키지 내에 모두 구현하여 각 상황에 맞는 최적의 통신 방식을 설계하고 적용한 사례를 보여줍니다.

## 🏗️ 시스템 아키텍처 및 노드 구성
본 시스템은 총 4개의 독립적인 노드로 구성되어 유기적으로 상호작용합니다.

1. **`sensor_node` (Topic Publisher)**
   * 가상의 온도 데이터(25.0 ~ 35.0도)를 생성하여 1초 주기로 `temperature` 토픽으로 발행합니다.
2. **`manager_node` (Core Controller)**
   * 온도를 실시간으로 구독하며, 특정 임계값(30도 초과)에 도달하면 제어 명령을 내립니다.
   * 서비스 클라이언트와 액션 클라이언트를 통합 관리합니다.
3. **`cooler_service` (Service Server)**
   * 매니저의 `Trigger` 요청을 받아 즉각적으로 쿨러를 작동시키고 성공 여부를 반환하는 단발성 동기/비동기 제어를 담당합니다.
4. **`switch_action_server` (Action Server)**
   * 스위치를 물리적으로 켜고 끄는 행동(약 2초 소요 시뮬레이션)을 수행합니다. 
   * 처리 과정 중 현재 상태(Switch ON/OFF)를 피드백(Feedback)으로 실시간 전송합니다.

## 🚀 핵심 기술 및 트러블슈팅
이 프로젝트는 실무적인 ROS 2 노드 설계의 주요 문제점들을 해결했습니다.

* **비동기 통신(Async) 적용으로 블로킹(Blocking) 방지:** 매니저 노드가 서비스와 액션을 요청할 때 메인 스레드가 멈추지 않도록 `call_async()` 및 `send_goal_async()`를 사용하여 비동기 콜백(Callback) 구조를 구현했습니다. 이를 통해 센서 데이터를 놓치지 않고 실시간으로 계속 수신할 수 있습니다.
* **상태 변수를 통한 중복 호출 제어:**
  온도가 30도를 넘는 동안 서비스와 액션 요청이 무한정 반복해서 쏟아지는 것을 방지하기 위해, 내부 상태 변수(`self.cooling_active`)를 활용하여 토글(Toggle) 방식의 안정적인 제어 로직을 구현했습니다.

## ⚙️ 의존성 (Dependencies)
이 패키지를 실행하기 위해 다음 환경과 패키지가 필요합니다.
* **OS:** Ubuntu 22.04 
* **ROS 2 Version:** Humble (또는 호환 버전)
* **Custom Interfaces:** 액션 통신을 위한 사용자 정의 인터페이스 패키지(`my_robot_interfaces`) 내의 `SwitchControl.action` 컴파일이 선행되어야 합니다.

## 💻 실행 방법 (How to Run)

1. **워크스페이스 빌드**
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select my_robot_system --symlink-install
   source install/local_setup.bash

   # Terminal 1: 액션 서버 실행
ros2 run my_robot_system switch_action_server

# Terminal 2: 서비스 서버 실행
ros2 run my_robot_system cooler_service

# Terminal 3: 센서 노드 실행 (데이터 퍼블리시 시작)
ros2 run my_robot_system sensor_node

# Terminal 4: 매니저 노드 실행 (핵심 제어기)
ros2 run my_robot_system manager_node

[INFO] [manager_node]: Received temperature: 32.45
[INFO] [manager_node]: Temperature high -> Cooling ON
[INFO] [manager_node]: Sending goal: ON
[INFO] [cooler_service]: Cooler activated!
[INFO] [manager_node]: Service result: Cooler turned on
[INFO] [manager_node]: Goal accepted
[INFO] [switch_action_server]: [Action] Received request: turn_on=True
[INFO] [switch_action_server]: [Action] Switch ON
[INFO] [manager_node]: Action result: True