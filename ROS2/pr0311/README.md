##  트러블슈팅 (Troubleshooting)

### 문제 상황 (Issue)
* 커스텀 메시지 패키지(`ros_study_msgs`)를 터미널에서 `colcon build` 했을 때 **"빌드 성공"** 메시지가 출력됨.
* 하지만 파이썬 노드 작성 시 `from ros_study_msgs.msg import MyMsg`를 입력하면 **임포트(Import) 에러**가 발생하거나 에디터에서 클래스를 인식하지 못함.

### 원인 파악 (Root Cause)
`package.xml` 파일에서 파이썬 코드 변환기인 `rosidl_default_generators`의 의존성 태그를 잘못 지정했습니다. 
* **오류 원인:** `<build_depend>` 태그를 사용하여 콜콘(colcon)이 이를 단순한 '재료'로 인식했습니다. 그 결과, 텍스트 파일(`.msg`)을 파이썬 코드(`_my_msg.py`)로 변환하는 자동 생성 프로그램이 실행되지 않아 껍데기만 빌드된 상태가 되었습니다.

### 상세 개념: `<build_depend>` vs `<buildtool_depend>`
이 두 태그는 빌드 시스템(colcon)에게 완전히 다른 명령을 내립니다. 이해하기 쉽게 **"요리"**에 비유해 보겠습니다.

1. **`<build_depend>` (빌드 의존성) = "요리 재료"**
   * **의미:** 내 코드를 빌드(컴파일)할 때 같이 섞어서 써야 하는 라이브러리나 헤더 파일입니다.
   * **비유:** 김치찌개를 끓일 때 필요한 '김치'나 '돼지고기' 같은 재료입니다.
   * **예시:** `rclcpp`, `std_msgs`, `geometry_msgs` 등 (내 코드 안에서 `import`로 가져다 쓰는 패키지들)

2. **`<buildtool_depend>` (빌드 툴 의존성) = "요리 도구"**
   * **의미:** 내 코드를 빌드하는 과정 자체를 도와주거나, **빌드 과정 중에 실행되어야 하는 도구(프로그램)**입니다.
   * **비유:** 김치찌개를 끓일 때 사용하는 '가스레인지'나 '냄비', '칼' 같은 도구입니다.
   * **예시:** `ament_cmake`, `rosidl_default_generators` 등

> **결론적으로:** > 이전 코드는 "도구"를 "재료"로 착각해서 작동시키지 않았던 것입니다. `<buildtool_depend>`로 수정하여 콜콘(colcon)에게 **"이 도구를 직접 실행시켜서 파이썬 파일을 만들어내라!"**고 올바르게 지시해야 합니다.

### 해결 방법 (Solution)

**1. `package.xml` 수정**
해당 패키지의 `package.xml`을 열고 의존성 태그를 다음과 같이 수정합니다.

```xml
<build_depend>rosidl_default_generators</build_depend>

<buildtool_depend>rosidl_default_generators</buildtool_depend>