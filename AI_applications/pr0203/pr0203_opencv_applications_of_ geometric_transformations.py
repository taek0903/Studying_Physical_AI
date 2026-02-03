import numpy as np
import matplotlib.pyplot as plt
import cv2

plt.rcParams["font.family"] = "Malgun Gothic"

def create_sample_image():
    # 흰색 배경 이미지 생성(400*400) 크기. 3채널
    img = np.ones((400,400,3), dtype=np.uint8) * 255

    # 파란색 사각형 그리기 (왼쪽 위)
    cv2.rectangle(
        img,        # 그림 이미지
        (50,50),    # 시작점(x,y)
        (150,150),  # 끝점(x',y')
        (255,0,0),  # BGR 색상(파란색)
        -1          # -1: 내부 채우기, 양수: 선 두께
    )

    # 초록색 원 그리기 (오른쪽 위)
    cv2.circle(
        img,        # 그림 이미지
        (300, 100), # 중심점(x,y)
        50,         # 반지름(radius)
        (0,255,0),  # BGR 색상(초록색)
        -1        
    )

    # 빨간색 삼각형 그리기 (아래쪽)
    triangle_pts = np.array([[200,250],[150,350],[250,350]], dtype=np.int32)
    # 세 꼭지점
    cv2.fillPoly(
        img,
        [triangle_pts], # 다각형 점들(리스트로 감싸줘야함)
        (0, 0, 255),    # BGR 색상(빨간색)       
    )

    # 텍스트 추가
    cv2.putText(
        img,                        
        'DosanRobotics',            # 표시할 텍스트
        (100, 250),                 # 텍스트 시작 위치(x,y)
        cv2.FONT_HERSHEY_SIMPLEX,   # 폰트 종류
        1,                          # 폰트 크기
        (0,0,0),                    # 폰트 색상(검은색)
        2,                          # 텍스트 두께
        cv2.LINE_AA                 # 선 종류(anti-aliasing => 앨리어싱 부드러운 선)
    )

    return img

# 이미지 생성
sample_img = create_sample_image()

# BGR => RGB 변환
sample_img_rgb = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)

# BGR RGB 사진 결과 비교
fig, axes = plt.subplots(1, 2, figsize=(12,6))

axes[0].imshow(sample_img)
axes[0].set_title('Sample image', fontsize=12)
axes[0].axis('off')
axes[1].imshow(sample_img_rgb)
axes[1].set_title('Sample image RGB', fontsize=12)
axes[1].axis('off')

plt.tight_layout()
plt.show()

# fillter
# 평균 블러(average blur)
# 커널 크기만큼 픽셀들을 평균값으로 대체

blur_avg11 = cv2.blur(
    sample_img,
    (11, 11)        # 커널크기(가로,세로), 클수록 더 흐릿함
)

blur_avg5 = cv2.blur(
    sample_img,
    (5, 5)        # 커널크기(가로,세로), 클수록 더 흐릿함
)

blur_avg3 = cv2.blur(
    sample_img,
    (3, 3)        # 커널크기(가로,세로), 클수록 더 흐릿함
)

fig, axes = plt.subplots(1,3, figsize=(18,6))

axes[0].imshow(blur_avg11)
axes[0].set_title('Blur Average11', fontsize=12)
axes[0].axis('off')
axes[1].imshow(blur_avg5)
axes[1].set_title('Blur Average5', fontsize=12)
axes[1].axis('off')
axes[2].imshow(blur_avg3)
axes[2].set_title('Blur Average3', fontsize=12)
axes[2].axis('off')

plt.tight_layout()
plt.show()
'''
커널의 크기가 클수록 더 흐릿함
'''

# 가우시안 필터(Gaussian Blur)
# 평균, 중심에 가까울수록 더 큰 가중치(가중평균)주는 블러

blur_gaussian = cv2.GaussianBlur(
    sample_img,
    (11,11),        # 커널크기
    0               # sigmaX(표준편차) 0: 자동계산
)

# 중간값 블러(medianBlur)
# 주변 픽셀의 중앙값으로 픽셀을 바궈 튀는 노이즈를 제거하면서 윤곽을 비교적 잘 지켜주는 필터

blur_median = cv2.medianBlur(
    sample_img,
    11              # 반드시 홀수 여야함
)

# 양방향 필터(Bilateral Filter)
# 경계선 보존, 부드럽게 만들어줘요 (가장 최고급 필터)

blur_bilateral = cv2.bilateralFilter(
    sample_img,
    15,             # 픽셀 이웃 직경
    80,             # 색상 공간의 표준편차 (클수록 더 많은 색상 표현 가능)
    80              # 좌표 공간의 표준편차 (클수록 더 넓은 영역 고려)
)

fig, axes = plt.subplots(2,2, figsize=(12,12))
fig.suptitle('Blur Filters Comparison', fontsize=16, fontweight='bold')

images = [blur_avg11, blur_gaussian, blur_median, blur_bilateral]
titles = ['Average Blur', 'Gaussian Blur', 'Median Blur', 'Bilateral Filter']

for idx, (ax, img, title) in enumerate(zip(axes.flat, images, titles)):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)
    ax.set_title(title, fontsize=12)
    ax.axis('off')

plt.tight_layout()
plt.show()

# 엣지 검출 필터 적용학

# 소벨 필터 : 가로/세로 방향 엣지 검출
# 경계선 찾아주는 필터 (이미지에서 밝기값이 얼마나 급격하게 변했는지 즉 기울기 계산)
# 가로방향 엣지 검출 (세로선 강조)

sobel_x = cv2.Sobel(
    sample_img, 
    cv2.CV_64F, # 출력 이미지 타입 지정(64비트 float)
    1,          # x방향 미분 차수 (1 = 1차 미분) 
    0,          # y방향 미분 차수 (0 = 미분 안함)
    ksize=3     # 커널 사이즈 (1,3,5,7.... 홀수)
)

# -부호 없애기 위해 np.uint8을 사용
sobel_x = np.uint8(np.absolute(sobel_x))

# 세로방향 엣지 검출 (가로선 강조)

sobel_y = cv2.Sobel(
    sample_img, 
    cv2.CV_64F, # 출력 이미지 타입 지정(64비트 float)
    0,          # x방향 미분 차수 (0 = 미분 안함) 
    1,          # y방향 미분 차수 (1 = 1차 미분)
    ksize=3     # 커널 사이즈 (1,3,5,7.... 홀수)
)

# -부호 없애기 위해 np.uint8을 사용
sobel_y = np.uint8(np.absolute(sobel_y))

fig, axes = plt.subplots(1,2, figsize=(12,6))

axes[0].imshow(sobel_x)
axes[0].set_title('Sobel_x', fontsize=12)
axes[0].axis('off')
axes[1].imshow(sobel_y)
axes[1].set_title('Sobel_y', fontsize=12)
axes[1].axis('off')

plt.tight_layout()
plt.show()

'''
변화가 큰 경우 (-) 존재 가능
예를 들어 0(어두운 밤) => 255(밝은 곳) 엄청나게 큰 양수 변환
255 => 0 엄청나게 큰 음수 변화
그래서, 절대값 씌움
근데 왜 실수 했다가 uint(부호없는 정수)로 바꿀까?
부호없다며 => 음수 저장 안됨 => 변화량만 보고 싶어서
'''

# 가로/세로 방향 걸합한 필터
sobel_combined = cv2.addWeighted(
    sobel_x, 0.5,   # 1번째 이미지 가중치
    sobel_y, 0.5,   # 2번째 이미지 가중치
    0               
    # 추가 상수 0 : (이미지 밝기 조절) => 이 수치를 조절한다. -> 
    # 가중치 조절하지 않는 이유: 노이즈도 같이 증가(평균의 원리 추종)
)

# 라플라시안 필터(Laplacian Filter)
# 2차 미분(곡률, 변환점) => 모든 방향 엣지 검출

laplacian = cv2.Laplacian(
    sample_img,
    cv2.CV_64F  # 출력 이미지 타입 저장(64비트 float)
)
laplacian = np.uint8(np.absolute(laplacian))

# 캐니 엣지(Canny) : 가장 정밀한 엣지 검출

# 1단계 : 먼저 그레이스케일(흑백) 변환
gray = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)

# 2단계 : 엣지 검출 함수 적용
canny = cv2.Canny(
    gray,   # 그레이스케일로 변환된 이미지(케니는 흑맥만 좋아해, 흑밴만 지원)
    50,     # 최소 임계값(이보다 낮으면 엣지 아님)
    150,    # 최대 임계값(이보다 높으면 엣지 확실)
)
'''
임계값 설정이 가장 중요
최대 임계값(이보다 높으면 확실한 엣지)
=> 이미지에서 밝기 변화량(기울기)
=> 여기서 150 설정했다는 것은 150이상 픽셀은 무조건 강한 엣지로 간주
=> 최종 결과에 포함

최소 임계값(이보다 낮으면 엣지 아님)
여기서 50 설정했다는 것은 50 미만은 무조건 엣지가 아니라고 간주 => 버림

사이 구간(50-150) => 약한 엣지 후보
=> 강한 엣지(150이상) 연결되어 있을 때만 최종적으로 엣지로 인정
=> 연결 안되어 있으면 => 버림
'''

# 4개의 결과를 2x2 그리드로 표시
fig, axes = plt.subplots(2, 2, figsize=(12, 12))  # 2행 2열 서브플롯
fig.suptitle('Edge Detection Filters', fontsize=16, fontweight='bold')  # 전체 제목

# 결과 표시
edge_images = [sobel_combined, laplacian, canny, gray]  # 이미지 리스트
edge_titles = ['Sobel (X+Y)', 'Laplacian', 'Canny Edge', 'Original (Gray)']  # 제목 리스트

for idx, (ax, img, title) in enumerate(zip(axes.flat, edge_images, edge_titles)):
    # 캐니와 그레이는 이미 흑백이므로 cmap='gray' 사용
    if idx >= 2:
        ax.imshow(img, cmap='gray')     # 흑백으로 표시
    else:
        img_rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)              # 컬러로 표시
    
    ax.set_title(title, fontsize=12)  # 제목 설정
    ax.axis('off')  # 축 숨기기    

# 회전, 크기 조절
# 다양한 각도 크기 이미지 회전하기

height, width = sample_img.shape[:2]

center = (width//2, height//2)
# 이미지 중심저점 계산 (// 몫 정수 나눗셈)

# 45도 회전
matrix45 = cv2.getRotationMatrix2D(
    center, # 회전 중심점
    45,     # 회전 각도(양수: 반시계, 음수: 시계)
    1.0     # scale 크기
)

rotated45 = cv2.warpAffine(
    sample_img,     # 원본 이미지
    matrix45,       # 변환 행렬
    (width, height) # 출력 이미지 크기(크기 유지)
)

# 90도 회전
matrix90 = cv2.getRotationMatrix2D(
    center, # 회전 중심점
    90,     # 회전 각도(양수: 반시계, 음수: 시계)
    1.0     # scale 크기
)

rotated90 = cv2.warpAffine(
    sample_img,     # 원본 이미지
    matrix90,       # 변환 행렬
    (width, height) # 출력 이미지 크기(크기 유지)
)

# 45도 회전 + 0.5배 축소
matrix45_half = cv2.getRotationMatrix2D(
    center, # 회전 중심점
    45,     # 회전 각도(양수: 반시계, 음수: 시계)
    0.5     # scale 크기
)

rotated45_half = cv2.warpAffine(
    sample_img,     # 원본 이미지
    matrix45_half,       # 변환 행렬
    (width, height) # 출력 이미지 크기(크기 유지)
)

# 30도 회전 + 1.5배 축소
matrix30_large = cv2.getRotationMatrix2D(
    center, # 회전 중심점
    30,     # 회전 각도(양수: 반시계, 음수: 시계)
    1.5     # scale 크기
)

rotated30_large = cv2.warpAffine(
    sample_img,     # 원본 이미지
    matrix30_large, # 변환 행렬
    (width, height) # 출력 이미지 크기(크기 유지)
)

# 결과를 2x2 그리드로 표시
fig, axes = plt.subplots(2, 2, figsize=(12, 12))  # 2행 2열 서브플롯
fig.suptitle('Rotation & Scaling', fontsize=16, fontweight='bold')  # 전체 제목

# 회전 결과들
rotation_images = [rotated45, rotated90, rotated45_half, rotated30_large]  # 이미지 리스트
rotation_titles = [
    'Rotate 45° (scale=1.0)',  # 45도 회전
    'Rotate 90° (scale=1.0)',  # 90도 회전
    'Rotate 45° (scale=0.5)',  # 45도 회전 + 축소
    'Rotate 30° (scale=1.5)'   # 30도 회전 + 확대
]

# 각 subplot에 이미지 표시
for ax, img, title in zip(axes.flat, rotation_images, rotation_titles):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR을 RGB로 변환
    ax.imshow(img_rgb)  # 이미지 표시
    ax.set_title(title, fontsize=11)  # 제목 설정
    ax.axis('off')  # 축 숨기기

plt.tight_layout()  # 레이아웃 자동 조정
plt.show()  # 화면에 표시

# 보간법
# 원본 이미지 2확대

# 1) INTER_NEAREST
# => 최근접 이웃 셀 값 복사 (가장 빠름, 품질 낮음, 계단 현상)
resized_nearest = cv2.resize(
    sample_img,
    (width*2, height*2),
    interpolation=cv2.INTER_NEAREST
)

# 2) INTER_LINEAR
# 기본값(default), 적당한 속도와 품질
resized_linear = cv2.resize(
    sample_img,
    (width*2, height*2),
    interpolation=cv2.INTER_LINEAR
)

# 3) INTER_CUBIC
# 4*4 (고품질), 약간 느림
resized_cubic = cv2.resize(
    sample_img,
    (width*2, height*2),
    interpolation=cv2.INTER_CUBIC
)

# 4) INTER_LANCZOS4
# 8*8, 최고품질, 매우 느림
resized_lanczos = cv2.resize(
    sample_img,
    (width*2, height*2),
    interpolation=cv2.INTER_LANCZOS4
)

# 각 방법의 일부분을 확대해서 비교 (차이를 명확히 보기 위해)
# 중앙 부근의 100x100 픽셀 영역 추출
crop_y, crop_x = 280, 280  # 자를 위치 (확대된 이미지 기준)
crop_size = 80  # 자를 크기

cropped_nearest = resized_nearest[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
cropped_linear = resized_linear[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
cropped_cubic = resized_cubic[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
cropped_lanczos = resized_lanczos[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]

# 결과를 2x2 그리드로 표시
fig, axes = plt.subplots(2, 2, figsize=(12, 12))  # 2행 2열 서브플롯
fig.suptitle('Interpolation Methods Comparison (2x Zoom, Cropped)',
             fontsize=16, fontweight='bold')  # 전체 제목

# 확대 결과들 (일부 영역만)
interp_images = [cropped_nearest, cropped_linear, cropped_cubic, cropped_lanczos]
interp_titles = [
    'NEAREST (fastest, lowest quality)',  # 가장 빠름
    'LINEAR (default, balanced)',  # 기본값
    'CUBIC (slow, high quality)',  # 고품질
    'LANCZOS4 (slowest, best quality)'  # 최고품질
]

# 각 subplot에 이미지 표시
for ax, img, title in zip(axes.flat, interp_images, interp_titles):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR을 RGB로 변환
    ax.imshow(img_rgb)  # 이미지 표시
    ax.set_title(title, fontsize=10)  # 제목 설정
    ax.axis('off')  # 축 숨기기

plt.tight_layout()  # 레이아웃 자동 조정
plt.show()  # 화면에 표시

# 아핀 변환
# 3개 점 이용한 변환

# 원본 이미지의 3개의 점 지점(삼각형 모서리)
src_pts = np.float32([
    [50,50],    # 왼쪽 위 점
    [350,50],   # 오른쪽 위 점
    [50,350]    # 왼쪽 아래 점
])

# 목표 위치의 3개 점 지정 (변환 후 위치))
dst_pts = np.float32([
    [80, 100],   # 첫번째 점 이동 [50,50] >> [80,100] 오른쪽,아래 이동
    [320, 80],  # 두번째 점 이동 [350, 50] >> [320, 80] 왼쪽 아래 이동
    [100, 320]  # 세번째 점 이동 [50, 350] >> [100, 320] 오른쪽 위 이동
])

# 아핀변환 행렬 계산
affine_matrix = cv2.getAffineTransform(
    src_pts,    # 원본 3개의 점
    dst_pts     # 목표 3개의 점
)

# 3개의 점 대응 => 2*3행렬이 생성
# array([[ 8.00000000e-01,  6.66666667e-02,  3.66666667e+01],
#        [-6.66666667e-02,  7.33333333e-01,  6.66666667e+01]]

# 아핀변환 적용
affine_result = cv2.warpAffine(
    sample_img,
    affine_matrix,
    (width, height)
)

# 원본 이미지에 점표시
original_with_pts = sample_img.copy()

for pt in src_pts:
    cv2.circle(
        original_with_pts,      # 그림 이미지 
        tuple(map(int, pt)),    # 점 위치
        5,                      # 반지름
        (255,0,0),              # BGR(파란색)
        -1                      # 전부 채워라
    )

affine_with_pts = affine_result.copy()

for pt in dst_pts:
    cv2.circle(
        affine_with_pts,      # 그림 이미지 
        tuple(map(int, pt)),    # 점 위치
        5,                      # 반지름
        (0,0,255),              # BGR(빨간색)
        -1                      # 전부 채워라
    )

fig, axes = plt.subplots(1, 2, figsize=(14, 7))  # 1행 2열 서브플롯
fig.suptitle('Affine Transformation (3 Points)', fontsize=16, fontweight='bold')

# 원본 (파란 점)
axes[0].imshow(cv2.cvtColor(original_with_pts, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original (Blue Points)', fontsize=12)
axes[0].axis('off')

# 변환 결과 (빨간 점)
axes[1].imshow(cv2.cvtColor(affine_with_pts, cv2.COLOR_BGR2RGB))
axes[1].set_title('Affine Transformed (Red Points)', fontsize=12)
axes[1].axis('off')

plt.tight_layout()
plt.show()

'''
# 수동으로 Affine 변환 행렬 만들기
# 행렬 구조 [[a, b, tx], [c, d, ty]]
# a, d : 크기 조절
# b, c : 기울기 (shear)
# tx, ty: 이동
'''

# 1) 단순이동(translation)
translate_matrix = np.float32([
    [1, 0, 50],     # x축: 크기 유지, 기울기 없음, 오른쪽 50px 이동
    [0, 1, 30]      # y축: 크기 유지, 기울기 없음, 아래로 30px 이동
])

translated = cv2.warpAffine(sample_img, translate_matrix, (width, height))

# 2) 수평 기울이기(Horizontal Shear)
shear_x_matrix = np.float32([
    [1, 0.3, 0],    # x축: 변화 없음        
    [0, 1, 0]       # y축: 크기 유지, x값에 따라서 y값 이동(기울임 이동없음)
])

sheared_x = cv2.warpAffine(sample_img, shear_x_matrix, (width, height))

# 2) 수평 기울이기(Horizontal Shear)
shear_y_matrix = np.float32([
    [1, 0, 0],        # x축: 크기 유지, y값에 따라서 x값 이동(기울임 이동없음)        
    [0.3, 1, 0]       # y축: 변화 없음
])

sheared_y = cv2.warpAffine(sample_img, shear_y_matrix, (width, height))

# 복합변환(회전 + 크기 + 이동)
# cos(30도) 0.866에 근사한 값 가짐, sin(30도) 0.5에 근사치 값 가짐
angle_rad = np.radians(30)
# np.float64(0.5235987755982988)
cos_val = np.cos(angle_rad)   # 코사인 값 계산
# np.float64(0.8660254037844387)
sin_val = np.sin(angle_rad)   # 사인 값 계산
# np.float64(0.49999999999999994)
scale = 0.8     # 0.8배 축소

complex_matrix = np.float32([
    [cos_val * scale, -sin_val * scale, 50],    # 회전, 축소, 이동
    [sin_val * scale, cos_val * scale, 80]
])

complex_transformed = cv2.warpAffine(sample_img, complex_matrix, (width, height+150))

fig, axes = plt.subplots(2, 2, figsize=(14, 14))
fig.suptitle('Manual Affine Transformations', fontsize=16, fontweight='bold')

# 변환 결과들
affine_results = [translated, sheared_x, sheared_y, complex_transformed]
affine_titles = [
    'Translation (50, 30)',  # 이동
    'Horizontal Shear (0.3)',  # 수평 기울임
    'Vertical Shear (0.3)',  # 수직 기울임
    'Rotate 30° + Scale 0.8 + Translate'  # 복합 변환
]

# 각 subplot에 이미지 표시
for ax, img, title in zip(axes.flat, affine_results, affine_titles):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)
    ax.set_title(title, fontsize=11)
    ax.axis('off')

plt.tight_layout()
plt.show()

# Perspective 반환: 4개의 점을 활용, 원근 반환
# 비스듬히 찍힌 사진을 정면으로 보정할 때 사용

# 원본 이미지의 4개 꼮지점 (사각형)
src_pts_persp = np.float32([
    [0, 0],           # 왼쪽 위
    [width-1, 0],     # 오른쪽 위
    [width-1, height-1],  # 오른쪽 아래
    [0, height-1]     # 왼쪽 아래
])

# 목표 위치 4개 점 (사다리꼴 모양으로 변환)
dst_pts_persp = np.float32([
    [50, 100],        # 왼쪽 위 → 오른쪽+아래 이동
    [width-50, 100],  # 오른쪽 위 → 왼쪽+아래 이동
    [width-20, height-50],  # 오른쪽 아래 → 왼쪽+위 이동
    [20, height-50]   # 왼쪽 아래 → 오른쪽+위 이동
])
# 결과: 위쪽이 좁고 아래쪽이 넓은 사다리꼴 (원근감)

# Perspective 변환 행렬 계산 (3*3 행렬)
perspective_matrix = cv2.getPerspectiveTransform(
    src_pts_persp,  # 원본 4점
    dst_pts_persp   # 목표 4점
)
#  결과
# [[ 7.51879699e-01 -8.35421888e-02  5.00000000e+01]
#  [ 0.00000000e+00  4.80367586e-01  1.00000000e+02]
#  [ 0.00000000e+00 -4.17710944e-04  1.00000000e+00]]

perspective_result = cv2.warpPerspective(
    sample_img,
    perspective_matrix,         # 3*3 변환행렬
    (width, height),            # 출력 크기
    borderValue=(200,200,200)   # 빈 공간을 회색으로 채움
)  

# 시각화: 원본과 목표 점들을 선으로 연결
original_persp = sample_img.copy()
perspective_persp = perspective_result.copy()

# 원본 이미지에 점과 선 그리기
for i, pt in enumerate(src_pts_persp):
    # 점 그리기
    cv2.circle(
        original_persp,
        tuple(pt.astype(int)),
        8,
        (255, 0, 0),  # 파란색
        -1
    )
    # 점 번호 표시
    cv2.putText(
        original_persp,
        str(i+1),
        tuple((pt + [10, 10]).astype(int)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

# 사각형 테두리 그리기
cv2.polylines(
    original_persp,
    [src_pts_persp.astype(int)],  # 점들을 int로 변환
    True,  # 닫힌 도형
    (255, 0, 0),  # 파란색
    3  # 선 두께
)

# 변환된 이미지에 점과 선 그리기
for i, pt in enumerate(dst_pts_persp):
    cv2.circle(
        perspective_persp,
        tuple(pt.astype(int)),
        8,
        (0, 0, 255),  # 빨간색
        -1
    )
    cv2.putText(
        perspective_persp,
        str(i+1),
        tuple((pt + [10, 10]).astype(int)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

cv2.polylines(
    perspective_persp,
    [dst_pts_persp.astype(int)],
    True,
    (0, 0, 255),  # 빨간색
    3
)

# 결과 표시
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.suptitle('Perspective Transformation (4 Points)', fontsize=16, fontweight='bold')

# 원본 (파란색 사각형)
axes[0].imshow(cv2.cvtColor(original_persp, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original (Blue Rectangle)', fontsize=12)
axes[0].axis('off')


# 변환 결과 (빨간색 사다리꼴)
axes[1].imshow(cv2.cvtColor(perspective_persp, cv2.COLOR_BGR2RGB))
axes[1].set_title('Perspective Transformed (Red Trapezoid)', fontsize=12)
axes[1].axis('off')

plt.tight_layout()
plt.show()

# 실전 예제 (문서 스케너 원리)

# step 1 : 정면 >> 비스듬하게 (원근 효과 추가)
src = np.float32([
    [50,50],
    [width-50, 50],
    [width-50, height-50],
    [50, height-50]
])

# 비스듬한 객체 생성
dst_skew = np.float32([
    [100,50],
    [width-50, 80],
    [width-80, height-100],
    [80, height-80]
])

skew_mat = cv2.getPerspectiveTransform(src, dst_skew)  # 변환 행렬
skew_doc = cv2.warpPerspective(
    sample_img,
    skew_mat,
    (width, height),
    borderValue=(200,200,200)
)

# step 2 : 비스듬한 객체 => 복원
# 역변환을 위해서 원본과 목표를 바꿈

correct_mat = cv2.getPerspectiveTransform(dst_skew, src)

correct_doc = cv2.warpPerspective(
    skew_doc,
    correct_mat,
    (width, height),
    borderValue=(255,255,255)
)

# step 3 : 원본 > 비스듬한 객체 > 보정(복원)

fig, axes = plt.subplots(1, 3, figsize=(18,6))
fig.suptitle('DOC READER', fontsize=16, fontweight='bold')

# 원본
axes[0].imshow(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
axes[0].set_title('Step 1. Original DC', fontsize=12)
axes[0].axis('off')

# 비스듬한 문서
axes[1].imshow(cv2.cvtColor(skew_doc, cv2.COLOR_BGR2RGB))
axes[1].set_title('Step 2. Skewd Doc(Camera View)', fontsize=12)
axes[1].axis('off')

# 우리 리더기를 통한 복원 문서
axes[2].imshow(cv2.cvtColor(correct_doc, cv2.COLOR_BGR2RGB))
axes[2].set_title('Step 3. Correct_doc', fontsize=12)
axes[2].axis('off')

plt.tight_layout()
plt.show()