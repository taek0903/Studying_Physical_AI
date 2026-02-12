import os, sys
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


file_path = r'D:\rokey\AI_applications\pr0212\tiny_nerf_data.npz'
data = np.load(file_path)   # NeRF 학습에 필요한 데이터를 NumPy 파일에서 로드
images = data['images']     # RGB 이미지 데이터 (N, H, W, 3)
poses = data['poses']       # 각 이미지에 대한 카메라 포즈 행렬 (N, 4, 4)
focal = data['focal']       # 카메라의 초점 거리 (f)
H, W = images.shape[1:3]    # 이미지의 높이(H)와 너비(W) 추출
print(images.shape, poses, focal)   # 로드된 데이터의 형태와 초점 거리 출력 (디버깅/확인용)

testing, testpose = images[101], poses[101] # 102번째 이미지와 포즈를 렌더링 테스트용으로 분리
images = images[:100, ..., :3]
'''

'''
poses = poses[:100]

# plt.imshow(testing)
# plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_default_dtype(torch.float32)

L_embed = 6

class NeRF(torch.nn.Module):
    """
    NeRF MLP model  # NeRF의 신경 방사 필드(NeRF)를 구현하는 MLP 모델 클래스
    """
    def __init__(self, filter_size=128, L_embed=6):
        super(NeRF, self).__init__()
        # 입력 레이어: (3차원 좌표 + 위치 인코딩된 좌표)를 입력으로 받음.
        # 입력 차원: 3 (원본 xyz) + 3 * 2 * L_embed (인코딩된 sin/cos 쌍)
        self.layer1 = torch.nn.Linear(3 + 3*2*L_embed, filter_size)
        self.layer2 = torch.nn.Linear(filter_size, filter_size)
        # 출력 레이어: 4차원 (RGB 색상 3차원 + Volume Density(부피 밀도) 1차원)
        self.layer3 = torch.nn.Linear(filter_size, 4) # 4 => 밀도
        self.relu = torch.nn.functional.relu # 활성화 함수로 ReLU 사용

    def forward(self, x):
        # 3개의 선형 레이어와 2개의 ReLU를 통과하는 순전파 로직
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        # layer1, layer2는 파동(인코딩 된 벡터)조합하는 단계
        x = self.layer3(x) 
        # 마지막 출력 레이어는 활성화 함수를 적용하지 않음
        # 왜냐하면 밀도는 항상 양수(0보다 커야 함)
        # 색상(rgb) [0,1] (sigmoid) 통과시켜서 값 제한함
        return x
    
def posenc(x):
    """
    Positional encoding  # 위치 정보에 고주파 정보를 추가하는 함수 (NeRF의 핵심)
    """
    rets = [x] # 원본 좌표(x, y, z)를 결과 리스트에 추가
    # L_embed 만큼의 주파수 대역에 대해 sin과 cos 함수를 적용하여 고차원 벡터 생성
    for i in range(L_embed):
        for fn in [torch.sin, torch.cos]:
            rets.append(fn(2.0 ** i * x)) # 2^i 주파수로 변환하여 리스트에 추가
    return torch.cat(rets, dim=-1) # 모든 결과를 마지막 차원(-1)으로 연결하여 반환

# 함수 결과물? 주파수가 2배씩 증가>> 다양한 스케일의 패턴 포착

embed_fn = posenc # 위치 인코딩 함수에 별칭을 부여하여 이후 사용 편의성 확보

# 왜 주파수(frequency)인가?
# sin, cos >> 파동(1초에 몇 번 흔들리는가, 진동수)
# sin(x) : 낮은 주파수 (천천히 흔들림) cf.sin(16x) 높은 주파수 매우 심하게 (빠르게) 흔들림
# 낮은 주파수: 전체인 구조 파악, 높은 주파수: 디테일 (테두리, 모서리(엣지), 질감)

def get_rays(H, W, focal, c2w):
    """
    Get ray origin, direction to each image pixels based on camera pose
    # 카메라 포즈(c2w: camera to world)를 기반으로 모든 픽셀에 대한 레이의 원점과 방향을 계산
    # 카메라에서 각 픽셀로 광선을 발사하는 함수

    볼룸 렌더링: 광선을 따라 색상을 적분하여 최종 픽셀 색상 계산

    과정:
    1. 광선을 따라 샘플 포인트 생성
    2. 각 포인트에서 색상과 밀도 예측
    3. 볼륨 렌더링 공식으로 최종 색상 계산

    """
    def meshgrid_xy(tensor1, tensor2):
        # PyTorch의 meshgrid 결과를 Numpy/TensorFlow와 호환되도록 재배치
        i, j = torch.meshgrid(tensor1, tensor2, indexing="ij")
        return i.transpose(-1, -2), j.transpose(-1, -2)
        # 이미지 좌표 (row, col) 즉 (y, x)

    # 픽셀 좌표 (i, j) 생성
    i, j = meshgrid_xy(
        torch.arange(W, dtype=torch.float32).to(device),
        torch.arange(H, dtype=torch.float32).to(device)
    )
    # 카메라 좌표계(Camera Coordinates)에서의 레이 방향 벡터 계산 (광축: -z)
    dirs = torch.stack([(i-W*0.5)/focal, -(j-H*0.5)/focal, -torch.ones_like(i)], dim=-1).to(device)
    # (i-W*0.5)/focal x 방향(중심 기준) : pixel 위치를 중심 기준 중심 거리로 변경한 후, 초점거리(focal) 나눔 => 각도(기울기)
    # -(j-H*0.5)/focal y방향 (위가 음수) : 이미지 y 좌표 아래로 갈수록 커짐. 3D 공간 y축은 위로 갈수록 커짐
    # 그렇다면 -(마이너스) 부호는? 마이너스 붙여야 같은 방향 
    # -torch.ones_like(i)] (카메라 앞쪽) 카메라가 바라보는 정면 방향 (-z방향)
    # 레이 방향: 카메라 좌표를 월드 좌표로 변환 (회전 행렬 c2w[:3, :3] 사용)
    # 카메라 방향 회전(dir.shape = (H,W,3), 여기서 3은 [dx,dy,dz] 방향 벡터)
    # => dirs[h,w] = [dx,dy,dz]
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)    # 회전 
    # '...' : 앞 차원을 전부가져와라
    # dirs[...:] = dirs[:,:,:] => 여기에 차원 하나 추가 (새로운 차원 하나 추가)
    # dirs[...,None, :] (h,w,1,3) == dirs[:,:,None, :]

    # shape 변화 추적 
    # 레이 원점(카메라 위치, 광선 시작점): 카메라 위치 (c2w[:3, -1])를 모든 레이 방향의 수만큼 확장
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    # 4*4 행렬 마지막(맨 오른쪽) 열
    # (카메라가 월드좌표계 어디에 있는지 알려줌: 카메라 실제위치)(x,y,z좌표) => 이동 값
    # 모든 광선은 카메라 위치에서 시작함 => expand 모든 픽셀에 대해 동일한 시작점 부여해서 확장
    return rays_o, rays_d # 월드 좌표계에서의 레이 원점과 방향 반환
    # 결론: 모든 픽셀에 대해 광선(ray)가 시작되는 월드좌표 할당해라

def render_rays(network_fn, rays_o, rays_d, near, far, N_samples, rand=False):
    """
    Volume rendering # NeRF 모델을 사용하여 샘플링된 점들을 따라 볼륨 렌더링을 수행
    => 쉽게 말하자면, 광선(ray)을 따라서 색상 계산해줘
    => 이거 다음에 뭐해? 각 점의 색상(rgb). 밀도나오면 그거 가지고 가중평규해서 최종색 계산
    """
    def batchify(fn, chunk=1024*32):
        # 메모리 효율을 위해 대량의 입력을 청크(chunk)로 나누어 MLP를 호출
        # fn: 실제 계산할 함수(mlp), chunk(한 번에 처리할 최대 입력 수) = 32768
        return lambda inputs : torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], dim=0)
        # inputs 100,000개
        # 1번째 : inputs[0: 32768]
        # 2번째 : inputs[32768:65536]
        # 3번째 : inputs[65536:98304]
        # 4번째: inputs[98304:100,000]
        # 이렇게 4번째 잘라서 fn(mlp) 넣는다
        # 최종 결과물은 원래 입력길이와 동일한 구조로 복원

    def exclusive_cumprod(tensor):
        # 누적 곱을 계산한 뒤 한 칸 오른쪽으로 시프트하여 T_i (투과율) 계산에 사용
        # 배타적 누적 곱 : i 번째 위치의 값은 자기 자신을 곱하지 않은 이전 값들의 누적 곱임
        cumprod = torch.cumprod(tensor, dim=-1)
        cumprod = torch.roll(cumprod, 1, dims=-1)
        cumprod[..., 0] = 1.0 # 첫 번째 값은 1.0으로 설정 (첫 번째 샘플까지의 투과율)
        return cumprod
        # T_i (투과율) i번째 이전 값들의 누적 곱만 포함하고 , 자기자신 alpha_i(샘플의 불투명도)포함
        # 즉, 각 샘플의 투과율 (T_i) 자기 이전의 샘플들만 누적
        # [0.7, (0.7*0.5), (0.7*0.5*0.8)] => (roll 1칸 밀어) => [(0.7*0.5*0.8), 0.7, (0.7*0.5)]
        # => ... : 무시하고 (0) 초기화 해 [1. 0.7. 0.35]
        # 여기에 투과율(살아 남아서 내 눈 앞에 보이는 거) 그 지점의 색(rgb), 불투명도(alpha) 곱함
        # => 볼륨 랜더링

    # --- 3D 쿼리 점 계산 (Sample 3D Points) ---
    z_vals = torch.linspace(near, far, N_samples).to(device) # 레이를 따라 균일하게 깊이 값(z) 생성
    if rand:
        # 훈련 시 정규화를 방지하기 위해 랜덤한 오프셋 추가 (Hierarchical Sampling 이전 단계)
        z_vals = z_vals + torch.rand(list(rays_o.shape[:-1]) + [N_samples]).to(device) * (far-near)/N_samples
        # rand() : 0-1 난수
        # rays_o.shape = (1024, 3) >> [:-1] >> (1024, ) >> (1024,N_samples)
        # ray(광선)마다 n개 난수 생성
        # (far-near) 전체 구간, (far-near)/N_samples : 각 샘플당 차지하는 구간 >> 각 샘플 간 간격

    # 3D 점 좌표 계산: pts = o + t * d (레이 원점 + 깊이 * 레이 방향)
    # 카메라 쏜 광선을 따라가면서 z_vals 깊이에 해당하는 3d 점들 생성
    # None 차원확장 역할
    # 2차원을 3차원으로 만드는 과정
    # 포토카드의 pixel을 이용해서 광선 100개 만들었어. (1000개의 광선이 있고, 각각 (x,y,z) 위치 정보 있어)
    # z_vals (n_rays 광선개수, n_samples ) 각 광선마다 100개의 점 찍을 수 있어.
    # rays_o[..., None, :] (n_rays, 1, 3)
    # z_vals[..., :, None] (n_rays, n_samples, 1)
    # 광선 하나 당 샘플 100개 점 전부 대응해 계산해줘

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    # 해당 공간의 3d 점 = 카메라 위치 + (방향*거리)
    # (방향 * 거리) => 어느 방향으로 얼마나 보낼까
    # rays_o[..., None, :] (h,w,c) >> (h,w,3) >> (N-rays, 3) >> (N-rays, 1, 3)
    # rays_d[..., None, :] rays_d : 각 광선의 1unit 당 direction(방향) 벡터 (길이 1 방향 벡터)
    # >> (N-rays, 1, 3)
    # z_vals[..., :, None] 깊이 샘플값(N_samples) (N_rays, N_samples, 1)
    # rays_d[..., None, :] * z_vals[..., :, None]
    # >> 각 ray(광선) 방향을 z(distance) 만큼 이동한 3d 좌표 변화량
    # 1000개의 광선을 쏴서 그 각각의 광선이 100개씩 찍힌 점들의
    # (x,y,z)좌표

    # --- NeRF MLP 실행 (Run Network) ---
    pts_flat = torch.reshape(pts, [-1, 3]) # 3D 점들을 평탄화하여 MLP 입력 형태로 만듦
    # (N_rays, N_samples, 3) >> (batch_size, 3)  # batch_size, [x,y,z]
    pts_flat = embed_fn(pts_flat) # 위치 인코딩 적용
    raw = batchify(network_fn)(pts_flat) # NeRF MLP를 청크 단위로 실행
    raw = torch.reshape(raw, list(pts.shape[:-1]) + [4])
    # 결과를 (레이 수, 샘플 수, 4) 형태로 복원
    # 4 가 뭐지? (r, g, b), d (density: 밀도)
    # mlp 통과하면 (N_samples, 4)


    # --- Opacities and Color (색상 및 밀도 추출) ---
    sigma_a = torch.nn.functional.relu(raw[..., 3])
    # raw[..., 3] 3: density (밀도=질량/부피) 밀도는 음수가 될 수 없어
    # relu 통과 시키면, 음수 >> 0 만들어 줌
    # 부피 밀도(sigma): ReLU를 적용하여 양수로 만듦

    rgb = torch.sigmoid(raw[..., :3])
    # 색상(RGB): 시그모이드(Sigmoid)를 적용하여 [0, 1] 범위로 만듦
    # rgb [0,255] >> torch 형태 [0,1]

    # --- 볼륨 렌더링 공식 적용 (Perform Volume Rendering) ---
    # 인접 샘플 간의 거리 계산 (마지막은 무한대로 가정)
    dists = torch.cat([z_vals[..., 1:] - z_vals[..., :-1],
                       torch.tensor([1e10], device=device).expand(z_vals[..., :1].shape)], dim=-1)
    # z_vals : 광선(ray) 을 따라 샘플링한 깊이 값 (인접한 두 점의 차이 = 거리)
    # 1e10 사실상 무한대 마지막 뒤로는 무한대다 => 모든 방향으로 조정
    # 투과율(T)과 밀도(sigma)를 사용하여 알파 값(가중치) 계산: alpha = 1 - exp(-sigma * delta)
    # 밀도 =질량 / 부피 >> 밀도가 높으면 빛이 더 많이 막힌다 >> alpha가 높아진다
    # 밀도가 0 >> 아무 것도 없다 >> alpha=0
    alpha = 1.0 - torch.exp(-sigma_a * dists)
    # 밀도 * 두께(dist) 크면 alpha(불투명도) => 1에 가까워짐 (빛이 소멸)
    # 볼륨 렌더링 가중치 계산: weight = T * alpha
    # 누적 투과율 사용
    weights = alpha * exclusive_cumprod(1.0-alpha + 1e-10)
    # 최종 색상 (RGB Map) 계산: C = sum(weight * color)
    # >> 볼륨 렌더린 된 최종 색상
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    # weights: (n_rats, n_samples), rgb(n_rays, n_sample, 3)
    # weights: (n_rats, n_samples) => weights: (n_rays, n_sample, 1) 차원 맞춰줌
    # dim=-2 => n_samples
    # => n_samples 차원에 맞추어서 광선을 따라 만들어진 점 100개의 점을 다 더해서 하나로 합쳐줘
    # => 결과물 (n_rays, 3)
    # 깊이 맵 (Depth Map) 계산: D = sum(weight * z_val)
    # 각 샘플 z 위치에 가중치를 곱한 뒤 합해주는 거네
    # >> 빛이 많이 닿는 위치일 수록 깊이에 반영되겠네

    depth_map = torch.sum(weights * z_vals, dim=-1)
    # 누적 투과율 (Accumulation Map) 계산: Acc = sum(weight)
    acc_map = torch.sum(weights, dim=-1)
    # 누적 투과율 acc_map (광선을 따라 존재하는 모든 가중치 다 더해)
    # 투과율이 높으면 전체 값이 낮아짐

    return rgb_map, depth_map, acc_map # 최종 렌더링된 색상, 깊이, 누적 투과율 반환

data = np.load(r'D:\rokey\AI_applications\pr0212\tiny_nerf_data.npz') # NumPy 파일에서 NeRF 데이터 로드
images = data['images']
poses = data['poses']
focal = data['focal']

# NumPy 데이터를 PyTorch Tensor로 변환하고 GPU/CPU 장치로 이동
images = torch.from_numpy(images).to(device)
poses = torch.from_numpy(poses).to(device)
focal = torch.from_numpy(focal).to(device)

H, W = images.shape[1:3] # 이미지 높이(H)와 너비(W) 추출

# Define test set, training set
testimg, testpose = images[101], poses[101] # 102번째 데이터를 테스트용으로 분리
images = images[:100,...,:3] # 처음 100개 이미지를 학습 데이터로 사용 (RGB만)
poses = poses[:100] # 처음 100개 포즈를 학습 데이터로 사용

# Hyperparameters
N_samples = 64 # 각 광선(Ray)을 따라 샘플링할 3D 점의 개수
N_iters = 10000 # 총 학습 반복 횟수 (Iteration)
psnrs = [] # PSNR 값 기록 리스트(테스트 이미지의 psnr 값 저장)
iternums = [] # 반복 횟수 기록 리스트
i_plot = 100 # 테스트 이미지 렌더링 및 시각화 빈도
lr = 5e-3 # 학습률 (Learning Rate)

# psnr (peak single to noise ratio)
# 이미지 품질 평가 지표
# mse 이 적을 수록 에러가 적어짐
# >> psnr 은 증가한다

# Define NeRF model, optimizer
model = NeRF() # NeRF MLP 모델 생성
model = model.to(device) # 모델을 지정된 장치(GPU/CPU)로 이동
optimizer = torch.optim.Adam(model.parameters(), lr=lr) # Adam 옵티마이저 정의

# NeRF optimization loop
print("NeRF optimization start")
plot_image = True # 시각화 여부 플래그

for i in tqdm(range(N_iters)): # 설정된 반복 횟수만큼 학습 루프 시작 (tqdm으로 진행 상황 표시)
    img_i = np.random.randint(images.shape[0]) # 학습 데이터에서 무작위로 하나의 이미지 인덱스 선택
    target = images[img_i].to(device) # 타겟 이미지 (정답)
    pose = poses[img_i].to(device) # 해당 이미지의 카메라 포즈

    # 1. Ray Generation: 무작위로 선택된 포즈에서 나오는 모든 광선(rays) 계산
    # 카메라 좌표계에서 각 픽셀방향으로 광선을 쏜다
    rays_o, rays_d = get_rays(H, W, focal, pose)

    # 2. Volume Rendering: 광선을 따라 3D 샘플링 및 NeRF 모델 쿼리 후 렌더링
    # rand=True를 사용하여 훈련 시 샘플링 깊이에 무작위 오프셋 추가 (노이즈 방지)
    rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2.0, far=6.0, N_samples=N_samples, rand=True)
    # 각 ray(광선) 따라서 64개 점 샘플링
    # NeRF MLP 통과 >> 각각의 색상, 밀도 예측
    # 볼륨렌더링 최종 C(rgb), depth

    # Back propagation
    # 3. Loss Calculation: 렌더링된 RGB와 타겟 이미지 간의 MSE 손실 계산
    loss = torch.nn.functional.mse_loss(rgb, target)
    loss.backward() # 역전파 (기울기 계산)
    optimizer.step() # 옵티마이저를 통해 모델 파라미터 업데이트
    optimizer.zero_grad() # 다음 반복을 위해 기울기 초기화

    if plot_image:
        if i % i_plot == 0: # i_plot 간격마다 테스트 및 시각화 수행
            # --- Test Rendering (Evaluation) ---
            # 테스트 포즈를 사용하여 광선 계산 (rand=False로 확정적 샘플링)
            rays_o, rays_d = get_rays(H, W, focal, testpose)
            # 테스트 이미지 렌더링
            rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2.0, far=6.0, N_samples=N_samples)

            # --- PSNR Calculation ---
            loss = torch.nn.functional.mse_loss(rgb, testimg) # 테스트 이미지와의 손실
            # PSNR(Peak Signal-to-Noise Ratio) 계산: 이미지 품질 측정 지표
            psnr = -10.0 * torch.log10(loss)

            # --- Record ---
            psnrs.append(psnr.item())
            iternums.append(i)

            # --- Visualization ---
            plt.figure(figsize=(12, 4))

            # 렌더링된 RGB 이미지 시각화
            plt.subplot(131)
            plt.imshow(rgb.cpu().detach().numpy())
            plt.title(f"Iteration {i}")

            # PSNR 변화 그래프 시각화
            plt.subplot(132)
            plt.plot(iternums, psnrs)
            plt.title("PSNR")

            # 렌더링된 깊이 맵(Depth Map) 시각화
            plt.subplot(133)
            plt.imshow(depth.cpu().detach().numpy(), cmap="gray")
            plt.title("Depth Map")

            # 파일 저장
            filename=os.path.join(r'D:\rokey\AI_applications\pr0212', f'iteration_{i}.png')
            plt.savefig(filename)

            # 플롯 창 자동 닫기 설정
            plt.show(block=False)
            plt.pause(1) # 1초 동안 잠시 화면 표시
            plt.close() # 플롯 창 닫기 (메모리 관리)

print("Done")
