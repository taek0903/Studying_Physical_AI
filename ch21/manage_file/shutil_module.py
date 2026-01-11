# shutil_module.py

import os
import shutil

# (1) 파일 복사
path = r'D:\rokey\python\ch21\manage_file'
src = f'{path}/file.txt'
dst = f'{path}/copiedfile.txt'

if not os.path.exists(dst):
    # shutil.copy(src, dst)
    shutil.copy2(src, dst)

# (2) 디렉토리 복사
# 디렉토리 생성
folder = f'{path}/test_dir'
if not os.path.exists(folder):
    os.mkdir(folder)

# 파일 복사
src = f'{path}/file.txt'
dst = f'{path}/test_dir/file.txt'
if not os.path.exists(dst):
    shutil.copy(src,dst)

# 디렉토리 전체 복사
src = f'{path}/test_dir'
dst = f'{path}/copied_dest_dir'

if not os.path.exists(dst):
    shutil.copytree(src,dst)

print("----------------------")
# (3) 파일 및 디렉토리 이동
src = f'{path}/copiedfile.txt'
dst = f'{path}/copied_dest_dir/copiedfile.txt'

shutil.move(src,dst)

print("----------------------")
# (4) 파일 및 디렉토리 삭제
dir1 = f'{path}/test_dir'
dir2 = f'{path}/copied_test_dir'
shutil.rmtree(dir1,dir2)