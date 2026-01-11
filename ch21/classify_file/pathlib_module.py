# pathlib_module.py
# python\ch21\classify_file
from pathlib import Path

# 현재 작업 디렉토리 확인
print(Path.cwd())

print("--------------------")
# 경로 생성 및 조작
path = Path(r'python\ch21\classify_file')
print(path)
print(path / "file.txt")

print("--------------------")
# 디렉토리 및 파일 존재 여부 확인
file = rf'python\ch21\classify_file/file.txt'
path = Path(file)
print(path.exists())    # 존재 여부 확인
print(path.is_file())   # 존재 및 파일 여부 확인
print(path.is_dir())    # 디렉토리 여부 확인

print("--------------------")
# 디렉토리 생성 및 삭제
new_folder = "python\ch21\classify_file/new_folder"
path = Path(new_folder)
path.mkdir(exist_ok=True)   # 폴더 생성
path.rmdir()

print("--------------------")
# 파일 생성 및 삭제
file = "python\ch21\classify_file/file.txt"
file_path = Path(file)
file_path.touch()   # 빈 파일(용량이 0인) 생성
file_path.unlink()  # 파일 삭제

print("--------------------")
# 파일 및 폴더 목록 조회
dir = "python\ch21\classify_file/"
path = Path(dir)
for item in path.iterdir():
    print(item)