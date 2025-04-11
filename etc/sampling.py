import os
import shutil
import glob
import re
import tkinter as tk
from tkinter import filedialog, messagebox


def sample_images_by_ratio(source_folder: str, destination_folder: str, sampling_ratio: int = 30):
    """
    각 하위 폴더의 파일 이름의 숫자 부분에서 sampling_ratio로 나눴을 때 1이 되는 파일만 복사하는 함수

    Args:
    - source_folder (str): 원본 이미지가 있는 폴더 경로 (하위 폴더 포함)
    - destination_folder (str): 샘플링된 이미지가 저장될 폴더 경로
    - sampling_ratio (int): 샘플링 비율 (기본값: 30)
    """
    for subdir, _, _ in os.walk(source_folder):
        # 하위 폴더 경로 얻기
        relative_path = os.path.relpath(subdir, source_folder)
        target_folder = os.path.join(destination_folder, relative_path)

        # 대상 폴더 생성 (하위 폴더 구조 유지)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        # 하위 폴더 내 파일 선택 (frame+숫자 또는 combined_processed_+숫자 형식)
        image_files = glob.glob(os.path.join(subdir, "*"))

        for file_path in image_files:
            file_name = os.path.basename(file_path)
            # 파일 이름에서 숫자 추출 (frame+숫자 또는 combined_processed_+숫자)
            match = re.search(r"(?:frame|combined_processed_)(\d+)", file_name)
            if match:
                number = int(match.group(1))
                # 조건: sampling_ratio로 나눴을 때 0인 경우만 복사
                if number % sampling_ratio == 0:
                    destination_path = os.path.join(target_folder, file_name)
                    shutil.copy(file_path, destination_path)

    messagebox.showinfo("완료", f"샘플링 기준에 맞는 파일들이 '{destination_folder}' 폴더에 복사되었습니다.")


def select_folders():
    root = tk.Tk()
    root.withdraw()

    # 원본 이미지 폴더 선택
    source_folder = filedialog.askdirectory(title="원본 이미지 폴더를 선택하세요")
    if not source_folder:
        messagebox.showwarning("경고", "원본 폴더를 선택하지 않았습니다.")
        return

    # 샘플링된 이미지 저장 폴더 선택
    destination_folder = filedialog.askdirectory(title="샘플링된 이미지를 저장할 폴더를 선택하세요")
    if not destination_folder:
        messagebox.showwarning("경고", "저장 폴더를 선택하지 않았습니다.")
        return

    # 샘플링 함수 실행 (샘플링 비율은 코드 내에서 설정)
    sample_images_by_ratio(source_folder, destination_folder)


# 폴더 선택 및 샘플링 함수 호출
select_folders()
