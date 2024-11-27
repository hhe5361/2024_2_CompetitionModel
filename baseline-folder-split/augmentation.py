import os
import shutil
from tqdm import tqdm

# 기존 'augmented' 폴더 경로
augmented_dir = '/data/coqls1229/repos/mip_res34/Galaxy10/augmented'
# 결과를 저장할 새로운 'augmented_result' 폴더 경로
augmented_result_dir = '/data/coqls1229/repos/mip_res34/Galaxy10/augmented_result'

# augmented_result_dir 경로가 없다면 생성
if not os.path.exists(augmented_result_dir):
    os.makedirs(augmented_result_dir)

# augmented 폴더 내의 augmentation 폴더를 순회
for aug_folder in tqdm(os.listdir(augmented_dir), desc="Processing augmentation folders"):
    aug_folder_path = os.path.join(augmented_dir, aug_folder)
    
    # augmentation 폴더가 디렉토리인 경우에만 처리
    if os.path.isdir(aug_folder_path):
        # 각 augmentation 폴더 하위의 클래스별 폴더를 순회
        for class_folder in os.listdir(aug_folder_path):
            class_folder_path = os.path.join(aug_folder_path, class_folder)
            
            # 클래스 폴더가 디렉토리인 경우에만 처리
            if os.path.isdir(class_folder_path):
                # 'augmented_result' 폴더 내에 클래스 폴더가 없다면 생성
                class_result_folder = os.path.join(augmented_result_dir, class_folder)
                if not os.path.exists(class_result_folder):
                    os.makedirs(class_result_folder)
                
                # 클래스 폴더 내의 모든 .jpg 파일을 순회
                for img_name in os.listdir(class_folder_path):
                    img_path = os.path.join(class_folder_path, img_name)
                    
                    # .jpg 파일인 경우에만 처리
                    if img_path.lower().endswith('.jpg'):
                        # 파일명이 겹칠 수 있기 때문에, 고유한 파일명으로 저장
                        new_img_name = f"{os.path.splitext(img_name)[0]}_{aug_folder}{os.path.splitext(img_name)[1]}"
                        new_img_path = os.path.join(class_result_folder, new_img_name)
                        
                        # 이미 존재하는 파일명이 있으면 번호를 붙여서 고유하게 만듦
                        counter = 1
                        while os.path.exists(new_img_path):
                            new_img_name = f"{os.path.splitext(img_name)[0]}_{aug_folder}_{counter}{os.path.splitext(img_name)[1]}"
                            new_img_path = os.path.join(class_result_folder, new_img_name)
                            counter += 1
                        
                        # 이미 존재하는 클래스 폴더로 이미지 파일을 이동
                        shutil.move(img_path, new_img_path)

print("Data has been successfully consolidated into the 'augmented_result' folder!")