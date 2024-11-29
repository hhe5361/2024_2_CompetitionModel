import numpy as np
import cv2
import os
from PIL import Image
import random
new_folder = "/data/hhe5361/downloads/MainGalaxy10/train"
ori_folder = "/data/hhe5361/downloads/Galaxy10/train"

#salt and pepper noise 추가한 거
def add_salt_and_pepper_noise(image, salt_prob = 0.007, pepper_prob = 0.007):
    """
    이미지에 salt and pepper 노이즈 추가.
    :param image: 원본 이미지 (BGR 형식)
    :param salt_prob: salt 노이즈의 확률 (0 ~ 1)
    :param pepper_prob: pepper 노이즈의 확률 (0 ~ 1)
    :return: 노이즈가 추가된 이미지
    """
    # 이미지 복사
    noisy_image = image.copy()

    # Salt 노이즈 추가
    salt_mask = np.random.rand(*image.shape[:2]) < salt_prob
    noisy_image[salt_mask] = 255  # salt는 흰색 (255)

    # Pepper 노이즈 추가
    pepper_mask = np.random.rand(*image.shape[:2]) < pepper_prob
    noisy_image[pepper_mask] = 0  # pepper는 검은색 (0)

    return noisy_image

def create_center_weighted_mask(image_shape):
    height, width = image_shape
    Y, X = np.ogrid[:height, :width]
    center_y, center_x = height//2, width//2
    
    # 가우시안 마스크 생성
    mask = np.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2.0 * (min(width, height)/4)**2))
    return mask

def apply_center_brightness_weight(image):
    # 가우시안 마스크 생성
    center_mask = create_center_weighted_mask(image.shape[:2])
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 밝기 정규화
    normalized_brightness = gray / 255.0
    
    # 가중치 적용
    weighted_image = image * center_mask[..., np.newaxis]
    weighted_image = weighted_image * normalized_brightness[..., np.newaxis]
    
    return weighted_image.astype(np.uint8)

def apply_median_and_gaussian_filters(image, median_kernel_size=3, gaussian_kernel_size=5, gaussian_sigma=1.5):
    """
    이미지에 Median Filter와 Gaussian Filter를 동시에 적용하는 함수
    :param image: 원본 이미지 (BGR 형식)
    :param median_kernel_size: Median Filter의 커널 크기 (홀수 값)
    :param gaussian_kernel_size: Gaussian Filter의 커널 크기 (홀수 값)
    :param gaussian_sigma: Gaussian Filter의 시그마 값
    :return: 필터가 적용된 이미지
    """
    # Median Filter 적용
    median_filtered = cv2.medianBlur(image, median_kernel_size)
    
    # Gaussian Filter 적용
    gaussian_filtered = cv2.GaussianBlur(median_filtered, (gaussian_kernel_size, gaussian_kernel_size), gaussian_sigma)
    
    return gaussian_filtered

def apply_speckle_noise(image, scale=0.2):
    """
    이미지에 speckle noise 추가
    :param image: 원본 이미지 (BGR 형식)
    :param scale: 노이즈 크기 (0 ~ 1)
    :return: 노이즈가 추가된 이미지
    """
    # Convert to float to apply speckle noise
    image_float = image.astype(np.float32) / 255.0

    # Generate speckle noise
    noise = np.random.normal(loc=0.0, scale=scale, size=image_float.shape)
    noisy_image = image_float + image_float * noise

    # Clip values to valid range [0, 1]
    noisy_image = np.clip(noisy_image, 0, 1)

    # Convert back to uint8
    noisy_image_uint8 = (noisy_image * 255).astype(np.uint8)

    return noisy_image_uint8

def split_and_overlay(image):
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 이미지 객체를 PIL Image로 변환
    image = Image.fromarray(image)
    
    # 이미지 크기 확인
    width, height = image.size
    
    # 각 블록의 크기 계산
    block_width = width // 3
    block_height = height // 3
    
    # 9개의 블록으로 분할
    blocks = []
    for i in range(3):
        for j in range(3):
            left = j * block_width
            upper = i * block_height
            right = left + block_width
            lower = upper + block_height
            block = image.crop((left, upper, right, lower))
            blocks.append(block)
    
    overlays = []
    offsets = [(0, 0), (0, 1), (1, 0), (1, 1)] 
    
    for dx, dy in offsets:
        overlay = Image.new('RGB', (block_width * 2, block_height * 2))
        for i in range(2):
            for j in range(2):
                block_index = (i + dx) * 3 + (j + dy)
                block = blocks[block_index]
                overlay.paste(block, (j * block_width, i * block_height))
        overlays.append(overlay)
    
    return overlays

def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:    
    # Convert to grayscale using cv2
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return grayscale_image

def random_erasing(image: np.ndarray) -> np.ndarray:
    """
    Apply random erasing to the input image without considering the center.

    Args:
        image (np.ndarray): Input image of shape (256, 256, 3).
    
    Returns:
        np.ndarray: Image with random erasing applied.
    """
    # 이미지 크기와 중심 좌표
    height, width = image.shape[:2]

    # 랜덤하게 지울 영역 크기 설정 (가로와 세로 각각 독립적으로 설정)
    erase_width = random.randint(width // 16, width // 4)  # 최소 1/16 ~ 최대 1/4 너비
    erase_height = random.randint(height // 16, height // 4)  # 최소 1/16 ~ 최대 1/4 높이

    # 랜덤 좌표 생성 (이미지 영역 내에서)
    top_left_x = random.randint(0, width - erase_width)
    top_left_y = random.randint(0, height - erase_height)

    # 랜덤하게 선택된 영역을 0(검정색)으로 설정
    image[top_left_y:top_left_y + erase_height, top_left_x:top_left_x + erase_width, :] = 0

    return image


def makeImage(fold):
    # 클래스별 폴더를 순회
    for class_folder in os.listdir(fold):
        class_folder_path = os.path.join(fold, class_folder)
        new_class_path = os.path.join(new_folder,class_folder)
        print(f"current : {class_folder_path}")
        print(f"new : {new_class_path}")

        if os.path.isdir(class_folder_path):  # 폴더가 클래스 폴더일 때만 진행
            for filename in os.listdir(class_folder_path):
                file_path = os.path.join(class_folder_path, filename)
                
                # 이미지를 BGR 형식으로 로드
                image = cv2.imread(file_path)

                # 이미지 overlay 저장
                overlays = split_and_overlay(image.copy())
                for idx, overlay in enumerate(overlays):
                    current_path = os.path.join(new_class_path, f"overlay{idx}_{filename}")
                    overlay.save(current_path, 'JPEG')
                
                # 1. Salt and Pepper 노이즈 추가
                noisy_image = add_salt_and_pepper_noise(image.copy())
                current_path = os.path.join(new_class_path, f"s&P_{filename}")
                cv2.imwrite(current_path, noisy_image)

                # 2. Center Brightness Weight 적용
                weighted_image = apply_center_brightness_weight(image.copy())
                current_path = os.path.join(new_class_path, f"Centerweighted_{filename}")
                cv2.imwrite(current_path, weighted_image)

                # 3. Median Filter와 Gaussian Filter 적용
                filtered_image = apply_median_and_gaussian_filters(image.copy())
                current_path = os.path.join(new_class_path, f"filtered_{filename}")
                cv2.imwrite(current_path, filtered_image)

                #4. Speckle Noise 추가
                speckle_noise = apply_speckle_noise(image.copy())
                current_path = os.path.join(new_class_path, f"speckle_{filename}")
                cv2.imwrite(current_path, speckle_noise)

                # 5. RGB to Grayscale
                grayscale_image = rgb_to_grayscale(image.copy())
                current_path = os.path.join(new_class_path, f"grayscale_{filename}")
                cv2.imwrite(current_path, grayscale_image)
                
                # 6. Random Erasing
                erased_image = random_erasing(image.copy())
                current_path = os.path.join(new_class_path, f"erased_{filename}")
                cv2.imwrite(current_path, erased_image)
                
            print("모든 이미지에 대해 전처리 완료.")

#makeImage(ori_folder)

# 클래스별 폴더를 순회하여 이미지 파일 개수 세기
for class_folder in os.listdir(new_folder):
    class_folder_path = os.path.join(new_folder, class_folder)
    
    if os.path.isdir(class_folder_path):  # 폴더가 클래스 폴더일 때만 진행
        image_count = 0
        for filename in os.listdir(class_folder_path):
            file_path = os.path.join(class_folder_path, filename)
            
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # 이미지 파일만 처리
                image_count += 1
        
        print(f"클래스 '{class_folder}' 폴더의 이미지 개수: {image_count}")
