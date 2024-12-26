import cv2
import numpy as np

RESOLUTION = 1000

YELLOW_BGR_LIST = [132, 227, 255]  # BGR格式的皮膚色
RED_BGR_LIST = [0, 0, 255]         # BGR格式的紅色

def classify_red_and_skin(image):
    '''
    轉灰階並根據平均值分類紅色與皮膚色，保留透明區域
    Input: image (np.ndarray) (with alpha channel)
    Output: np.ndarray (with alpha channel)
    '''
    # 分離 BGRA 通道
    bgr = image[:, :, :3]  # 提取 BGR 通道
    alpha = image[:, :, 3]  # 提取 Alpha 通道
    
    # 轉換為灰階（僅處理 BGR 部分）
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    
    # 計算灰階平均值
    mean_value = np.mean(gray[alpha > 0])  # 只計算非透明部分的灰階值
    print(f"灰階平均值: {mean_value}")
    
    # 建立結果影像，初始為原圖
    result = np.zeros_like(image)
    
    # 保留透明區域（Alpha 為 0 的部分保持透明）
    result[:, :, 3] = alpha  # 保留原圖的 Alpha 通道
    
    # 高於平均值的設為紅色
    result[gray <= mean_value, :3] = RED_BGR_LIST  # 非透明部分設為紅色
    # 低於平均值的設為皮膚色
    result[gray > mean_value, :3] = YELLOW_BGR_LIST  # 非透明部分設為皮膚色
    
    return result

def mosaic_image(image):
    '''
    馬賽克化圖片，並基於灰階值分類紅色與皮膚色，保留透明區域
    Input: image (np.ndarray) (with alpha channel)
    Output: np.ndarray (with alpha channel)
    '''
    size = image.shape
    level = max(1, min(size[0], size[1]) / RESOLUTION)  # 計算縮放比例
    h = int(size[0] / level)  # 計算馬賽克高度
    w = int(size[1] / level)  # 計算馬賽克寬度
    
    # 馬賽克縮小並放大
    mosaic = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(mosaic, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    
    # 對馬賽克圖片進行分類
    result = classify_red_and_skin(mosaic)
    
    return result
