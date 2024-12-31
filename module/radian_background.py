import numpy as np
from PIL import Image
import math
# 使用gamma transform，會讓圖片的紅白色更加分明，更符合視覺感受
def gamma_transform(x, gamma=2.6):
    return x ** (1 / gamma)
# 產生背景圖片
def create_image(width=800, height=800, red_white_gamma_index=2.6, center_white_gamma_index=0.9):
    # 將圓分成720份，每分角度的強度隨機分配，並進行gamma transform
    angle_intensity = (gamma_transform(np.random.rand(720), red_white_gamma_index)*255).astype(np.uint8)
    # 初始化結果圖片
    output = np.zeros((height, width, 3), dtype=np.uint8)
    # 計算圖片中心座標
    center_x = width // 2
    center_y = height // 2
    # 計算圖片中心到四角的最大距離
    max_distance = ((width//2)**2 + (height//2)**2)**0.5
    # 初始化模糊權重（越中心越白色）
    blur_weights = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            dx = x - center_x
            dy = y - center_y
            # 計算中心到(x, y)的距離
            distance = (dx**2 + dy**2)**(0.5)
            # 計算模糊權重（經過gamma transform後，視覺感受更好）
            blur_weights[y, x] = min(1.0, gamma_transform(distance / max_distance, center_white_gamma_index))
            # 計算弧度
            angle_rad = math.atan2(dy, dx)
            # 將弧度轉換為角度
            angle_deg = int((angle_rad * 180 / math.pi + 360) % 360 * 2)
            # 根據角度得出強度
            intensity = angle_intensity[angle_deg]
            # 根據強度設定顏色
            red, green, blue = 255, 255-intensity, 255-intensity
            # 設定顏色
            output[y, x] = [red, green, blue]
    
    # 白色
    white = np.ones((3,), dtype=np.uint8) * 255
    for y in range(height):
        for x in range(width):
            # 進行模糊（根據模糊權重與白色做混合）
            blend_factor = blur_weights[y, x]
            output[y, x] = (white * (1 - blend_factor) + 
                          output[y, x] * blend_factor).astype(np.uint8)
    # 返回圖片    
    return Image.fromarray(output)
# 好像沒用到，但是可以用來保存圖片
def save_image(image_path="radian_background.png", width=800, height=800, red_white_gamma_index=2.6, center_white_gamma_index=0.9):
    image = create_image(width, height, red_white_gamma_index, center_white_gamma_index)
    image.save(image_path)