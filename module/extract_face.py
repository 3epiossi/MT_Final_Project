import mediapipe as mp
import cv2
import numpy as np

def extract_face(image):
    '''
    Input: image (np.ndarray)
    Output: np.ndarray (with alpha channel)
    '''
    # 確保圖片通道數正確
    if len(image.shape) == 2:  # 如果是灰階圖片
        print("檢測到灰階圖片，將轉換為 3 通道 (RGB)")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 3:  # 如果是三通道圖片:
        print("檢測到 3 通道圖片，轉換為 RGB 格式")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image.shape[2] == 4:  # 如果是帶有 Alpha 通道的圖片
        print("檢測到帶有 Alpha 通道的圖片，將轉換為 3 通道 (RGB)")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        raise Exception("不支援的圖片通道數")

    # 確保圖片尺寸
    height, width, channel = image_rgb.shape[:3]

    # 初始化 MediaPipe
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    mp_face_mesh = mp.solutions.face_mesh

    # 創建分割器和面部檢測器
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5
    )

    try:
        # 步驟 1: 去除背景
        results_segmentation = selfie_segmentation.process(image_rgb)
        mask = results_segmentation.segmentation_mask > 0.1  # 生成二值化的分割掩碼

        # 創建 BGRA 圖片（4 通道）
        image_bgra = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGRA)

        # 將背景設為透明（Alpha 通道值為 0），人像設為不透明（Alpha 通道值為 255）
        image_bgra[:, :, 3] = (mask * 255).astype(np.uint8)

        # 保存處理後的圖片
        cv2.imwrite("person_only_with_alpha.png", image_bgra)

        # 步驟 2: 檢測面部特徵點
        results_mesh = face_mesh.process(image_rgb)

        # 檢查是否檢測到面部特徵
        if results_mesh.multi_face_landmarks:
            face_landmarks = results_mesh.multi_face_landmarks[0]

            # 計算人臉左右邊界
            x_coords = [int(landmark.x * width) for landmark in face_landmarks.landmark]
            left_x, right_x = max(0, min(x_coords)), min(width, max(x_coords))
            margin = int((right_x - left_x) * 0.2)  # 調整這個值以增加裁切的範圍
            left_x, right_x = max(0, left_x - margin), min(width, right_x + margin)

            # 獲取下巴、頭頂
            chin_y = max(landmark.y for landmark in face_landmarks.landmark) * height
            top_y = next((y for y in range(height) if not
                        all(image_bgra[y][x][3] == 0
                            for x in range(left_x, right_x))), 0)  # 找到第一個不透明的像素行
            face_height = chin_y - top_y
            bottom_y = int(chin_y + face_height * 0.2)  # 稍微包含一點脖子

            # 裁切圖片
            print(f"裁切區域: 左上=({top_y}, {left_x}), 右下=({bottom_y}, {right_x})")
            cropped_face = image_bgra[top_y:bottom_y, left_x:right_x]

            # 返回裁切後的結果
            cv2.imwrite("face_only_with_alpha_cropped.png", cropped_face)
            return cropped_face

        else:
            raise Exception("未檢測到面部")
        
    except Exception as e:
        raise e
    
    finally:
        # 釋放資源
        selfie_segmentation.close()
        face_mesh.close()
