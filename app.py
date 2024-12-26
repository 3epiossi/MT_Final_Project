# app.py
from flask import Flask, request, render_template, jsonify
from process_image import process_image
import os
import cv2

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
RESULT_FOLDER = 'static/results/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    file = request.files.get('image')

    if not file:
        return jsonify({"error": "未收到圖片檔案！"}), 400

    if not file.content_type.startswith('image/'):
        return jsonify({"error": "上傳的檔案不是圖片格式！"}), 400

    try:
        # 儲存上傳的檔案
        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(input_path)

        # 開啟圖片並轉為 NumPy 陣列
        image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

        # 圖像處理邏輯
        processed_image = process_image(image)

        # 將處理後的圖片儲存
        result_path = os.path.join(RESULT_FOLDER, 'result.png')
        cv2.imwrite(result_path, processed_image)

        return jsonify({"result_url": f"/{result_path}"})

    except Exception as e:
        return jsonify({"error": f"圖片處理時發生錯誤: {str(e)}"}), 400

if __name__ == '__main__':
    app.run(debug=True)
