document.getElementById('uploadForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const resultDiv = document.getElementById('result');
    const downloadBtn = document.getElementById('download-btn');

    // 顯示「處理中」訊息
    resultDiv.innerHTML = '<p>處理中，請稍候...</p>';
    downloadBtn.style.display = 'none';

    const formData = new FormData();
    const fileInput = document.getElementById('imageInput');
    formData.append('image', fileInput.files[0]);

    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/process', true);

    xhr.onload = () => {
        if (xhr.status === 200) {
            const response = JSON.parse(xhr.responseText);

            // 強制瀏覽器重新載入圖片，加上隨機參數
            const randomParam = `?t=${new Date().getTime()}`;
            const imgUrl = response.result_url + randomParam;

            const img = document.createElement('img');
            img.src = imgUrl;

            resultDiv.innerHTML = ''; // 清除「處理中」訊息
            resultDiv.appendChild(img);

            downloadBtn.href = imgUrl;
            downloadBtn.style.display = 'inline-block';
        } else {
            resultDiv.innerHTML = `<p style="color: red;">圖片處理失敗: ${xhr.responseText}</p>`;
        }
    };

    xhr.onerror = () => {
        resultDiv.innerHTML = `<p style="color: red;">上傳過程中發生錯誤。</p>`;
    };

    xhr.send(formData);
});
