# AutoFocus 清晰度評估(機器視覺作業一)

以 Sobel(Tenengrad)、Laplacian(變異數)、Canny(邊緣密度)三種方法評估影像清晰度,
從同一場景不同對焦位置的影像 / 影片中找出最佳焦點。

## 檔案
- `autofocus.py`:三種清晰度方法、前處理(灰階 / 中心 ROI / 縮放)、影片讀取
- `run_experiment.py`:產生報告中所有表格與圖表(results/)
- `resolution_test.py`:分析解析度對最佳影格的影響
- `focusscope.html`:互動介面,用 Chrome / Edge 直接開啟,拖曳圖片或影片即可分析

## 執行
```bash
pip install -r requirements.txt
python run_experiment.py ../images ../video/AF.mp4 ../results
python resolution_test.py ../video/AF.mp4 ../results/resolution.json
```

## 主要結果
- 10 張影像:三種方法皆選 img_05.jpg
- AF.mp4(224 幀):全畫面 Sobel #92、Laplacian/Canny #102(被背景紋理誤導);中心 ROI 50% 後 Sobel #92、Laplacian #91
