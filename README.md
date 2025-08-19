# WOA_ANFIS_HEMS
WOA × ANFIS 用電分配最佳化家庭能源管理系統

## 專案簡介

本專案實作 鯨魚優化演算法 (Whale Optimization Algorithm, WOA)，結合 自適應神經模糊推理系統 (Adaptive Neuro-Fuzzy Inference System, ANFIS) 與 PMV (Predicted Mean Vote) 指標，模擬家庭能源管理系統（HEMS）的溫濕度調控，達成能源消耗與室內舒適度的最佳平衡。

系統會依據室內環境條件（溫度、濕度）與電價時段，動態決策：

 - 冷氣模式、冷氣風扇模式與設定溫度

 - 除濕機開關與設定濕度

 - 電扇開關

並輸出最佳化結果、用電成本與舒適度指標。

## 核心功能

#### 1.WOA 優化溫濕度設定

 - 搜索最佳室內溫度與濕度，使能源成本最低且 PMV 舒適度在指定範圍內。

#### 2.ANFIS 預測設備狀態

 - 根據室內條件，輸出各設備的最佳運行設定。

#### 3.PMV 舒適度評估

 - 使用 pythermalcomfort 計算室內舒適度。

#### 4.電費計算

 - 根據台灣分時電價計算不同時段的耗電成本。

## 主要程式架構

```
WOA_ANFIS_HEMS
│
├── anfis_price.py        # ANFIS模型訓練
├── WOA_price_online.py   # 主程式
├─ config/
│   ├─ anfis_model_price.pt  # 訓練好的模型存檔
│   ├─ scaler_price.pkl      # 縮放器
│   └─ config_price.pkl      # 配置參數
├─ data/
│   ├─ 紅外線遙控器冷氣調控指令集_woa.csv  # 冷氣遙控指令集
│   └─ nilm_data_ritaluetb_hour.csv     # 歷史用電數據
└─ result/
    └─ PPO用電分配參數測試結果.csv
```

## 執行流程

#### 1.執行anfis_price.py

生成 訓練模型 `anfis_model_price.pt` 、縮放器 `scaler_price.pkl` 、配置參數 `config_price.pkl`

#### 2.執行 WOA_price_online.py 初始化 WOA

```
woa = WhaleOptimizationHEMS(
    n_whales=24,
    max_iter=100,
    temp_bounds=(26.0, 35.0),
    humidity_bounds=(40.0, 80.0)
)
```

#### 3.初始化室內環境

 - 第一次迭代隨機生成室內溫濕度

 - 後續迭代移除舊資料並新增模擬數據

#### 4.執行迴圈

 - 呼叫 `optimize` 找出最佳溫濕度與設備策略。

 - 呼叫 `change` 模擬室內環境變化，更新下一步的室內數據。

 - 呼叫 `calculate_power`計算耗能，透過PMV指標決定是否增加懲罰值

 - 根據該次迴圈最佳位置計算設備狀態並模擬耗能影響

 - 紀錄設備狀態、耗能、pmv

#### 5.比較與分析

 - 匯出結果表格，欄位包含:`env_temp`、`env_humd`、`best_temp`、`best_humd`、`dehumidifier`、`dehumidifier_hum`、`ac_temp`、`ac_fan`、`ac_mode`、`fan_state`、`pmv`

 - 與實際 NILM 測試數據比較節能率

## 安裝需求

請先安裝所需套件：
```
pandas == 1.2.4
numpy == 1.24.4
pythermalcomfort == 2.10.0
matplotlib == 3.7.4
scikit-fuzzy==0.5.0
scikit-learn==1.2.2
joblib == 1.3.2
Flask == 3.0.3
psycopg2 == 2.8.6
pymssql == 2.2.1
sshtunnel == 0.4.0
torch == 2.2.0
torchvision == 0.17.0
```

另外需提供：
 
 - 模型參數 `anfis_model_price.pt`
 - 縮放器 `scaler_price.pkl`
 - 配置參數 `config_price.pkl`
 - `anfis_price.py`(包含`ANFIS`模型與`load()`方法）
 - 測試數據檔案 `nilm_data_ritaluetb_hour.csv` (用於計算與實際用電差異)

## 範例輸出

程式執行後會輸出：

 - 運算所需時間
 - 與原始數據比較的節能率
 - 室內環境變化模擬與電器調整結果資料表
