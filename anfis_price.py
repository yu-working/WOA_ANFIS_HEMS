import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
#from WOA_sample_code import WhaleOptimizationHEMS
import joblib
from pythermalcomfort.models import pmv_ppd

class EnvironmentDataset(Dataset):
    """溫濕度數據集"""
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

class FuzzyLayer(nn.Module):
    """使用skfuzzy實現的模糊層"""
    def __init__(self, input_dim, n_membership):
        super().__init__()
        self.input_dim = input_dim  # 輸入維度
        self.n_membership = n_membership  # 隸屬函數數量
        
        # 為每個輸入特徵創建可學習的中心點和寬度參數
        self.centers = nn.Parameter(torch.linspace(0, 1, n_membership).repeat(input_dim, 1))
        self.sigmas = nn.Parameter(torch.ones(input_dim, n_membership) * 0.15)
        
    def forward(self, x):
        # x shape: (batch_size, input_dim)
        batch_size = x.size(0)
        memberships = torch.zeros(batch_size, self.input_dim, self.n_membership).to(x.device)
        
        # 對每個輸入維度計算隸屬度值
        for i in range(self.input_dim):
            for j in range(self.n_membership):
                # 使用skfuzzy的高斯隸屬度函數公式
                center = self.centers[i, j]
                sigma = torch.abs(self.sigmas[i, j]) + 1e-5  # 避免除以零
                x_i = x[:, i]
                memberships[:, i, j] = torch.exp(-((x_i - center) ** 2) / (2 * sigma ** 2))
        
        return memberships

class ANFISNet(nn.Module):
    """ANFIS網絡"""
    def __init__(self, input_dim=2, n_membership=3):
        super().__init__()
        self.input_dim = input_dim  # 輸入維度
        self.n_membership = n_membership  # 隸屬函數數量
        self.rules_dim = n_membership ** input_dim  # 規則數量
        
        # 模糊化層
        self.fuzzification = FuzzyLayer(input_dim, n_membership)
        
        # 規則層權重 - 每個輸出都有自己的權重
        self.rule_weights_fan = nn.Parameter(torch.randn(self.rules_dim) * 0.1)  # 電扇
        self.rule_weights_temp = nn.Parameter(torch.randn(self.rules_dim) * 0.1)  # 溫度
        self.rule_weights_ac_fan = nn.Parameter(torch.randn(self.rules_dim) * 0.1)  # 冷氣風速
        self.rule_weights_ac_mode = nn.Parameter(torch.randn(self.rules_dim) * 0.1)  # 冷氣模式
        self.rule_weights_dehumidifier = nn.Parameter(torch.randn(self.rules_dim) * 0.1)  # 除濕機
        self.rule_weights_dehumidifier_hum = nn.Parameter(torch.randn(self.rules_dim) * 0.1)  # 除濕機濕度設定
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # 第一層：模糊化
        membership_values = self.fuzzification(x)  # (batch_size, input_dim, n_membership)
        
        # 第二層：規則層
        # 生成所有規則組合的強度
        rule_strengths = torch.ones(batch_size, self.rules_dim).to(x.device)
        for i in range(self.input_dim):
            # 獲取當前輸入特徵的所有隸屬度值
            feature_memberships = membership_values[:, i, :]  # (batch_size, n_membership)
            
            # 為每個規則選擇對應的隸屬度值
            rule_indices = torch.arange(self.rules_dim).to(x.device)
            membership_idx = (rule_indices // (self.n_membership ** i)) % self.n_membership
            
            # 更新規則強度
            selected_memberships = feature_memberships[:, membership_idx]
            rule_strengths *= selected_memberships
        
        # 第三層：正規化層
        normalized_strengths = rule_strengths / (rule_strengths.sum(dim=1, keepdim=True) + 1e-10)
        
        # 第四層和第五層：結論層和輸出層
        # 使用sigmoid函數將輸出限制在合理範圍內
        fan_output = torch.sigmoid(normalized_strengths @ self.rule_weights_fan)  # 電扇開關(0-1)
        temp_output = 20 + 10 * torch.sigmoid(normalized_strengths @ self.rule_weights_temp)  # 溫度設定(20-30度)
        ac_fan_output = torch.sigmoid(normalized_strengths @ self.rule_weights_ac_fan)  # 冷氣風速(0-1)
        ac_mode_output = torch.sigmoid(normalized_strengths @ self.rule_weights_ac_mode)  # 冷氣模式(0-1)
        dehumidifier_output = torch.sigmoid(normalized_strengths @ self.rule_weights_dehumidifier)  # 除濕機開關(0-1)
        dehumidifier_hum_output = 40 + 100 * torch.sigmoid(normalized_strengths @ self.rule_weights_dehumidifier_hum)  # 除濕機濕度設定(40-140%)
        
        return torch.stack([dehumidifier_output, dehumidifier_hum_output, temp_output, ac_fan_output,  
                            ac_mode_output , fan_output], dim=1)

class ANFIS:
    def __init__(self, n_membership=3, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.n_membership = n_membership  # 隸屬函數數量
        self.device = device  # 運算設備
        self.model = ANFISNet(input_dim=2, n_membership=n_membership).to(device)  # ANFIS模型
        self.scaler = MinMaxScaler()  # 數據標準化
        
        # 創建模糊集的宇集
        self.x_temp = np.linspace(20, 32, 100)  # 溫度範圍
        self.x_humidity = np.linspace(40, 80, 100)  # 濕度範圍
        
        # 定義溫度的模糊集
        self.temp_low = fuzz.gaussmf(self.x_temp, 22, 2)  # 低溫
        self.temp_medium = fuzz.gaussmf(self.x_temp, 26, 2)  # 中溫
        self.temp_high = fuzz.gaussmf(self.x_temp, 30, 2)  # 高溫
        
        # 定義濕度的模糊集
        self.humidity_low = fuzz.gaussmf(self.x_humidity, 50, 5)  # 低濕
        self.humidity_medium = fuzz.gaussmf(self.x_humidity, 60, 5)  # 中濕
        self.humidity_high = fuzz.gaussmf(self.x_humidity, 70, 5)  # 高濕
        
        self.isa = pd.read_csv('./data/紅外線遙控器冷氣調控指令集_woa.csv')
        
    def generate_target(self, temperature, humidity, price):
        """生成目標輸出 - 針對台灣夏季氣候優化"""
        n_samples = len(temperature)
        targets = np.zeros((n_samples, 6))
        isa = pd.DataFrame([self.isa['ac_temp'], self.isa['ac_fan'], self.isa['ac_mode']]).T
        # 計算PMV指標
        pmv = pmv_ppd(tdb=temperature, tr=temperature, vr=0.25, 
                      rh=humidity, met=1, clo=0.5, limit_inputs=False)
        
        # 電扇開關邏輯 - 台灣夏季幾乎整天都需要通風
        fan_condition = (
            ((temperature > 27) & (humidity <= 65)) |  # 高溫但相對較乾燥時用風扇通風
            ((temperature <= 27) & (temperature > 24) & (humidity <= 70))  # 中溫時也適合風扇
        )
        targets[:, 5] = fan_condition.astype(float)
        
        # 溫度設定邏輯 - 考慮台灣夏季悶熱環境
        temp_setting = np.full_like(temperature, 27.0)  # 默認設定溫度
        
        # 極端高溫高濕 (台灣夏季常見)
        mask_extreme = (temperature >= 31) | ((temperature >= 29) & (humidity > 75))
        temp_setting[mask_extreme] = 26.0  # 更低的溫度設定以應對極端情況
        
        # 中等高溫條件 (台灣夏季典型)
        mask_med_high = (temperature >= 27) & (temperature < 29) & (humidity > 60)
        temp_setting[mask_med_high] = 27.0
        
        # 中等條件 
        mask_med = (temperature <= 26) & (humidity <= 60)
        temp_setting[mask_med] = 28.0
        
        targets[:, 2] = temp_setting
        
        # 冷氣風扇狀態 - 台灣夏季通常需要較強的風速來快速降溫
        ac_fan_setting = np.full_like(temperature, 0.0)  # 默認低風速
        
        # 高溫高濕時使用高風速
        ac_fan_mask_high = (temperature >= 29) | ((temperature >= 27) & (humidity >= 70))
        ac_fan_setting[ac_fan_mask_high] = 1.0
        
        targets[:, 3] = ac_fan_setting
        
        # 冷氣模式狀態 - 考慮到節能與除濕需求
        ac_mode_setting = np.full_like(temperature, 1.0)  # 默認製冷模式
        
        # 高溫高濕時使用製冷模式
        ac_mode_mask_cooling = (temperature >= 28) | (humidity >= 70)
        ac_mode_setting[ac_mode_mask_cooling] = 1.0
        
        # 溫度不高但濕度高時，可以使用送風模式
        ac_mode_mask_fan = (pmv['pmv'] < 0.5) & (pmv['pmv'] > -0.5) | (price == 6.92) & (pmv['pmv'] < 1) & (pmv['pmv'] > -1)\
            | (price == 4.54) & (pmv['pmv'] < 0.7) & (pmv['pmv'] > -0.7)
        ac_mode_setting[ac_mode_mask_fan] = 0.0
        
        targets[:, 4] = ac_mode_setting
        '''
        targets[:, 2:5] = isa.loc[2]
        mask_extreme = (temperature >= 33) | ((temperature >= 30) & (humidity > 75))
        targets[mask_extreme, 2:5] = isa.loc[0]  # 只在條件為True時更新
        mask_extreme = (temperature >= 30) | ((temperature >= 28) & (humidity > 70))
        targets[mask_extreme, 2:5] = isa.loc[9]  # 只在條件為True時更新
        mask_extreme = (temperature >= 30) & (pmv['pmv'] >= 1.0)
        targets[mask_extreme, 2:5] = isa.loc[1]  # 只在條件為True時更新
        mask_extreme = (temperature >= 28) & (pmv['pmv'] >= 1.0)
        targets[mask_extreme, 2:5] = isa.loc[3]  # 只在條件為True時更新
        mask_extreme = (temperature < 26) & (pmv['pmv'] >= 1.0)
        targets[mask_extreme, 2:5] = isa.loc[4]  # 只在條件為True時更新
        mask_extreme = (temperature >= 29) | ((temperature >= 27) & (humidity > 70))
        targets[mask_extreme, 2:5] = isa.loc[10]  # 只在條件為True時更新
        mask_extreme = (temperature >= 28) | ((temperature >= 26) & (humidity > 70))
        targets[mask_extreme, 2:5] = isa.loc[11]  # 只在條件為True時更新
        mask_extreme = (temperature < 26) | ((temperature < 26) & (humidity < 60))
        targets[mask_extreme, 2:5] = isa.loc[12]  # 只在條件為True時更新
        mask_extreme = (pmv['pmv'] < 1.0) & (pmv['pmv'] > -1.0) & (pmv['pmv'] > 0.5) & (pmv['pmv'] < -0.5)
        targets[mask_extreme, 2:5] = isa.loc[5]  # 只在條件為True時更新
        mask_extreme = (pmv['pmv'] < 0.85) & (pmv['pmv'] > -0.85) & (pmv['pmv'] > 0.5) & (pmv['pmv'] < -0.5)
        targets[mask_extreme, 2:5] = isa.loc[6]  # 只在條件為True時更新
        mask_extreme = (pmv['pmv'] < 0.7) & (pmv['pmv'] > -0.7) & (pmv['pmv'] > 0.5) & (pmv['pmv'] < -0.5)
        targets[mask_extreme, 2:5] = isa.loc[7]  # 只在條件為True時更新
        mask_extreme = (pmv['pmv'] < 0.6) & (pmv['pmv'] > -0.6) & (pmv['pmv'] > 0.5) & (pmv['pmv'] < -0.5)
        targets[mask_extreme, 2:5] = isa.loc[8]  # 只在條件為True時更新
        mask_extreme = (pmv['pmv'] < 0.5) & (pmv['pmv'] > -0.5)
        targets[mask_extreme, 2:5] = isa.loc[13]  # 只在條件為True時更新
        mask_extreme = (pmv['pmv'] < 0.4) & (pmv['pmv'] > -0.4)
        targets[mask_extreme, 2:5] = isa.loc[14]  # 只在條件為True時更新
        mask_extreme = (pmv['pmv'] < 0.6) & (pmv['pmv'] > -0.6)
        targets[mask_extreme, 2:5] = isa.loc[15]  # 只在條件為True時更新
        '''
       
        # 除濕機狀態 - 台灣梅雨季與颱風季特別需要除濕
        dehumidifier_condition = (
            (humidity >= 75) |  # 非常潮濕
            ((humidity >= 70) & (temperature <= 27)) |  # 中等溫度但潮濕
            ((temperature >= 30) & (humidity >= 65))   # 極端高溫且潮濕
        )
        targets[:, 0] = dehumidifier_condition.astype(float)
        
        # 除濕機濕度設定 - 考慮到台灣夏季的高濕環境
        dehumidifier_hum_setting = np.full_like(humidity, 55)  # 默認設定
        
        # 根據濕度程度調整目標濕度
        dehumidifier_hum_mask_extreme = (humidity > 80)
        dehumidifier_hum_setting[dehumidifier_hum_mask_extreme] = 50.0  # 極高濕度時設定較低目標
        
        dehumidifier_hum_mask_high = (humidity > 70) & (humidity <= 80)
        dehumidifier_hum_setting[dehumidifier_hum_mask_high] = 55.0
        
        dehumidifier_hum_mask_med = (humidity > 65) & (humidity <= 70)
        dehumidifier_hum_setting[dehumidifier_hum_mask_med] = 60.0
        
        dehumidifier_hum_mask_low = (humidity <= 65)
        dehumidifier_hum_setting[dehumidifier_hum_mask_low] = 65.0  # 較低濕度時可設定較寬鬆目標
        
        targets[:, 1] = dehumidifier_hum_setting
        
        return targets
    
    def fit(self, X, batch_size=32, epochs=100, learning_rate=0.01):
        """訓練模型"""
        # 生成目標輸出並準備訓練數據
        y = self.generate_target(X[:, 0], X[:, 1], X[:, 2])
        X = np.delete(X, 2, axis=1)  # 移除價格列
        X_scaled = self.scaler.fit_transform(X)  # 標準化輸入數據
        
        # 創建數據加載器
        dataset = EnvironmentDataset(X_scaled, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 定義損失函數和優化器
        criterion_binary = nn.CrossEntropyLoss()  # 二元分類損失
        criterion_regression = nn.MSELoss()  # 回歸損失
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 訓練迭代
        losses = []
        best_loss = float('inf')
        patience = 10  # 早停參數
        no_improve = 0
        
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # 前向傳播
                outputs = self.model(batch_X)
                
                # 計算各個輸出的損失
                fan_loss = criterion_binary(outputs[:, 5], batch_y[:, 5])  # 電扇
                temp_loss = criterion_regression(outputs[:, 2], batch_y[:, 2])  # 溫度
                ac_fan_loss = criterion_binary(outputs[:, 3], batch_y[:, 3])  # 冷氣風速
                ac_mode_loss = criterion_binary(outputs[:, 4], batch_y[:, 4])  # 冷氣模式
                dehumidifier_loss = criterion_binary(outputs[:, 0], batch_y[:, 0])  # 除濕機
                dehumidifier_hum_loss = criterion_regression(outputs[:, 1], batch_y[:, 1])  # 除濕機濕度
                
                # 總損失
                loss = fan_loss + temp_loss + ac_fan_loss + ac_mode_loss + dehumidifier_loss + dehumidifier_hum_loss
                
                # 反向傳播和優化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            epoch_loss /= len(dataloader)
            losses.append(epoch_loss)
            
            # 早停檢查
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                no_improve = 0
            else:
                no_improve += 1
                
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {epoch_loss:.4f}')
        
        return losses
    
    def predict(self, X):
       """預測輸出"""
       self.model.eval()
       X_scaled = self.scaler.transform(X)  # 特徵標準化
       X_tensor = torch.FloatTensor(X_scaled).to(self.device)
       
       with torch.no_grad():
           outputs = self.model(X_tensor)
           predictions = outputs.cpu().numpy()
           p = predictions.copy()
       
       # 將開關預測轉換為二元值
       predictions[:, 0] = (predictions[:, 0] > 0.5).astype(int)  # 除濕機開關
       predictions[:, 1] = (predictions[:, 1]).astype(int)  # 除濕機濕度
       predictions[:, 2] = (predictions[:, 2]).astype(int)  # 溫度設定
       predictions[:, 3] = np.where(predictions[:, 3] >= 0.8, 2, predictions[:, 3])  # 冷氣風速
       predictions[:, 3] = np.where(predictions[:, 3] == 1, 1, predictions[:, 3])  # 冷氣風速
       predictions[:, 3] = np.where(predictions[:, 3] < 0.8, 0, predictions[:, 3])  # 冷氣風速
       predictions[:, 4] = (predictions[:, 4] > 0.5).astype(int)  # 冷氣模式
       predictions[:, -1] = (predictions[:, -1] > 0.5).astype(int)  # 風扇開關
       
       return predictions

    def save(self):
        """儲存模型"""
        # 儲存模型參數
        torch.save(self.model.state_dict(), './config/anfis_model_price.pt')
        
        # 儲存縮放器
        joblib.dump(self.scaler, './config/scaler_price.pkl')
        
        # 儲存配置參數
        config = {
            'n_membership': self.n_membership,
            'device': self.device
        }
        joblib.dump(config, './config/config_price.pkl')
        print('儲存成功')
    
    @classmethod
    def load(cls):
        """載入模型"""
        # 載入配置
        config = joblib.load('./config/config_price.pkl')
        anfis = cls(n_membership=config['n_membership'], device=config['device'])
        
        # 載入模型參數
        anfis.model.load_state_dict(torch.load('./config/anfis_model_price.pt', map_location=config['device']))
        
        # 載入縮放器
        anfis.scaler = joblib.load('./config/scaler_price.pkl')
        print('讀取成功')

        return anfis
    
#%% anfis 訓練
if __name__ == '__main__':
    '''
    woa = WhaleOptimizationHEMS(
        n_whales=30,
        max_iter=100,
        temp_bounds=(24.0, 32.0),
        humidity_bounds=(40.0, 75.0)
    )
    '''
    #%% 生成測試數據
    np.random.seed(42)  # 設定隨機種子
    n_samples = 100000  # 樣本數量
    
    # 生成隨機環境數據
    temperature = np.random.uniform(24, 35, n_samples)  # 溫度範圍24-35度
    humidity = np.random.uniform(40, 80, n_samples)  # 濕度範圍40-80%
    price = np.random.choice([6.92, 4.54, 1.96], n_samples)  # 電價
    
    # 組合輸入數據
    X = np.column_stack([temperature, humidity, price])
    ramdon_data = pd.DataFrame(X, columns=['env_temp', 'env_hum', 'price'])
    #ramdon_data.to_csv('C:/Users/hankli/Documents/114計劃相關/隨機參數/ramdon_env_data.csv')
    
    # 設定運算設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 創建和訓練模型
    anfis = ANFIS(n_membership=3, device=device)
    losses = anfis.fit(X, epochs=100, batch_size=32)
    anfis.save()
    test = anfis.load()
    
    #%% 繪製損失曲線
    '''
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('ANFIS Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Combined Loss')
    plt.show()
    '''
    
    #%% 測試模型
    # 生成測試數據
    test_sample = 100
    temperature_test = np.random.uniform(24, 35, test_sample)  # 測試溫度
    humidity_test = np.random.uniform(40, 80, test_sample)  # 測試濕度
    test_conditions = np.column_stack([temperature_test, humidity_test])
    
    # 進行預測
    predictions = anfis.predict(test_conditions)
    
    # 整理結果
    test_conditions = pd.DataFrame(test_conditions, columns=['temperature_env', 'humidity_env'])
    predictions = pd.DataFrame(predictions, columns=['dehumidifier', 'dehumidifier_hum', 'ac_temp', 'ac_fan', 'ac_mode', 'fan_state'])
    fin = pd.concat([test_conditions, predictions], axis=1)