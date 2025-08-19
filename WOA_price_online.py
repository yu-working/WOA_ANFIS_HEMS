import numpy as np
from typing import List, Tuple
import random
from anfis_price import ANFIS
from random import choice
import pandas as pd
import time
from pythermalcomfort.models import pmv_ppd
#%%
class WhaleOptimizationHEMS:
    def __init__(
        self,
        n_whales: int = 30,
        max_iter: int = 100,
        temp_bounds: Tuple[float, float] = (18.0, 28.0),
        humidity_bounds: Tuple[float, float] = (30.0, 70.0),
        b: float = 1.0,
        a_decrease_factor: float = 2,
        pmv_up: float = 0.5,
        pmv_down: float = -0.5,
        device: list = ['dehumidifier', 'ac', 'fan']
    ):
        """
        初始化鯨魚優化演算法用於家庭能源管理系統
        
        參數:
            n_whales: 搜索代理(鯨魚)的數量
            max_iter: 最大迭代次數
            temp_bounds: 溫度範圍(最小值,最大值)
            humidity_bounds: 濕度範圍(最小值,最大值)
            b: 螺旋路徑參數
            a_decrease_factor: 'a'參數的遞減因子
        """
        self.n_whales = n_whales
        self.max_iter = max_iter
        self.temp_bounds = temp_bounds
        self.humidity_bounds = humidity_bounds
        self.n_dims = 2  # 溫度和濕度兩個維度
        self.b = b
        self.a_decrease_factor = a_decrease_factor
        self.anfis = ANFIS.load()
        self.pmv_up = pmv_up
        self.pmv_down = pmv_down
        self.device = device

    def initialize_population(self) -> np.ndarray:
        """初始化鯨魚群的隨機位置"""
        population = np.zeros((self.n_whales, self.n_dims))
        
        # 初始化溫度
        population_temp = []
        for i in range(self.n_whales): 
            population_temp.append(random.randint(self.temp_bounds[0], self.temp_bounds[1]))
        population_temp = np.array(population_temp)
        # 初始化濕度
        population_humd = []
        for i in range(self.n_whales):
            population_humd.append(random.randint(self.humidity_bounds[0], self.humidity_bounds[1]))
        population_humd = np.array(population_humd)
        # 室外溫度數據 (24-35°C)
        out_temperature = []
        for i in range(self.n_whales):
            out_temperature.append(random.randint(24, 35))
        out_temperature = np.array(out_temperature)
        # 室外濕度數據 (40-80%)
        out_humidity = []
        for i in range(self.n_whales):
            out_humidity.append(random.randint(40, 80))
        out_humidity = np.array(out_humidity)
           
        out_conditions = np.column_stack([out_temperature, out_humidity])
        population = np.column_stack([population_temp, population_humd])
        return population, out_conditions
    
    def calculate_power(self, dehumidifier_humidity, indoor_humidity,
                        dehumidifier_on, fan_on, ac_temperature, indoor_temp,
                        ac_mode, ac_fan_speed):
        # 計算總功耗
        dehumidifier_power = 0.12  # 除濕機功率(kW)
        fan_power = 0.06           # 電扇功率(kW)
        ac_base_power = 1.35       # 冷氣基礎功率(kW)
        ac_mode_weight = {0:0.5, 1:1}        # 冷氣運轉模式係數 0:送風 1:冷氣 2:舒眠
        ac_fan_weight = {0:0.6, 1:1, 2:0.8} # 風速係數 0:低速 1:高速 2:自動

        # 計算除濕機功耗
        dehumidifier_consumption = dehumidifier_power * max(0, (indoor_humidity - dehumidifier_humidity)/100) if dehumidifier_on else 0

        # 計算電扇功耗
        fan_consumption = fan_power if fan_on else 0

        # 計算冷氣功耗
        if ac_mode_weight[ac_mode] == 1:
            ac_consumption = ac_base_power * ac_mode_weight[ac_mode] * (1 + 0.1 * max(0, indoor_temp-ac_temperature)) * ac_fan_weight[ac_fan_speed] 
        else: 
            ac_consumption = ac_base_power * ac_mode_weight[ac_mode] * ac_fan_weight[ac_fan_speed] 

        return [dehumidifier_consumption, fan_consumption, ac_consumption]
    
    def fitness_function(self, temp: float, humidity: float, step) -> float:
        """
        計算適應度值(基於舒適度和能源消耗)
        
        參數:
            temp: 溫度值
            humidity: 濕度值
            outdoor_temp: 室外溫度
            outdoor_hum: 室外濕度
            step: 時間步驟
            
        返回:
            適應度值(越低越好)
        """
        # 舒適度參數(理想值)
        ideal_temp_high = 28.0
        ideal_humidity_high = 60.0
        ideal_temp_low = 26.0
        ideal_humidity_low = 50.0
        
        # 計算濕度權重
        if humidity < ideal_humidity_low:
            w_humidity = 1-(1/abs(humidity - ideal_humidity_low))
        elif humidity > ideal_humidity_low:
            w_humidity = 1-(1/abs(humidity - ideal_humidity_low))
        elif (humidity > ideal_humidity_high) or (humidity < ideal_humidity_low):
            w_humidity = 1
        else:
            w_humidity = 0.5
         
        # 計算溫度權重
    
        if temp < ideal_temp_low:
            w_temp = 1-(1/abs(temp - ideal_temp_low))
        elif temp > ideal_temp_low:
            w_temp = 1-(1/abs(temp - ideal_temp_low))
        elif (temp > ideal_temp_high) or (temp < ideal_temp_low):
            w_temp = 1   
        else:
            w_temp = 0.5
      
        #temp_deviation = abs(temp-outdoor_temp)
       # humidity_deviation = abs(humidity-outdoor_hum)
      
        # 使用ANFIS預測設備狀態
        device_state = self.anfis.predict(X=np.array([temp, humidity]).reshape(1, -1))
        device_state = pd.DataFrame(device_state, columns=['dehumidifier', 'dehumidifier_hum', 'ac_temp', 'ac_fan', 'ac_mode', 'fan_state'])
        
        if 'dehumidifier' in self.device: # 未持有除濕機時不運算
              pass
        else:
            device_state.loc[device_state.index[0], 'dehumidifier'] = 0  # 除濕機開關
            device_state.loc[device_state.index[0], 'dehumidifier_hum'] = 0  # 除濕機設定濕度 40-70%
        
        if 'fan' in self.device: # 未持有電扇時不運算
            pass
        else: device_state.loc[device_state.index[0], 'fan_state'] = 0
        
        # 計算能源消耗
        energy_consumption = self.calculate_power(device_state['dehumidifier_hum'][0], humidity,
                                                  device_state['dehumidifier'][0],device_state['fan_state'][0], device_state['ac_temp'][0], temp, 
                                                  device_state['ac_mode'][0], device_state['ac_fan'][0])
        
        # 計算PMV值(預測平均投票值)
        pmv = pmv_ppd(tdb=temp, tr=temp, vr=0.25, 
                      rh=humidity, met=1, clo=0.5, limit_inputs=False)
        
        # 根據時間段設定電價
        if step >= 24:
            step = step-24
        if step in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
            price = 1.96  # 離峰時段
        elif step in [9, 10, 11, 12, 13, 14, 15, 22, 23]:
            price = 4.54  # 半尖峰時段
        else:
            price = 6.92  # 尖峰時段

        # 計算最終適應度值
        fitness = sum(energy_consumption) * price
        if self.pmv_down<pmv['pmv']<self.pmv_up:
            fitness += 10000  # 當PMV值不在舒適範圍內時增加懲罰值
        
        return fitness
    
    def apply_bounds(self, position: np.ndarray) -> np.ndarray:
        """確保位置在定義的範圍內"""
        position[0] = np.clip(
            position[0], 
            self.temp_bounds[0], 
            self.temp_bounds[1]
        )
        position[1] = np.clip(
            position[1], 
            self.humidity_bounds[0], 
            self.humidity_bounds[1]
        )
        return position
    
    def change(self, device_date, indoor_data, outdoor_data):
        """模擬室內環境變化"""
        temp_change = 0
        humidity_change = 0
        
        # 除濕機對濕度的影響
        if device_date.iloc[0, 0] == 1:
            target_humidity = device_date.iloc[0, 1]
            humidity_change = -7.1 * 0.9
        
        # 冷氣對溫度的影響
        temp_diff = indoor_data[0] - device_date.iloc[0, 2]
        cooling_rate = 0.7 if device_date.iloc[0, 4] == 1 else 0
        outdoor_influence = (outdoor_data[0] - indoor_data[0]) * 0.05
        
        temp_change = -cooling_rate * temp_diff + outdoor_influence
        
        # 電扇的影響
        if device_date.iloc[0, 5] == 1:
            temp_change += 0.05 * (outdoor_data[0] - indoor_data[0])
        
        # 更新室內狀態
        indoor_temp = np.clip(indoor_data[0] + temp_change, 20, 35)
        indoor_humidity = np.clip(indoor_data[1] + humidity_change, 40, 85)
        '''
        # 更新室外狀態
        outdoor_temp = np.clip(
            outdoor_data[0] + np.random.uniform(-1, 1), 
            20, 40
        )
        outdoor_humidity = np.clip(
            outdoor_data[1] + np.random.uniform(-2, 2), 
            40, 90
        )
        '''
        new_indoor_data = np.array([indoor_temp, indoor_humidity]).reshape(1, 2)
       # new_outdoor_data = np.array([outdoor_temp, outdoor_humidity]).reshape(1, 2)
        return new_indoor_data#, new_outdoor_data
    
    
    def optimize(self, indoor_data, steps) -> Tuple[np.ndarray, float, List[float]]:
        """
        執行鯨魚優化演算法
        
        返回:
            best_position: 最佳溫度和濕度值
            best_fitness: 找到的最佳適應度值
            fitness_history: 每次迭代的最佳適應度值列表
        """
        # 根據時間設定電價
        if steps >= 24:
            steps = steps-24
        if steps in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
            price = 1.96
        elif steps in [9, 10, 11, 12, 13, 14, 15, 22, 23]:
            price = 4.54
        else:
            price = 6.92
        
        # 初始化群體
        population = indoor_data.copy()
        
        # 初始化最佳解
        fitness_values = np.array([
            self.fitness_function(pos[0], pos[1], steps) 
            for pos in population
        ])
        best_whale_idx = np.argmin(fitness_values)
        best_position = population[best_whale_idx].copy()
        best_fitness = fitness_values[best_whale_idx]
        
        # 優化歷史記錄
        fitness_history = [best_fitness]
        
        # 主要迭代循環
        for iteration in range(self.max_iter):
            # 更新a參數
            a = 2 - iteration * (2 / self.max_iter)
            
            # 更新每個鯨魚的位置
            for i in range(self.n_whales):
                # 隨機參數
                r = random.random()
                A = 2 * a * r - a
                C = 2 * r
                l = random.uniform(-1, 1)
                p = random.random()
                
                if p < 0.5:
                    # 包圍獵物或搜索獵物
                    if abs(A) < 1:
                        # 包圍獵物
                        D = abs(C * best_position - population[i])
                        new_position = best_position - A * D
                    else:
                        # 搜索獵物
                        random_whale = population[random.randint(0, self.n_whales-1)]
                        D = abs(C * random_whale - population[i])
                        new_position = random_whale - A * D
                else:
                    # 螺旋更新位置
                    D = abs(best_position - population[i])
                    spiral = (
                        D * np.exp(self.b * l) * 
                        np.cos(2 * np.pi * l) + 
                        best_position
                    )
                    new_position = spiral
                
                # 應用邊界並更新位置
                population[i] = self.apply_bounds(new_position)
                
                # 必要時更新最佳解
                current_fitness = self.fitness_function(
                    population[i][0], 
                    population[i][1], i
                )
                if current_fitness < best_fitness:
                    best_position = population[i].copy()
                    best_fitness = current_fitness
            
            fitness_history.append(best_fitness)
            
        # 根據最佳位置計算設備狀態和PMV值
        if indoor_data[-1][0] > best_position[0]:
            device_state = ANFIS.predict(self.anfis, indoor_data[-1].reshape(1, -1))
            pmv = pmv_ppd(tdb=indoor_data[-1][0], tr=indoor_data[-1][0], vr=0.25, 
                          rh=indoor_data[-1][1], met=1, clo=0.5, limit_inputs=False)
        else:
            device_state = ANFIS.predict(self.anfis, best_position.reshape(1, -1))
            pmv = pmv_ppd(tdb=best_position[0], tr=best_position[0], vr=0.25, 
                          rh=best_position[1], met=1, clo=0.5, limit_inputs=False)
                          
        device_state = pd.DataFrame(device_state, columns=['dehumidifier', 'dehumidifier_hum', 'ac_temp', 'ac_fan', 'ac_mode', 'fan_state'])
        
        if 'dehumidifier' in self.device: # 未持有除濕機時不運算
              pass
        else:
            device_state.loc[device_state.index[0], 'dehumidifier'] = 0  # 除濕機開關
            device_state.loc[device_state.index[0], 'dehumidifier_hum'] = 0  # 除濕機設定濕度 40-70%
        
        if 'fan' in self.device: # 未持有電扇時不運算
            pass
        else: device_state.loc[device_state.index[0], 'fan_state'] = 0
        
        energy_consumption = self.calculate_power(device_state['dehumidifier_hum'][0], indoor_data[-1][1],
                                                   device_state['dehumidifier'][0],device_state['fan_state'][0], device_state['ac_temp'][0], indoor_data[-1][0], 
                                                   device_state['ac_mode'][0], device_state['ac_fan'][0]) 
        return best_position, best_fitness, fitness_history, device_state, fitness_history, sum(energy_consumption)/1000*price, pmv['pmv']
#%%
if __name__ == '__main__':
    start = time.time()
    # 初始化優化器
    woa = WhaleOptimizationHEMS(
    n_whales=24,
    max_iter=100,
    temp_bounds=(26.0, 35.0),
    humidity_bounds=(40.0, 80.0)
    )
#%%
    Result = pd.DataFrame()
    fit = []
    Fitness_history = []
    history_indoor, history_outdoor, Cost, Pmv = [], [], [], []
    # 執行優化
    for i in range(14, 21):
        if i == 14:
            indoor_data, outdoor_data = woa.initialize_population()
            indoor_data[-1][0], indoor_data[-1][1] = 30, 70
            outdoor_data[-1][0], outdoor_data[-1][1] = 35, 80
        else:
            indoor_data = np.delete(indoor_data, 0, 0)
        history_indoor.append(indoor_data)
        best_position, best_fitness, fitness_history, device_state, fitness_history, cost, pmv = woa.optimize(indoor_data, i)
        Pmv.append(pmv)
        Cost.append(cost)
        env = pd.DataFrame(indoor_data[-1]).T
        env.columns = ['env_temp', 'env_humd']
        fit.append(best_fitness)
        Fitness_history.append(fitness_history)
        best_position = pd.DataFrame(best_position).T
        best_position.columns = ['best_temp', 'best_humd']
        result = pd.concat([env, best_position, device_state], axis=1)
        Result = Result.append(result, ignore_index=True)
        new_indoor_data = woa.change(device_state, indoor_data[-1], outdoor_data[-1])
        indoor_data = np.append(indoor_data, new_indoor_data, 0)
    Pmv = pd.DataFrame(Pmv, columns=['pmv'])
    Result = pd.concat([Result, Pmv], axis=1)    
    Result['ac_temp'] = np.where(Result['ac_mode'] == 0, '-', Result['ac_temp'])
    Result['dehumidifier_hum']  = Result['dehumidifier_hum'].astype(int)
    Result['dehumidifier_hum'] = round(Result['dehumidifier_hum'] / 5) * 5
    Result['dehumidifier_hum'] = np.where(Result['dehumidifier'] == 0, '-', Result['dehumidifier_hum'])
    Result.to_csv('./result/WOA_ANFIS用電分配最佳化參數測試結果.csv')
    end = time.time()
    print(end-start)
#%%
    # 讀取並分析實際用電數據
    data = pd.read_csv('./測試數據/nilm_data_ritaluetb_hour.csv')
    data = data.iloc[14:21, :]
    
    # 根據時段計算電費
    price = [[], [], []]
    for i in range(len(data)):
        if int(data.iloc[i, 0].split(' ')[1].split(':')[0]) in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
            price[0].append(data.iloc[i, 6])
        elif int(data.iloc[i, 0].split(' ')[1].split(':')[0]) in [9, 10, 11, 12, 13, 14, 15, 22, 23]:
            price[1].append(data.iloc[i, 6])
        elif int(data.iloc[i, 0].split(' ')[1].split(':')[0]) in [16, 17, 18, 19, 20, 21]:
            price[2].append(data.iloc[i, 6])
            
    # 計算總電費
    kwh = 0
    for i in range(3):
        if i == 0:
            kwh += (sum(price[i])/60000) * 1.96
        elif i == 1:
            kwh += (sum(price[i])/60000) * 4.54
        else:
            kwh += (sum(price[i])/60000) * 6.92
    print(sum(Cost), kwh)
    print((1-(sum(Cost)/kwh))*100)