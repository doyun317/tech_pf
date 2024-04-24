import os
import numpy as np
import math
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from finrl.finrl_meta.preprocessor.preprocessors import FeatureEngineer
from finrl.drl_agents.stablebaselines3.models import DRLAgent
from stable_baselines3 import PPO
from finrl.finrl_meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import random
from sklearn.model_selection import TimeSeriesSplit
import itertools
import seaborn as sns

def best_model_eval(folder_path, result_dir, MATHOD_NAME):

    test_env_config = {
        "price_array": test_data.close.values.reshape(-1, 1),
        "tech_array": test_data[TECHNICAL_INDICATORS_LIST].values,
        "turbulence_array": np.zeros(len(test_data)),
        "if_train": False
    }
    e_test_gym = StockTradingEnv(config=test_env_config, **env_kwargs)

    best_sharpe_ratio = -np.inf
    best_cum_return = -np.inf
    best_eval_values=[]
    # 각 방식의 최고 성능 모델로 백테스팅 진행
    cwd = result_dir
    for i in range(50):
        df_best_account_value = DRLAgent.DRL_prediction_load_from_file("ppo",environment=e_test_gym,cwd=cwd)
        df_best_account_value = pd.DataFrame(df_best_account_value,columns=["account_value"])
        
        perf_stats = DRLAgent.get_perf_stats(df_best_account_value['account_value'].values)
        if MATHOD_NAME == "CPCV":
            sharpe_ratio = perf_stats['Sharpe Ratio']
            cum_return = perf_stats["Cumulative Return"][-1]
            
            if cum_return > best_cum_return:
                best_eval_values = df_best_account_value
                best_cum_return = cum_return
        
        if MATHOD_NAME != "CPCV":
            sharpe_ratio = perf_stats['Sharpe Ratio']
            cum_return = perf_stats["Cumulative Return"][-1]
            
            if sharpe_ratio > best_sharpe_ratio:
                best_eval_values = df_best_account_value
                best_sharpe_ratio = sharpe_ratio
    
    df_best_account_value = best_eval_values
        
    # 최고 모델의 성능 메트릭 저장
    perf_stats = DRLAgent.get_perf_stats(df_best_account_value['account_value'].values)
    sharpe_ratio = perf_stats['Sharpe Ratio']
    cum_return = perf_stats['Cumulative Return'][-1]
    max_drawdown = perf_stats['Max Drawdown']

    # 성능 메트릭 저장
    best_model_metrics = pd.DataFrame({
        'Sharpe Ratio': [sharpe_ratio],
        'Cumulative Return': [cum_return],
        'Max Drawdown': [max_drawdown]
    })

    best_account_value_path = os.path.join(folder_path, f"best_account_value_{MATHOD_NAME}.csv")
    df_best_account_value.to_csv(best_account_value_path, index=False)
    
    
    best_model_metrics_path = os.path.join(folder_path, f"best_model_metrics_{MATHOD_NAME}.csv")
    best_model_metrics.to_csv(best_model_metrics_path, index=False)
    
    print(f"Sharpe Ratio: {sharpe_ratio}")
    print(f"Best cumlative Return: {cum_return}")

EVAL_TEST = False

TECHNICAL_INDICATORS_LIST = ['open',
                             'high',
                             'low',
                             'close',
                             'volume',
                             'macd',
                             'rsi',
                             'cci',
                             'dx'
                             ]

env_kwargs = {
    "initial_account": 1e6,
    "gamma": 0.99,
    "turbulence_thresh": 99,
    "min_stock_rate": 0.1,
    "max_stock": 1e3,
    "initial_capital": 1e6,
    "buy_cost_pct": 1e-3,
    "sell_cost_pct": 1e-3,
    "reward_scaling": 2 ** -11,
    "initial_stocks": None,
}

# val_plot_data 폴더 경로 설정
val_plot_data_path = "val_plot_data"

# 데이터 저장 경로
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
test_data['date'] = pd.to_datetime(test_data['date'])

# 모델 zip 파일과 결과 csv 파일을 저장할 리스트 초기화
model_zip_files = {}
account_value_csv_files = {}
model_metrics_csv_files = {}

# 모델 문자열 리스트
model_strings = ["WF", "KFOLD", "CPCV"]

# 각 모델 문자열에 대해 딕셔너리 초기화
for model_string in model_strings:
    model_zip_files[model_string] = []
    account_value_csv_files[model_string] = []
    model_metrics_csv_files[model_string] = []

# val_plot_data 폴더 내의 모든 폴더 탐색
for folder_name in os.listdir(val_plot_data_path):
    folder_path = os.path.join(val_plot_data_path, folder_name)
    
    # 폴더 내의 모든 파일 탐색
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        # 모델 문자열 확인
        for model_string in model_strings:
            if model_string in file_name:
                # 모델 zip 파일인 경우
                if file_name.endswith(".zip"):
                    model_zip_files[model_string].append(file_path)
                    if EVAL_TEST:
                        best_model_eval(folder_path,file_path,model_string)
                    
                # 결과 csv 파일인 경우
                elif file_name.endswith(".csv"):
                    if "account_value" in file_name:
                        account_value_csv_files[model_string].append(file_path)
                    elif "model_metrics" in file_name:
                        model_metrics_csv_files[model_string].append(file_path)
                
                break


wf_av = pd.read_csv(account_value_csv_files["WF"][0])
Kfold_av = pd.read_csv(account_value_csv_files["KFOLD"][0])
cpcv_av = pd.read_csv(account_value_csv_files["CPCV"][0])

wf_perf_stats =  DRLAgent.get_perf_stats(wf_av["account_value"].values)
Kfold_perf_stats = DRLAgent.get_perf_stats(Kfold_av["account_value"].values)
cpcv_perf_stats =  DRLAgent.get_perf_stats(cpcv_av["account_value"].values)

wf_cum_return = wf_perf_stats["Cumulative Return"]
Kfold_cum_return = Kfold_perf_stats["Cumulative Return"]
cpcv_cum_return = cpcv_perf_stats["Cumulative Return"]

# random walk 전략 누적수익률 계산
def random_walk_strategy(data):
    prices = data[['close']].values
    num_days, num_stocks = prices.shape
    simulated_values = np.zeros(num_days)
    initial_capital = 1e6

    portfolio = np.zeros(num_stocks)
    capital = initial_capital
    for day in range(num_days):
        for stock in range(num_stocks):
            action = random.randint(0, 2)  # 0: 매도, 1: 홀드, 2: 매수
            
            if action == 0:  # 매도
                num_shares_to_sell = random.randint(0, int(portfolio[stock]))
                capital += num_shares_to_sell * prices[day, stock]
                portfolio[stock] -= num_shares_to_sell
            elif action == 2:  # 매수
                available_capital = capital / (num_stocks - stock)
                max_shares_to_buy = int(available_capital / (prices[day, stock] * (1 + 1e-3)))
                if max_shares_to_buy > 0:
                    num_shares_to_buy = random.randint(1, max_shares_to_buy)
                    portfolio[stock] += num_shares_to_buy
                    capital -= num_shares_to_buy * prices[day, stock] * (1 + 1e-3)
        
        simulated_values[day] = capital + np.sum(portfolio * prices[day])

    return simulated_values

def buy_and_hold_strategy(data, initial_capital=1e6):
    prices = data[['close']].values  # 'close' 열만 선택하고 2차원 배열로 만듦
    num_days, num_stocks = prices.shape
    
    # 첫 날에 모든 자산을 매수
    num_shares = initial_capital // (num_stocks * prices[0, 0])
    capital = initial_capital % (num_stocks * prices[0, 0])
    portfolio = num_shares * np.ones(num_stocks)
    simulated_values = np.zeros(num_days)
    
    for day in range(num_days):
        simulated_values[day] = np.sum(capital + portfolio * (prices[day]))
    
    return simulated_values

def plot_cumulative_returns(test_data, wf_cum_return, Kfold_cum_return, cpcv_cum_return):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title('Cumulative Returns Comparison')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Returns')

    # 날짜 포맷 설정
    date_formatter = mdates.DateFormatter('%Y-%m-%d')

    # 테스트 기간 설정
    test_dates = test_data.date.values[1:]

    # random walk 전략 누적수익률 계산
    random_walk_returns = random_walk_strategy(test_data)
    
    random_walk_perf_stats = DRLAgent.get_perf_stats(random_walk_returns)
    random_walk_cum_returns = random_walk_perf_stats["Cumulative Return"]


    # buy and hold 전략 누적수익률 계산
    buy_and_hold_returns = buy_and_hold_strategy(test_data, initial_capital=1e6)
    buy_and_hold_perf_stats = DRLAgent.get_perf_stats(buy_and_hold_returns)
    print(buy_and_hold_returns)
    buy_and_hold_cum_returns = buy_and_hold_perf_stats["Cumulative Return"]
    

    # 강화학습 전략 누적수익률
    
    ax.plot(test_dates, random_walk_cum_returns, label='Random Walk Strategy', color='red')
    ax.plot(test_dates, buy_and_hold_cum_returns, label='Buy and Hold Strategy', color='green')
    ax.plot(test_dates, wf_cum_return, label='WF Strategy', color='blue')
    ax.plot(test_dates, cpcv_cum_return, label='CPCV Strategy', color='purple')
    ax.plot(test_dates, Kfold_cum_return, label='KFold Strategy', color='orange',linestyle="-.")
    
    
    ax.legend()
    
    # val_result 폴더 생성
    os.makedirs("val_result", exist_ok=True)
    plt.savefig("val_result/cumulative_returns_comparison.png")
    plt.show()
    
    return random_walk_cum_returns,buy_and_hold_cum_returns

def plot_cumulative_returns_3methods(test_data, wf_cum_return, random_walk_cum_returns, buy_and_hold_cum_returns):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title('Cumulative Returns Comparison (Random Walk, Buy and Hold, WF)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Returns')

    test_dates = test_data.date.values[1:]

    ax.plot(test_dates, random_walk_cum_returns, label='Random Walk Strategy', color='red')
    ax.plot(test_dates, buy_and_hold_cum_returns, label='Buy and Hold Strategy', color='green')
    ax.plot(test_dates, wf_cum_return, label='WF Strategy', color='blue')

    ax.legend()
    plt.savefig("val_result/cumulative_returns_comparison_3methods.png")
    plt.show()

def plot_cumulative_returns_wf_kfold_cpcv(test_data, wf_cum_return, Kfold_cum_return, cpcv_cum_return):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title('Cumulative Returns Comparison (WF, KFold, CPCV)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Returns')

    test_dates = test_data.date.values[1:]

    ax.plot(test_dates, wf_cum_return, label='WF Strategy', color='blue')
    ax.plot(test_dates, cpcv_cum_return, label='CPCV Strategy', color='purple')
    ax.plot(test_dates, Kfold_cum_return, label='KFold Strategy', color='orange',linestyle="-.")
    
    ax.legend()
    plt.savefig("val_result/cumulative_returns_comparison_wf_kfold_cpcv.png")
    plt.show()

# 그래프 그리기 함수 호출
test_start_date = pd.to_datetime("2024-03-10 0:00")
test_end_date = pd.to_datetime("2024-03-20 0:00")
random_walk_cum_returns, buy_and_hold_cum_returns = plot_cumulative_returns(test_data, wf_cum_return, Kfold_cum_return, cpcv_cum_return)

# Random Walk, Buy and Hold, WF 방식 그래프 그리기
plot_cumulative_returns_3methods(test_data, wf_cum_return, random_walk_cum_returns, buy_and_hold_cum_returns)

# WF, KFold, CPCV 방식 그래프 그리기
plot_cumulative_returns_wf_kfold_cpcv(test_data, wf_cum_return, Kfold_cum_return, cpcv_cum_return)

# 각 전략의 model metrics 저장
wf_metrics = pd.DataFrame([wf_perf_stats])
Kfold_metrics = pd.DataFrame([Kfold_perf_stats])
cpcv_metrics = pd.DataFrame([cpcv_perf_stats])

random_walk_returns = random_walk_strategy(test_data)
random_walk_perf_stats = DRLAgent.get_perf_stats(random_walk_returns)
random_walk_metrics = pd.DataFrame([random_walk_perf_stats])

buy_and_hold_returns = buy_and_hold_strategy(test_data, initial_capital=1e6)
buy_and_hold_perf_stats = DRLAgent.get_perf_stats(buy_and_hold_returns)
buy_and_hold_metrics = pd.DataFrame([buy_and_hold_perf_stats])

wf_metrics["Cumulative Return"]=wf_metrics["Cumulative Return"][0][-1]
Kfold_metrics["Cumulative Return"]=Kfold_metrics["Cumulative Return"][0][-1]
cpcv_metrics["Cumulative Return"]=cpcv_metrics["Cumulative Return"][0][-1]
random_walk_metrics["Cumulative Return"]=random_walk_metrics["Cumulative Return"][0][-1]
buy_and_hold_metrics["Cumulative Return"]=buy_and_hold_metrics["Cumulative Return"][0][-1]

# val_result 폴더에 각 전략의 model metrics 저장
wf_metrics.to_csv("val_result/wf_model_metrics.csv", index=False)
Kfold_metrics.to_csv("val_result/Kfold_model_metrics.csv", index=False)
cpcv_metrics.to_csv("val_result/cpcv_model_metrics.csv", index=False)
random_walk_metrics.to_csv("val_result/random_walk_model_metrics.csv", index=False)
buy_and_hold_metrics.to_csv("val_result/buy_and_hold_model_metrics.csv", index=False)