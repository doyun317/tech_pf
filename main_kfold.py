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

# 데이터 저장 경로
data_dir = 'stock_data'
MATHOD_NAME = "_KFOLD"

# 현재 시간을 이름으로 하는 폴더 생성
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
result_dir = os.path.join("results", current_time+MATHOD_NAME)
os.makedirs(result_dir, exist_ok=True)

# 5분 간격 데이터 다운로드 및 저장
def fetch_save_five_minute_data(ticker, start_date, end_date):
    data_file = os.path.join(data_dir, f"{ticker}_{start_date}_{end_date}.csv")
    if not os.path.exists(data_file):
        os.makedirs(data_dir, exist_ok=True)
        data = yf.download(ticker, start=start_date, end=end_date, interval='5m')
        data.to_csv(data_file)
    else:
        data = pd.read_csv(data_file)
    return data

# BTC-USD 5분 간격 데이터 다운로드 및 로드
start_date = '2024-01-25'
end_date = '2024-03-21'
df = fetch_save_five_minute_data('BTC-USD', start_date, end_date)
df = df.reset_index()
df = df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
df['tic'] = 'BTC-USD'
df['date'] = pd.to_datetime(df['date'])  # 'date' 열을 datetime으로 변환
df['date'] = df['date'].dt.tz_localize(None)  # 시간대 정보 제거


# RSI 계산
rsi_indicator = RSIIndicator(close=df['close'], window=14)
df['rsi'] = rsi_indicator.rsi()

print(f"Shape of DataFrame: {df.shape}")

# 기술 지표 계산
fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=['macd', 'cci', 'dx'],
    use_turbulence=False,
    user_defined_feature=False)

processed_df = fe.preprocess_data(df)

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

# 트레인과 테스트 기간 설정

TRAIN_START_DATE = '2024-01-25'
TRAIN_END_DATE = '2024-03-10'
TEST_START_DATE = '2024-03-10'
TEST_END_DATE = '2024-03-20'

# 트레인과 테스트 데이터로 분할
train_data = processed_df[(processed_df.date >= TRAIN_START_DATE) & (processed_df.date < TRAIN_END_DATE)]
test_data = processed_df[(processed_df.date >= TEST_START_DATE) & (processed_df.date <= TEST_END_DATE)]

# 환경 설정
stock_dimension = len(processed_df.tic.unique())
state_space = 1 + 2 + 3 * stock_dimension + len(TECHNICAL_INDICATORS_LIST)
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

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

def train_and_evaluate_model(train_data, test_data, env_kwargs):
    train_env_config = {
        "price_array": train_data.close.values.reshape(-1, 1),
        "tech_array": train_data[TECHNICAL_INDICATORS_LIST].values,
        "turbulence_array": np.zeros(len(train_data)),
        "if_train": True
    }
    e_train_gym = StockTradingEnv(config=train_env_config, **env_kwargs)
    
    test_env_config = {
        "price_array": test_data.close.values.reshape(-1, 1),
        "tech_array": test_data[TECHNICAL_INDICATORS_LIST].values,
        "turbulence_array": np.zeros(len(test_data)),
        "if_train": False
    }
    e_test_gym = StockTradingEnv(config=test_env_config, **env_kwargs)
    
    # 반복 학습
    best_model = None
    best_sharpe_ratio = -np.inf
    total_timesteps = 200000
    iter_ = 10
    
    for i in range(iter_):
        print(f"Iteration {i+1}")
        
        # PPO 에이전트 설정
        agent = DRLAgent(env=e_train_gym)
        model_ppo = agent.get_model("ppo", model_kwargs={"learning_rate": 0.0005})
        
        trained_ppo = agent.train_model(model=model_ppo,
                                        tb_log_name=f'ppo_{i}',
                                        total_timesteps=total_timesteps)
        
        # 검증 데이터에 대한 평가
        df_account_value, _ = DRLAgent.DRL_prediction(model=trained_ppo, environment=e_test_gym)
        
        # sharpe ratio 계산
        perf_stats = DRLAgent.get_perf_stats(df_account_value['account_value'].values)
        sharpe_ratio = perf_stats['Sharpe Ratio']
        cum_return = perf_stats["Cumulative Return"]
        print(f"Sharpe Ratio: {sharpe_ratio}")
        print(f"Cumulative Return: {cum_return[-1]}")
        
        # 최고 sharpe ratio 모델 저장
        if sharpe_ratio > best_sharpe_ratio:  
            best_sharpe_ratio = sharpe_ratio 
            best_model = trained_ppo
        
    
    return best_model, best_sharpe_ratio

def backtesting(best_model):
    # 백테스팅을 위한 환경 설정
    test_data = processed_df[(processed_df.date >= TEST_START_DATE) & (processed_df.date <= TEST_END_DATE)]

    test_env_config = {
        "price_array": test_data.close.values.reshape(-1, 1),
        "tech_array": test_data[TECHNICAL_INDICATORS_LIST].values,
        "turbulence_array": np.zeros(len(test_data)),
        "if_train": False
    }
    e_test_gym = StockTradingEnv(config=test_env_config, **env_kwargs)

    # 각 방식의 최고 성능 모델로 백테스팅 진행

    df_account_value, _ = DRLAgent.DRL_prediction(model=best_model, environment=e_test_gym)
    
    perf_stats = DRLAgent.get_perf_stats(df_account_value['account_value'].values)
    sharpe_ratio = perf_stats['Sharpe Ratio']
    cum_return = perf_stats["Cumulative Return"]
    
    return best_model, sharpe_ratio

def k_fold_cross_validation(processed_df, env_kwargs, result_dir, n_splits=5):
    train_data = processed_df[(processed_df.date >= TRAIN_START_DATE) & (processed_df.date <= TRAIN_END_DATE)]
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    models = []
    best_sharpe_ratios = []
    
    for i, (train_index, test_index) in enumerate(tscv.split(train_data)):
        print(f"Fold {i+1}")
        
        train_fold_data = train_data.iloc[train_index]
        test_fold_data = train_data.iloc[test_index]
        
        best_model, _ = train_and_evaluate_model(train_fold_data, test_fold_data, env_kwargs)

        backtest_best_mdoel, best_sharpe_ratio = backtesting(best_model)
        
        models.append(backtest_best_mdoel)
        best_sharpe_ratios.append(best_sharpe_ratio)
    
    # 최고 sharpe ratio를 가진 모델 선택
    best_model_index = np.argmax(best_sharpe_ratios)
    best_model = models[best_model_index]
    
    # 최고 sharpe ratio 모델 저장
    best_model_path = os.path.join(result_dir, f"best_model{MATHOD_NAME}")
    best_model.save(best_model_path)
    
    return best_model, best_sharpe_ratios 

kfcv_model, k_fold_models_cum_returns = k_fold_cross_validation(processed_df, env_kwargs, result_dir)
print(k_fold_models_cum_returns)

def best_model_eval(processed_df, result_dir):
    test_data = processed_df[(processed_df.date >= TEST_START_DATE) & (processed_df.date <= TEST_END_DATE)]

    test_env_config = {
        "price_array": test_data.close.values.reshape(-1, 1),
        "tech_array": test_data[TECHNICAL_INDICATORS_LIST].values,
        "turbulence_array": np.zeros(len(test_data)),
        "if_train": False
    }
    e_test_gym = StockTradingEnv(config=test_env_config, **env_kwargs)
    
    best_sharp_ratio = -np.inf
    best_eval_values=[]
    # 각 방식의 최고 성능 모델로 백테스팅 진행
    cwd = os.path.join(result_dir, "best_model_KFOLD")
    for i in range(50):
        df_best_account_value = DRLAgent.DRL_prediction_load_from_file("ppo",environment=e_test_gym,cwd=cwd)
        df_best_account_value = pd.DataFrame(df_best_account_value,columns=["account_value"])
        
        perf_stats = DRLAgent.get_perf_stats(df_best_account_value['account_value'].values)
        sharpe_ratio = perf_stats['Sharpe Ratio']
        
        if sharpe_ratio > best_sharp_ratio:
            best_eval_values = df_best_account_value
            best_sharp_ratio = sharpe_ratio
    
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

    best_account_value_path = os.path.join(result_dir, f"best_account_value{MATHOD_NAME}.csv")
    df_best_account_value.to_csv(best_account_value_path, index=False)
    
    best_model_metrics_path = os.path.join(result_dir, f"best_model_metrics{MATHOD_NAME}.csv")
    best_model_metrics.to_csv(best_model_metrics_path, index=False)
    
    print(f"Sharpe Ratio: {sharpe_ratio}")
    print(f"Best cumlative Return: {cum_return}")
    
    
best_model_eval(processed_df,result_dir)