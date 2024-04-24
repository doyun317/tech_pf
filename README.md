# BTC-USD 강화학습 트레이딩 전략 비교

이 프로젝트는 BTC-USD 데이터를 사용하여 다양한 강화학습 기반 트레이딩 전략을 비교하고 분석합니다. 비교 대상 전략으로는 Walk Forward Validation (WF), K-Fold Cross Validation (KFold), Combinatorial Purged Cross Validation (CPCV)이 포함되며, 이를 Random Walk 및 Buy and Hold 전략과 비교합니다.

## 파일 구성

- `main_kfold.py`: K-Fold Cross Validation 전략을 구현하고 평가하는 메인 파일입니다.
- `main_wf.py`: Walk Forward Validation 전략을 구현하고 평가하는 메인 파일입니다.
- `main_cpcv.py`: Combinatorial Purged Cross Validation 전략을 구현하고 평가하는 메인 파일입니다.
- `plot_eval.py`: 각 전략의 성능을 시각화하고 비교하는 파일입니다.

## 사용 방법

1. 필요한 라이브러리를 설치합니다.
2. `main_kfold.py`, `main_wf.py`, `main_cpcv.py` 파일을 순차적으로 실행하여 각 전략을 학습하고 평가합니다.
3. `plot_eval.py` 파일을 실행하여 각 전략의 누적 수익률을 비교하는 그래프를 생성합니다.

## 결과

- 각 전략의 학습 및 평가 결과는 `results` 폴더에 저장됩니다.
- 전략별 누적 수익률 비교 그래프는 `val_result` 폴더에 저장됩니다.
- 각 전략의 성능 메트릭(Sharpe Ratio, Cumulative Return, Max Drawdown)은 `val_result` 폴더에 CSV 파일로 저장됩니다.

## 요구사항

- Python 3.x
- pandas
- numpy
- yfinance
- ta
- finrl
- stable-baselines3
- matplotlib
- seaborn
- scikit-learn

## 참고

- 데이터 소스: Yahoo Finance (BTC-USD)
- 강화학습 프레임워크: FinRL, Stable Baselines3

이 프로젝트는 다양한 강화학습 기반 트레이딩 전략을 비교하고 분석하여 최적의 전략을 찾는 것을 목표로 합니다. 각 전략의 구현과 평가 과정을 통해 BTC-USD 데이터에 대한 강화학습 트레이딩 전략의 성능을 이해할 수 있습니다.
