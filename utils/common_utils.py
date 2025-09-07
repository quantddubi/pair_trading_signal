"""
페어트레이딩 공통 유틸리티 함수들
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
from statsmodels.tsa.stattools import adfuller
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kendalltau, spearmanr
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path: str) -> pd.DataFrame:
    """
    BBG 가격 데이터 로딩 및 전처리
    첫번째 행: 티커, 두번째 행: 이름, 세번째 행부터 가격
    """
    # 헤더 2행을 건너뛰고 데이터 읽기
    df = pd.read_csv(file_path, skiprows=2)
    
    # 첫 번째 열을 Date 컬럼으로 설정
    df.columns = ['Date'] + [f'Asset_{i}' for i in range(1, len(df.columns))]
    
    # 실제 컬럼명을 첫 번째 행에서 읽어오기
    with open(file_path, 'r') as f:
        ticker_line = f.readline().strip()
        tickers = ticker_line.split(',')
    
    # 컬럼명을 티커명으로 변경
    if len(tickers) == len(df.columns):
        df.columns = tickers
    
    # 날짜를 인덱스로 설정
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.set_index('Date')
    
    # 숫자형으로 변환
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df.dropna(how='all')

def normalize_prices(prices: pd.DataFrame, method: str = 'rebase') -> pd.DataFrame:
    """
    가격 데이터 표준화
    method: 'rebase' (첫날=1) 또는 'zscore' (평균0, 표준편차1)
    """
    if method == 'rebase':
        return prices.div(prices.iloc[0])
    elif method == 'zscore':
        return (prices - prices.mean()) / prices.std()
    else:
        raise ValueError("Method must be 'rebase' or 'zscore'")

def calculate_spread(price1: pd.Series, price2: pd.Series, hedge_ratio: float = 1.0) -> pd.Series:
    """
    스프레드 계산: price1 - hedge_ratio * price2
    """
    return price1 - hedge_ratio * price2

def calculate_zscore(spread: pd.Series, window: int = 60) -> pd.Series:
    """
    롤링 z-스코어 계산
    """
    rolling_mean = spread.rolling(window=window).mean()
    rolling_std = spread.rolling(window=window).std()
    return (spread - rolling_mean) / rolling_std

def calculate_half_life(spread: pd.Series) -> float:
    """
    AR(1) 모델의 half-life 계산
    HL = -ln(2) / ln(phi)
    """
    spread_diff = spread.diff().dropna()
    spread_lag = spread.shift(1).dropna()
    
    # 길이 맞추기
    min_len = min(len(spread_diff), len(spread_lag))
    spread_diff = spread_diff[-min_len:]
    spread_lag = spread_lag[-min_len:]
    
    try:
        # OLS 회귀: ΔS_t = α + (φ-1) * S_{t-1} + ε_t
        X = np.column_stack([np.ones(len(spread_lag)), spread_lag])
        y = spread_diff.values
        
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        phi = beta[1] + 1
        
        if phi >= 1 or phi <= 0:
            return np.inf
            
        half_life = -np.log(2) / np.log(phi)
        return half_life if half_life > 0 else np.inf
    except:
        return np.inf

def calculate_hedge_ratio_ols(price1: pd.Series, price2: pd.Series) -> Tuple[float, float, pd.Series]:
    """
    OLS 회귀로 헤지 비율 계산
    return: (beta, p_value, residuals)
    """
    X = np.column_stack([np.ones(len(price2)), price2.values])
    y = price1.values
    
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - (beta[0] + beta[1] * price2.values)
        residuals_series = pd.Series(residuals, index=price1.index)
        
        # ADF 테스트
        adf_result = adfuller(residuals, autolag='AIC')
        p_value = adf_result[1]
        
        return beta[1], p_value, residuals_series
    except:
        return 1.0, 1.0, price1 - price2

def calculate_transaction_cost_ratio(spread: pd.Series, cost_estimate: float = 0.001) -> float:
    """
    1σ / 거래비용 비율 계산
    """
    spread_std = spread.std()
    return spread_std / cost_estimate if cost_estimate > 0 else 0

def is_blackout_period(date: pd.Timestamp, blackout_dates: List[pd.Timestamp], buffer_days: int = 2) -> bool:
    """
    이벤트 블랙아웃 기간 체크
    """
    for blackout_date in blackout_dates:
        if abs((date - blackout_date).days) <= buffer_days:
            return True
    return False

def generate_trading_signals(zscore: pd.Series, enter_threshold: float = 2.0, 
                           exit_threshold: float = 0.5, stop_loss: float = 3.0) -> pd.Series:
    """
    진입/청산/손절 신호 생성
    1: 진입 (z > threshold), -1: 진입 (z < -threshold), 0: 청산/관망
    """
    signals = pd.Series(0, index=zscore.index)
    position = 0  # 현재 포지션 상태
    
    for i in range(len(zscore)):
        z = zscore.iloc[i]
        
        if position == 0:  # 포지션 없음
            if abs(z) >= enter_threshold:
                signals.iloc[i] = 1 if z > 0 else -1
                position = signals.iloc[i]
        else:  # 포지션 있음
            # 손절 조건
            if abs(z) >= stop_loss:
                signals.iloc[i] = 0
                position = 0
            # 청산 조건
            elif abs(z) <= exit_threshold:
                signals.iloc[i] = 0
                position = 0
            else:
                signals.iloc[i] = position  # 포지션 유지
                
    return signals

def euclidean_distance_matrix(prices: pd.DataFrame) -> np.ndarray:
    """
    모든 자산 쌍의 유클리드 거리 행렬 계산
    """
    # 각 자산의 표준화된 가격 벡터 간 거리 계산
    distance_matrix = pdist(prices.T.values, metric='euclidean')
    return squareform(distance_matrix)

def get_non_overlapping_pairs(distance_matrix: np.ndarray, asset_names: List[str], 
                            n_pairs: int = 10) -> List[Tuple[str, str, float]]:
    """
    중복 없는 최적 페어 선택 (그리디 방법)
    """
    n_assets = len(asset_names)
    used_assets = set()
    pairs = []
    
    # 거리를 (거리, i, j) 형태로 정리하고 정렬
    distances = []
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            distances.append((distance_matrix[i][j], i, j))
    
    distances.sort()  # 거리가 가장 가까운 순으로 정렬
    
    for dist, i, j in distances:
        if len(pairs) >= n_pairs:
            break
        if i not in used_assets and j not in used_assets:
            pairs.append((asset_names[i], asset_names[j], dist))
            used_assets.add(i)
            used_assets.add(j)
    
    return pairs

def calculate_correlation_regime_change(prices: pd.DataFrame, long_window: int = 252, 
                                      short_window: int = 60) -> pd.DataFrame:
    """
    장기 vs 단기 상관관계 변화 계산
    """
    correlations = {}
    
    for col1 in prices.columns:
        for col2 in prices.columns:
            if col1 < col2:  # 중복 방지
                returns1 = prices[col1].pct_change().dropna()
                returns2 = prices[col2].pct_change().dropna()
                
                # 공통 인덱스
                common_idx = returns1.index.intersection(returns2.index)
                if len(common_idx) < max(long_window, short_window):
                    continue
                    
                returns1_common = returns1[common_idx]
                returns2_common = returns2[common_idx]
                
                # 장기 상관관계
                if len(returns1_common) >= long_window:
                    long_corr = returns1_common.tail(long_window).corr(returns2_common.tail(long_window))
                else:
                    long_corr = returns1_common.corr(returns2_common)
                
                # 단기 상관관계
                if len(returns1_common) >= short_window:
                    short_corr = returns1_common.tail(short_window).corr(returns2_common.tail(short_window))
                else:
                    short_corr = returns1_common.corr(returns2_common)
                
                correlations[(col1, col2)] = {
                    'long_corr': long_corr,
                    'short_corr': short_corr,
                    'delta_corr': abs(short_corr - long_corr)
                }
    
    return pd.DataFrame(correlations).T

def calculate_rank_correlation(price1: pd.Series, price2: pd.Series, method: str = 'kendall') -> float:
    """
    순위 상관관계 계산 (Kendall τ 또는 Spearman ρ)
    """
    returns1 = price1.pct_change().dropna()
    returns2 = price2.pct_change().dropna()
    
    # 공통 인덱스
    common_idx = returns1.index.intersection(returns2.index)
    if len(common_idx) < 30:  # 최소 표본 수
        return 0
        
    returns1_common = returns1[common_idx]
    returns2_common = returns2[common_idx]
    
    try:
        if method == 'kendall':
            corr, _ = kendalltau(returns1_common, returns2_common)
        elif method == 'spearman':
            corr, _ = spearmanr(returns1_common, returns2_common)
        else:
            raise ValueError("Method must be 'kendall' or 'spearman'")
        return corr if not np.isnan(corr) else 0
    except:
        return 0

def calculate_tail_dependence(price1: pd.Series, price2: pd.Series, quantile: float = 0.1) -> float:
    """
    간단한 꼬리 의존성 계산 (동시 극단 사건 발생 빈도)
    """
    returns1 = price1.pct_change().dropna()
    returns2 = price2.pct_change().dropna()
    
    # 공통 인덱스
    common_idx = returns1.index.intersection(returns2.index)
    if len(common_idx) < 100:  # 최소 표본 수
        return 0
        
    returns1_common = returns1[common_idx]
    returns2_common = returns2[common_idx]
    
    # 극단 임계값 계산
    threshold1_low = returns1_common.quantile(quantile)
    threshold1_high = returns1_common.quantile(1 - quantile)
    threshold2_low = returns2_common.quantile(quantile)
    threshold2_high = returns2_common.quantile(1 - quantile)
    
    # 동시 극단 사건
    joint_extreme_low = ((returns1_common <= threshold1_low) & 
                        (returns2_common <= threshold2_low)).sum()
    joint_extreme_high = ((returns1_common >= threshold1_high) & 
                         (returns2_common >= threshold2_high)).sum()
    
    total_extreme = len(returns1_common) * quantile * 2
    return (joint_extreme_low + joint_extreme_high) / total_extreme if total_extreme > 0 else 0