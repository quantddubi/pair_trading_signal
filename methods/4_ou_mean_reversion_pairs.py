"""
4) OU(Ornstein-Uhlenbeck) 평균회귀 속도 기반 - 속도로 품질 선별
핵심: 스프레드가 얼마나 빨리 평균으로 돌아오는가(속도/반감기)로 페어를 필터링
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from scipy.optimize import minimize
import os
import sys
import importlib.util

# 공통 유틸리티 동적 import
def import_common_utils():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    utils_path = os.path.join(os.path.dirname(current_dir), "utils", "common_utils.py")
    spec = importlib.util.spec_from_file_location("common_utils", utils_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# 공통 함수들 import
common_utils = import_common_utils()
load_data = common_utils.load_data
normalize_prices = common_utils.normalize_prices
calculate_spread = common_utils.calculate_spread
calculate_zscore = common_utils.calculate_zscore
calculate_half_life = common_utils.calculate_half_life
generate_trading_signals = common_utils.generate_trading_signals
calculate_transaction_cost_ratio = common_utils.calculate_transaction_cost_ratio
calculate_hedge_ratio_ols = common_utils.calculate_hedge_ratio_ols

class OUMeanReversionPairTrading:
    def __init__(self, formation_window: int = 252, signal_window: int = 60,
                 enter_threshold: float = 2.0, exit_threshold: float = 0.5,
                 stop_loss: float = 3.0, min_half_life: int = 5, max_half_life: int = 60,
                 min_cost_ratio: float = 5.0, min_mean_reversion_speed: float = 0.01):
        """
        OU 평균회귀 속도 기반 페어트레이딩 파라미터
        
        Args:
            formation_window: 페어 선정 기간 (영업일)
            signal_window: z-스코어 계산 롤링 윈도우
            enter_threshold: 진입 z-스코어 임계값
            exit_threshold: 청산 z-스코어 임계값
            stop_loss: 손절 z-스코어 임계값
            min_half_life: 최소 반감기 (영업일)
            max_half_life: 최대 반감기 (영업일)
            min_cost_ratio: 최소 1σ/거래비용 비율
            min_mean_reversion_speed: 최소 평균회귀 속도 (κ)
        """
        self.formation_window = formation_window
        self.signal_window = signal_window
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss = stop_loss
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.min_cost_ratio = min_cost_ratio
        self.min_mean_reversion_speed = min_mean_reversion_speed
    
    def fit_ou_process_ar1(self, spread: pd.Series) -> Tuple[float, float, float]:
        """
        AR(1) 모델로 OU 프로세스 파라미터 추정
        S_t = φ * S_{t-1} + ε_t
        OU 파라미터: κ = -ln(φ), half_life = ln(2)/κ
        
        Returns:
            (kappa, half_life, phi): 평균회귀 속도, 반감기, AR(1) 계수
        """
        try:
            spread_clean = spread.dropna()
            if len(spread_clean) < 30:
                return 0, np.inf, 1
            
            spread_diff = spread_clean.diff().dropna()
            spread_lag = spread_clean.shift(1).dropna()
            
            # 길이 맞추기
            min_len = min(len(spread_diff), len(spread_lag))
            spread_diff = spread_diff[-min_len:]
            spread_lag = spread_lag[-min_len:]
            
            # AR(1): ΔS_t = α + (φ-1) * S_{t-1} + ε_t
            X = np.column_stack([np.ones(len(spread_lag)), spread_lag.values])
            y = spread_diff.values
            
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            phi = beta[1] + 1  # φ = 1 + (φ-1)
            
            if phi >= 1 or phi <= 0:
                return 0, np.inf, phi
                
            kappa = -np.log(phi)
            half_life = np.log(2) / kappa if kappa > 0 else np.inf
            
            return kappa, half_life, phi
            
        except Exception:
            return 0, np.inf, 1
    
    def fit_ou_process_mle(self, spread: pd.Series, dt: float = 1.0/252) -> Tuple[float, float, float, float]:
        """
        최대우도법(MLE)으로 OU 프로세스 파라미터 추정
        dS_t = κ(μ - S_t)dt + σ dW_t
        
        Args:
            spread: 스프레드 시계열
            dt: 시간 간격 (일 단위, 기본값: 1영업일 = 1/252년)
            
        Returns:
            (kappa, mu, sigma, half_life): 평균회귀 속도, 장기 평균, 변동성, 반감기
        """
        try:
            spread_clean = spread.dropna()
            if len(spread_clean) < 50:
                return 0, spread_clean.mean(), spread_clean.std(), np.inf
            
            def neg_log_likelihood(params):
                kappa, mu, sigma = params
                if kappa <= 0 or sigma <= 0:
                    return 1e10
                
                n = len(spread_clean)
                log_likelihood = 0
                
                for i in range(1, n):
                    s_prev = spread_clean.iloc[i-1]
                    s_curr = spread_clean.iloc[i]
                    
                    # OU 프로세스의 조건부 분포
                    mean = s_prev * np.exp(-kappa * dt) + mu * (1 - np.exp(-kappa * dt))
                    variance = (sigma**2) * (1 - np.exp(-2 * kappa * dt)) / (2 * kappa)
                    
                    if variance <= 0:
                        return 1e10
                    
                    # 로그 우도
                    log_likelihood += -0.5 * np.log(2 * np.pi * variance) - \
                                     (s_curr - mean)**2 / (2 * variance)
                
                return -log_likelihood
            
            # 초기값 설정
            initial_mu = spread_clean.mean()
            initial_sigma = spread_clean.std()
            initial_kappa = 0.1  # 적당한 평균회귀 속도
            
            # 제약조건
            bounds = [(1e-6, 10), (-10, 10), (1e-6, 10)]  # (kappa, mu, sigma)
            initial_guess = [initial_kappa, initial_mu, initial_sigma]
            
            result = minimize(neg_log_likelihood, initial_guess, bounds=bounds, 
                            method='L-BFGS-B')
            
            if result.success:
                kappa, mu, sigma = result.x
                half_life = np.log(2) / kappa if kappa > 0 else np.inf
                return kappa, mu, sigma, half_life
            else:
                return 0, initial_mu, initial_sigma, np.inf
                
        except Exception:
            return 0, spread.mean() if len(spread) > 0 else 0, \
                   spread.std() if len(spread) > 0 else 1, np.inf
    
    def calculate_ou_quality_score(self, kappa: float, half_life: float, sigma: float) -> float:
        """
        OU 프로세스 품질 점수 계산
        높은 평균회귀 속도, 적당한 반감기, 낮은 변동성 선호
        
        Returns:
            quality_score: 0~100 점수
        """
        score = 0
        
        # 1. 평균회귀 속도 점수 (κ)
        if kappa >= self.min_mean_reversion_speed:
            kappa_score = min(100, kappa * 500)  # κ=0.2이면 100점
            score += kappa_score * 0.4
        
        # 2. 반감기 점수
        if self.min_half_life <= half_life <= self.max_half_life:
            # 최적 반감기: 20일 정도
            optimal_hl = 20
            hl_score = max(0, 100 - abs(half_life - optimal_hl) * 2)
            score += hl_score * 0.4
        
        # 3. 안정성 점수 (변동성 역수)
        if sigma > 0:
            stability_score = min(100, 50 / sigma)
            score += stability_score * 0.2
        
        return score
    
    def select_pairs(self, prices: pd.DataFrame, n_pairs: int = 20) -> List[Dict]:
        """
        OU 평균회귀 속도 기반 페어 선정
        
        Args:
            prices: 가격 데이터
            n_pairs: 선정할 페어 개수
            
        Returns:
            선정된 페어 정보 리스트
        """
        # 최근 formation_window 기간 데이터 추출
        formation_data = prices.tail(self.formation_window)
        
        # 결측치가 많은 자산 제외
        valid_assets = []
        for col in formation_data.columns:
            if formation_data[col].notna().sum() >= self.formation_window * 0.9:
                valid_assets.append(col)
        
        if len(valid_assets) < 2:
            return []
            
        formation_data = formation_data[valid_assets].fillna(method='ffill')
        
        # OU 프로세스 분석 결과
        ou_results = []
        
        for i, asset1 in enumerate(valid_assets):
            for j, asset2 in enumerate(valid_assets):
                if i >= j:  # 중복 방지
                    continue
                
                # OLS 헤지비율 추정
                hedge_ratio, p_value, residuals = calculate_hedge_ratio_ols(
                    formation_data[asset1], formation_data[asset2]
                )
                
                if len(residuals) < 50:  # 최소 표본 수
                    continue
                
                # AR(1) 방법으로 OU 파라미터 추정
                kappa_ar1, half_life_ar1, phi = self.fit_ou_process_ar1(residuals)
                
                # MLE 방법으로 OU 파라미터 추정
                kappa_mle, mu, sigma, half_life_mle = self.fit_ou_process_mle(residuals)
                
                # 두 방법 결과 평균
                kappa_avg = (kappa_ar1 + kappa_mle) / 2
                half_life_avg = (half_life_ar1 + half_life_mle) / 2
                
                # 거래비용 대비 수익성
                cost_ratio = calculate_transaction_cost_ratio(residuals)
                
                # OU 품질 점수
                quality_score = self.calculate_ou_quality_score(kappa_avg, half_life_avg, sigma)
                
                # 필터링 조건
                if (self.min_half_life <= half_life_avg <= self.max_half_life and
                    cost_ratio >= self.min_cost_ratio and
                    kappa_avg >= self.min_mean_reversion_speed and
                    quality_score >= 30):  # 최소 품질 점수
                    
                    ou_results.append({
                        'asset1': asset1,
                        'asset2': asset2,
                        'hedge_ratio': hedge_ratio,
                        'p_value': p_value,
                        'kappa_ar1': kappa_ar1,
                        'kappa_mle': kappa_mle,
                        'kappa_avg': kappa_avg,
                        'half_life_ar1': half_life_ar1,
                        'half_life_mle': half_life_mle,
                        'half_life_avg': half_life_avg,
                        'mu': mu,
                        'sigma': sigma,
                        'cost_ratio': cost_ratio,
                        'quality_score': quality_score,
                        'method': 'ou_mean_reversion'
                    })
        
        # 품질 점수가 높은 순으로 정렬
        ou_results.sort(key=lambda x: x['quality_score'], reverse=True)
        
        # 중복 없는 페어 선정
        selected_pairs = []
        used_assets = set()
        
        for result in ou_results:
            if len(selected_pairs) >= n_pairs:
                break
                
            asset1, asset2 = result['asset1'], result['asset2']
            if asset1 not in used_assets and asset2 not in used_assets:
                selected_pairs.append(result)
                used_assets.add(asset1)
                used_assets.add(asset2)
        
        return selected_pairs
    
    def calculate_time_stop(self, half_life: float, multiplier: float = 2.0) -> int:
        """
        반감기 기반 타임스탑 계산
        
        Args:
            half_life: 반감기 (영업일)
            multiplier: 반감기 배수 (기본값: 2배)
            
        Returns:
            time_stop: 타임스탑 일수
        """
        return int(half_life * multiplier) if half_life != np.inf else 60
    
    def generate_signals(self, prices: pd.DataFrame, pair_info: Dict) -> Dict:
        """
        특정 페어에 대한 트레이딩 신호 생성
        
        Args:
            prices: 전체 가격 데이터
            pair_info: 페어 정보 딕셔너리
            
        Returns:
            신호 정보 딕셔너리
        """
        asset1, asset2 = pair_info['asset1'], pair_info['asset2']
        
        # 최근 데이터 확보
        recent_data = prices[[asset1, asset2]].tail(self.signal_window * 2).fillna(method='ffill')
        
        if len(recent_data) < self.signal_window:
            return {'status': 'insufficient_data'}
        
        # 스프레드 계산
        spread = calculate_spread(
            recent_data[asset1],
            recent_data[asset2],
            hedge_ratio=pair_info['hedge_ratio']
        )
        
        # Z-스코어 계산
        zscore = calculate_zscore(spread, window=self.signal_window)
        current_zscore = zscore.iloc[-1] if not zscore.empty else 0
        
        # 신호 생성
        signals = generate_trading_signals(
            zscore, 
            enter_threshold=self.enter_threshold,
            exit_threshold=self.exit_threshold,
            stop_loss=self.stop_loss
        )
        
        current_signal = signals.iloc[-1] if not signals.empty else 0
        
        # 타임스탑 계산
        time_stop = self.calculate_time_stop(pair_info['half_life_avg'])
        
        # 신호 해석
        if current_signal == 1:
            signal_type = "ENTER_LONG"
            direction = f"Long {asset1}, Short {asset2} (ratio: {pair_info['hedge_ratio']:.3f})"
        elif current_signal == -1:
            signal_type = "ENTER_SHORT"
            direction = f"Short {asset1}, Long {asset2} (ratio: {pair_info['hedge_ratio']:.3f})"
        else:
            signal_type = "EXIT_OR_WAIT"
            direction = "Exit or Wait"
        
        return {
            'status': 'success',
            'pair': f"{asset1}-{asset2}",
            'signal_type': signal_type,
            'direction': direction,
            'current_zscore': current_zscore,
            'kappa_avg': pair_info['kappa_avg'],
            'half_life_avg': pair_info['half_life_avg'],
            'quality_score': pair_info['quality_score'],
            'time_stop': time_stop,
            'hedge_ratio': pair_info['hedge_ratio'],
            'cost_ratio': pair_info['cost_ratio'],
            'method': 'ou_mean_reversion'
        }
    
    def screen_pairs(self, prices: pd.DataFrame, n_pairs: int = 10) -> Tuple[List[Dict], List[Dict]]:
        """
        전체 페어 스크리닝 및 신호 생성
        
        Returns:
            (enter_signals, watch_signals): 진입 신호와 관찰 대상 리스트
        """
        # 페어 선정
        selected_pairs = self.select_pairs(prices, n_pairs * 2)
        
        enter_signals = []
        watch_signals = []
        
        for pair_info in selected_pairs:
            signal_result = self.generate_signals(prices, pair_info)
            
            if signal_result['status'] != 'success':
                continue
                
            current_z = abs(signal_result['current_zscore'])
            
            # 진입 신호 (|z| >= 2.0)
            if current_z >= self.enter_threshold:
                enter_signals.append(signal_result)
            # 관찰 대상 (1.5 <= |z| < 2.0)
            elif 1.5 <= current_z < self.enter_threshold:
                watch_signals.append(signal_result)
        
        # 품질 점수가 높은 순으로 정렬
        enter_signals.sort(key=lambda x: x['quality_score'], reverse=True)
        watch_signals.sort(key=lambda x: x['quality_score'], reverse=True)
        
        return enter_signals[:n_pairs], watch_signals[:n_pairs]

def main():
    """
    OU 평균회귀 속도 기반 페어트레이딩 실행 예제
    """
    # 데이터 로딩
    file_path = "/Users/a/PycharmProjects/pair_trading_signal/data/MU Price(BBG).csv"
    prices = load_data(file_path)
    
    # OU 평균회귀 기반 페어트레이딩 객체 생성
    ou_trader = OUMeanReversionPairTrading(
        formation_window=252,         # 1년
        signal_window=60,             # 3개월
        enter_threshold=2.0,
        exit_threshold=0.5,
        stop_loss=3.0,
        min_half_life=5,
        max_half_life=60,
        min_cost_ratio=5.0,
        min_mean_reversion_speed=0.01  # 최소 평균회귀 속도
    )
    
    # 페어 스크리닝
    enter_list, watch_list = ou_trader.screen_pairs(prices, n_pairs=10)
    
    print("=" * 75)
    print("OU 평균회귀 속도 기반 페어트레이딩 신호")
    print("=" * 75)
    
    print(f"\n📈 진입 신호 ({len(enter_list)}개):")
    print("-" * 65)
    for i, signal in enumerate(enter_list, 1):
        print(f"{i:2d}. {signal['pair']:20s} | {signal['direction']:35s}")
        print(f"     Z-Score: {signal['current_zscore']:6.2f} | Half-Life: {signal['half_life_avg']:4.1f}D")
        print(f"     κ (평균회귀속도): {signal['kappa_avg']:6.4f} | 품질점수: {signal['quality_score']:5.1f}")
        print(f"     타임스탑: {signal['time_stop']:2d}일 | 비용비율: {signal['cost_ratio']:5.1f}")
        print(f"     헤지비율: {signal['hedge_ratio']:6.3f}")
        print()
    
    print(f"\n👀 관찰 대상 ({len(watch_list)}개):")
    print("-" * 65)
    for i, signal in enumerate(watch_list, 1):
        print(f"{i:2d}. {signal['pair']:20s} | Z-Score: {signal['current_zscore']:6.2f}")
        print(f"     κ: {signal['kappa_avg']:6.4f} | Half-Life: {signal['half_life_avg']:4.1f}D")
        print(f"     품질점수: {signal['quality_score']:5.1f} | 타임스탑: {signal['time_stop']:2d}일")
        print()

if __name__ == "__main__":
    main()