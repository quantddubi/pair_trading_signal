"""
4) OU(Ornstein-Uhlenbeck) 평균회귀 속도 기반 - 속도로 품질 선별
핵심: 스프레드가 얼마나 빨리 평균으로 돌아오는가(속도/반감기)로 페어를 필터링
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from scipy.optimize import minimize
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # tqdm이 없으면 더미 클래스 사용
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, unit=None):
            self.total = total or 0
            self.n = 0
            self.desc = desc or ""
            print(f"\n{self.desc}: 시작")
            
        def __enter__(self):
            return self
            
        def __exit__(self, *args):
            print(f"완료! 총 {self.n}/{self.total} 처리됨")
            
        def update(self, n=1):
            self.n += n
            if self.n % 100 == 0 or self.n == self.total:  # 100개마다 또는 마지막에 출력
                print(f"진행률: {self.n}/{self.total} ({self.n/self.total*100:.1f}%)")
        
        def set_postfix(self, postfix_dict):
            pass  # 간단하게 무시

import time
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
    def __init__(self, formation_window: int = 252, rolling_window: int = 60,
                 base_threshold: float = 1.25, exit_threshold: float = 0.5,
                 stop_loss: float = 3.0, min_half_life: int = 5, max_half_life: int = 60,
                 min_cost_ratio: float = 5.0, min_mean_reversion_speed: float = 0.01,
                 max_kappa_cv: float = 0.6, data_coverage_threshold: float = 0.9,
                 winsorize_percentile: float = 0.01):
        """
        개선된 OU 평균회귀 속도 기반 페어트레이딩 파라미터
        
        Args:
            formation_window: 페어 선정 기간 (영업일)
            rolling_window: OU 파라미터 추정 롤링 윈도우 (권장: 60일)
            base_threshold: 기본 s-score 진입 임계값 (1.25)
            exit_threshold: 청산 임계값
            stop_loss: 손절 임계값
            min_half_life: 최소 반감기 (영업일)
            max_half_life: 최대 반감기 (영업일)
            min_cost_ratio: 최소 1σ/거래비용 비율
            min_mean_reversion_speed: 최소 평균회귀 속도 (κ >= 0.01)
            max_kappa_cv: κ 변동계수 최대값 (안정성 체크)
            data_coverage_threshold: 최소 데이터 커버리지 (90%)
            winsorize_percentile: 윈저라이즈 퍼센타일 (1%)
        """
        self.formation_window = formation_window
        self.rolling_window = rolling_window
        self.base_threshold = base_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss = stop_loss
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.min_cost_ratio = min_cost_ratio
        self.min_mean_reversion_speed = min_mean_reversion_speed
        self.max_kappa_cv = max_kappa_cv
        self.data_coverage_threshold = data_coverage_threshold
        self.winsorize_percentile = winsorize_percentile
    
    def winsorize_data(self, data: pd.Series) -> pd.Series:
        """
        윈저라이즈를 통한 이상치 처리 (상하 1% 컷)
        """
        if len(data.dropna()) < 10:
            return data
        
        lower_percentile = self.winsorize_percentile * 100
        upper_percentile = 100 - self.winsorize_percentile * 100
        
        lower_bound = data.quantile(lower_percentile / 100)
        upper_bound = data.quantile(upper_percentile / 100)
        
        return data.clip(lower=lower_bound, upper=upper_bound)
    
    def check_data_quality(self, prices: pd.DataFrame, asset1: str, asset2: str) -> bool:
        """
        데이터 품질 체크 (커버리지 >= 90%)
        """
        formation_data = prices[[asset1, asset2]].tail(self.formation_window)
        
        # 각 자산의 데이터 커버리지 체크
        asset1_coverage = formation_data[asset1].notna().sum() / len(formation_data)
        asset2_coverage = formation_data[asset2].notna().sum() / len(formation_data)
        
        return (asset1_coverage >= self.data_coverage_threshold and 
                asset2_coverage >= self.data_coverage_threshold)
    
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
    
    def fit_ou_process_mle(self, spread: pd.Series, dt: float = 1.0) -> Tuple[float, float, float, float]:
        """
        최대우도법(MLE)으로 OU 프로세스 파라미터 추정
        dS_t = κ(μ - S_t)dt + σ dW_t
        
        Args:
            spread: 스프레드 시계열
            dt: 시간 간격 (일 단위, 기본값: 1영업일)
            
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
    
    def combine_ou_estimates(self, ar1_result: Tuple, mle_result: Tuple, 
                           ar1_weight: float = 0.4, mle_weight: float = 0.6) -> Tuple[float, float, float, float]:
        """
        AR(1)과 MLE 추정 결과를 가중 평균으로 결합
        
        Args:
            ar1_result: (kappa, half_life, phi)
            mle_result: (kappa, mu, sigma, half_life)
            ar1_weight: AR(1) 가중치 (40%)
            mle_weight: MLE 가중치 (60%)
            
        Returns:
            (kappa_avg, theta_avg, sigma, half_life_avg): 결합된 파라미터
        """
        ar1_kappa, ar1_hl, ar1_phi = ar1_result
        mle_kappa, mle_mu, mle_sigma, mle_hl = mle_result
        
        # 유효한 추정값이 있는지 확인
        ar1_valid = (ar1_kappa > 0 and np.isfinite(ar1_hl) and 0 < ar1_phi < 1)
        mle_valid = (mle_kappa > 0 and np.isfinite(mle_hl))
        
        if ar1_valid and mle_valid:
            # 둘 다 유효한 경우 가중 평균
            kappa_avg = ar1_kappa * ar1_weight + mle_kappa * mle_weight
            hl_avg = ar1_hl * ar1_weight + mle_hl * mle_weight
            theta_avg = mle_mu  # MLE의 장기 평균 사용
            sigma = mle_sigma   # MLE의 변동성 사용
        elif ar1_valid:
            # AR(1)만 유효한 경우
            kappa_avg = ar1_kappa
            hl_avg = ar1_hl
            theta_avg = 0  # 평균 회귀 수준을 0으로 가정
            sigma = 1      # 기본 변동성
        elif mle_valid:
            # MLE만 유효한 경우
            kappa_avg = mle_kappa
            hl_avg = mle_hl
            theta_avg = mle_mu
            sigma = mle_sigma
        else:
            # 둘 다 유효하지 않은 경우
            return 0, 0, 1, np.inf
            
        return kappa_avg, theta_avg, sigma, hl_avg
    
    def calculate_s_score(self, spread: pd.Series, kappa: float, theta: float, sigma: float) -> pd.Series:
        """
        OU 정상상태 표준편차를 사용한 s-score 계산
        
        Args:
            spread: 스프레드 시계열
            kappa: 평균회귀 속도
            theta: 장기 평균
            sigma: 변동성
            
        Returns:
            s_score: OU 조정된 표준화 점수
        """
        if kappa <= 0:
            # 유효하지 않은 kappa인 경우 일반 z-score 반환
            return (spread - spread.mean()) / spread.std()
        
        # OU 정상상태 표준편차: σ_eq = σ / sqrt(2κ)
        sigma_eq = sigma / np.sqrt(2 * kappa)
        
        # s-score = (X_t - θ) / σ_eq
        s_score = (spread - theta) / sigma_eq
        
        return s_score
    
    def calculate_dynamic_threshold(self, kappa: float, cost_ratio: float, 
                                  base_threshold: float = None) -> float:
        """
        동적 진입 임계값 계산
        κ↑, cost_ratio↑일수록 문턱 완화
        """
        if base_threshold is None:
            base_threshold = self.base_threshold
            
        threshold = base_threshold
        
        # κ가 높으면 문턱을 낮춤 (빠른 평균회귀)
        if kappa > 0.05:
            threshold *= 0.9  # 10% 완화
        if kappa > 0.1:
            threshold *= 0.95  # 추가 5% 완화
            
        # 비용 여유가 클수록 문턱을 낮춤
        if cost_ratio > 10:
            threshold *= 0.95  # 5% 완화
        if cost_ratio > 20:
            threshold *= 0.9   # 추가 10% 완화
            
        return threshold
    
    def check_anti_chop_filter(self, s_score: pd.Series, window: int = 5) -> bool:
        """
        앤티-챱(anti-chop) 필터: 최근 며칠간 과도한 왕복 여부 체크
        
        Args:
            s_score: s-score 시계열
            window: 체크할 기간 (기본: 5일)
            
        Returns:
            bool: True면 통과(진입 가능), False면 차단(과도한 왕복)
        """
        if len(s_score) < window:
            return True
            
        recent_scores = s_score.tail(window)
        
        # 방향 변화 횟수 계산
        sign_changes = 0
        for i in range(1, len(recent_scores)):
            if (recent_scores.iloc[i] > 0) != (recent_scores.iloc[i-1] > 0):
                sign_changes += 1
        
        # 변화가 너무 많으면 (3회 이상) 차단
        return sign_changes <= 3
    
    def _check_kappa_stability(self, spread: pd.Series) -> float:
        """
        롤링 윈도우로 κ의 안정성(변동계수) 체크
        
        Args:
            spread: 스프레드 시계열
            
        Returns:
            kappa_cv: κ의 변동계수 (CV = std/mean)
        """
        if len(spread) < self.rolling_window * 2:
            return 1.0  # 데이터 부족시 불안정으로 간주
        
        window_size = self.rolling_window
        kappa_values = []
        
        # 롤링 윈도우로 κ 추정
        for i in range(window_size, len(spread) - window_size + 1, 10):  # 10일 간격으로 샘플링
            window_data = spread.iloc[i-window_size:i]
            kappa, _, _ = self.fit_ou_process_ar1(window_data)
            if kappa > 0 and np.isfinite(kappa):
                kappa_values.append(kappa)
        
        if len(kappa_values) < 3:
            return 1.0  # 충분한 추정값이 없으면 불안정
        
        kappa_array = np.array(kappa_values)
        mean_kappa = np.mean(kappa_array)
        std_kappa = np.std(kappa_array)
        
        # 변동계수 계산
        cv = std_kappa / mean_kappa if mean_kappa > 0 else 1.0
        
        return cv
    
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
        개선된 OU 평균회귀 속도 기반 페어 선정
        
        Args:
            prices: 가격 데이터
            n_pairs: 선정할 페어 개수
            
        Returns:
            선정된 페어 정보 리스트
        """
        # 최근 formation_window 기간 데이터 추출
        formation_data = prices.tail(self.formation_window)
        
        # 데이터 품질 체크 및 유효 자산 선별 (커버리지 >= 90%)
        valid_assets = []
        for col in formation_data.columns:
            coverage = formation_data[col].notna().sum() / len(formation_data)
            if coverage >= self.data_coverage_threshold:
                valid_assets.append(col)
        
        if len(valid_assets) < 2:
            return []
            
        # FFill로 결측치 보간 (과도하지 않게)
        formation_data = formation_data[valid_assets].fillna(method='ffill')
        
        # OU 프로세스 분석 결과
        ou_results = []
        
        # 전체 페어 개수 계산
        total_pairs = len(valid_assets) * (len(valid_assets) - 1) // 2
        print(f"\n[INFO] OU Mean Reversion 페어 분석 시작")
        print(f"[INFO] 분석 대상: {len(valid_assets)}개 자산, {total_pairs}개 페어 조합")
        print(f"[INFO] 예상 소요시간: {total_pairs * 0.5:.0f}초 (페어당 ~0.5초)")
        
        processed_pairs = 0
        
        # 프로그레스 바와 함께 페어 분석
        with tqdm(total=total_pairs, desc="[PROGRESS] 페어 분석 중", unit="pairs") as pbar:
            for i, asset1 in enumerate(valid_assets):
                for j, asset2 in enumerate(valid_assets):
                    if i >= j:  # 중복 방지
                        continue
                    
                    # 개별 자산 데이터 품질 체크
                    if not self.check_data_quality(prices, asset1, asset2):
                        pbar.update(1)
                        continue
                
                    # OLS 헤지비율 추정
                    hedge_ratio, p_value, residuals = calculate_hedge_ratio_ols(
                        formation_data[asset1], formation_data[asset2]
                    )
                    
                    if len(residuals) < 50:  # 최소 표본 수
                        pbar.update(1)
                        continue
                    
                    # 윈저라이즈로 이상치 처리
                    residuals_clean = self.winsorize_data(residuals)
                    
                    # AR(1) 방법으로 OU 파라미터 추정
                    ar1_result = self.fit_ou_process_ar1(residuals_clean)
                    
                    # MLE 방법으로 OU 파라미터 추정 (일 단위)
                    mle_result = self.fit_ou_process_mle(residuals_clean, dt=1.0)
                    
                    # 두 방법 결과를 가중 평균으로 결합 (MLE 60%, AR1 40%)
                    kappa_avg, theta_avg, sigma, half_life_avg = self.combine_ou_estimates(
                        ar1_result, mle_result, ar1_weight=0.4, mle_weight=0.6
                    )
                    
                    # 롤링 윈도우로 κ 안정성 체크
                    kappa_stability = self._check_kappa_stability(residuals_clean)
                    
                    # 거래비용 대비 수익성
                    cost_ratio = calculate_transaction_cost_ratio(residuals_clean)
                    
                    # OU 품질 점수
                    quality_score = self.calculate_ou_quality_score(kappa_avg, half_life_avg, sigma)
                    
                    # 품질 1차 필터 (Quality gate)
                    passes_quality_gate = (
                        self.min_half_life <= half_life_avg <= self.max_half_life and  # 반감기
                        kappa_avg >= self.min_mean_reversion_speed and              # 속도
                        cost_ratio >= self.min_cost_ratio and                       # 비용 여유
                        kappa_stability <= self.max_kappa_cv and                    # 안정성
                        quality_score >= 30                                         # 최소 품질
                    )
                    
                    if passes_quality_gate:
                        ou_results.append({
                            'asset1': asset1,
                            'asset2': asset2,
                            'hedge_ratio': hedge_ratio,
                            'p_value': p_value,
                            'kappa_ar1': ar1_result[0],
                            'kappa_mle': mle_result[0],
                            'kappa_avg': kappa_avg,
                            'half_life_ar1': ar1_result[1],
                            'half_life_mle': mle_result[3],
                            'half_life_avg': half_life_avg,
                            'theta_avg': theta_avg,
                            'sigma': sigma,
                            'kappa_stability': kappa_stability,
                            'cost_ratio': cost_ratio,
                            'quality_score': quality_score,
                            'method': 'ou_mean_reversion'
                        })
                    
                    # 프로그레스 바 업데이트
                    pbar.update(1)
                    pbar.set_postfix({
                        'Current': f"{asset1[:8]}-{asset2[:8]}",
                        'Found': len(ou_results)
                    })
        
        # 분석 완료 메시지
        print(f"\n[COMPLETE] 페어 분석 완료!")
        print(f"[RESULT] 총 분석: {total_pairs}개 페어")
        print(f"[RESULT] 품질 통과: {len(ou_results)}개 페어 ({len(ou_results)/total_pairs*100:.1f}%)")
        
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
        개선된 s-score 기반 트레이딩 신호 생성
        
        Args:
            prices: 전체 가격 데이터
            pair_info: 페어 정보 딕셔너리
            
        Returns:
            신호 정보 딕셔너리
        """
        asset1, asset2 = pair_info['asset1'], pair_info['asset2']
        
        # 최근 데이터 확보 (롤링 윈도우 기준)
        recent_data = prices[[asset1, asset2]].tail(self.rolling_window * 2).fillna(method='ffill')
        
        if len(recent_data) < self.rolling_window:
            return {'status': 'insufficient_data'}
        
        # 스프레드 계산 (X_t = asset2_t - h * asset1_t)
        spread = calculate_spread(
            recent_data[asset1],
            recent_data[asset2],
            hedge_ratio=pair_info['hedge_ratio']
        )
        
        # 윈저라이즈로 이상치 처리
        spread_clean = self.winsorize_data(spread)
        
        # 롤링 윈도우로 OU 파라미터 재추정 (최신 데이터 반영)
        rolling_data = spread_clean.tail(self.rolling_window)
        ar1_result = self.fit_ou_process_ar1(rolling_data)
        mle_result = self.fit_ou_process_mle(rolling_data, dt=1.0)
        
        # 파라미터 결합
        kappa_avg, theta_avg, sigma, half_life_avg = self.combine_ou_estimates(
            ar1_result, mle_result, ar1_weight=0.4, mle_weight=0.6
        )
        
        # s-score 계산 (OU 정상상태 표준편차 사용)
        s_score = self.calculate_s_score(spread_clean, kappa_avg, theta_avg, sigma)
        current_s_score = s_score.iloc[-1] if not s_score.empty else 0
        
        # 거래비용 비율 (최신 데이터 기준)
        cost_ratio = pair_info.get('cost_ratio', self.min_cost_ratio)
        
        # 동적 임계값 계산
        dynamic_threshold = self.calculate_dynamic_threshold(kappa_avg, cost_ratio)
        
        # 앤티-챱 필터 체크
        anti_chop_ok = self.check_anti_chop_filter(s_score)
        
        # 타임스탑 계산
        time_stop = self.calculate_time_stop(half_life_avg)
        
        # 신호 생성 (s-score + 동적 임계값 기반)
        signal_type = "EXIT_OR_WAIT"
        direction = "대기"
        
        if anti_chop_ok and abs(current_s_score) >= dynamic_threshold:
            if current_s_score > 0:  # 스프레드가 평균 대비 높음 → 스프레드 숏
                signal_type = "ENTER_SHORT_SPREAD"
                direction = f"숏 스프레드 (숏 {asset2}, 롱 {asset1}×{pair_info['hedge_ratio']:.3f})"
            else:  # 스프레드가 평균 대비 낮음 → 스프레드 롱
                signal_type = "ENTER_LONG_SPREAD" 
                direction = f"롱 스프레드 (롱 {asset2}, 숏 {asset1}×{pair_info['hedge_ratio']:.3f})"
        
        return {
            'status': 'success',
            'pair': f"{asset1}-{asset2}",
            'signal_type': signal_type,
            'direction': direction,
            'current_zscore': current_s_score,  # s-score를 zscore로 반환
            's_score': current_s_score,
            'dynamic_threshold': dynamic_threshold,
            'kappa_avg': kappa_avg,
            'theta_avg': theta_avg,
            'sigma': sigma,
            'half_life_avg': half_life_avg,
            'anti_chop_passed': anti_chop_ok,
            'time_stop': time_stop,
            'hedge_ratio': pair_info['hedge_ratio'],
            'cost_ratio': cost_ratio,
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
                
            current_s = abs(signal_result['current_zscore'])  # 실제로는 s-score
            dynamic_threshold = signal_result['dynamic_threshold']
            anti_chop_passed = signal_result['anti_chop_passed']
            
            # 진입 신호 (|s-score| >= dynamic_threshold & anti-chop 통과)
            if current_s >= dynamic_threshold and anti_chop_passed:
                enter_signals.append(signal_result)
            # 관찰 대상 (임계값의 80% 이상 & anti-chop 실패 또는 임계값 미달)
            elif current_s >= dynamic_threshold * 0.8:
                watch_signals.append(signal_result)
        
        # s-score 크기 순으로 정렬 (높은 신호 강도 우선)
        enter_signals.sort(key=lambda x: abs(x['current_zscore']), reverse=True)
        watch_signals.sort(key=lambda x: abs(x['current_zscore']), reverse=True)
        
        return enter_signals[:n_pairs], watch_signals[:n_pairs]

def main():
    """
    OU 평균회귀 속도 기반 페어트레이딩 실행 예제
    """
    # 데이터 로딩
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    file_path = os.path.join(project_root, "data", "MU Price(BBG).csv")
    prices = load_data(file_path)
    
    # 개선된 OU 평균회귀 기반 페어트레이딩 객체 생성
    ou_trader = OUMeanReversionPairTrading(
        formation_window=252,           # 1년 
        rolling_window=60,              # 60일 롤링 윈도우 (권장)
        base_threshold=1.25,            # 기본 s-score 임계값
        exit_threshold=0.5,
        stop_loss=3.0,
        min_half_life=5,               # 5~60일 반감기
        max_half_life=60,
        min_cost_ratio=5.0,            # 비용 여유 ≥ 5
        min_mean_reversion_speed=0.01, # κ ≥ 0.01
        max_kappa_cv=0.6,              # κ 변동계수 ≤ 0.6
        data_coverage_threshold=0.9,    # 데이터 커버리지 ≥ 90%
        winsorize_percentile=0.01       # 상하 1% 윈저라이즈
    )
    
    # 페어 스크리닝
    enter_list, watch_list = ou_trader.screen_pairs(prices, n_pairs=10)
    
    print("=" * 75)
    print("개선된 OU 평균회귀 속도 기반 페어트레이딩 신호 (s-score)")
    print("=" * 75)
    
    print(f"\n[ENTRY SIGNALS] 진입 신호 ({len(enter_list)}개):")
    print("-" * 65)
    for i, signal in enumerate(enter_list, 1):
        print(f"{i:2d}. {signal['pair']:20s} | {signal['direction']}")
        print(f"     s-Score: {signal['current_zscore']:6.2f} | 동적임계값: {signal['dynamic_threshold']:6.2f}")
        print(f"     κ (평균회귀속도): {signal['kappa_avg']:6.4f} | Half-Life: {signal['half_life_avg']:4.1f}D")
        print(f"     θ (장기평균): {signal['theta_avg']:6.3f} | σ (변동성): {signal['sigma']:6.3f}")
        print(f"     Anti-chop 통과: {signal['anti_chop_passed']} | 비용비율: {signal['cost_ratio']:5.1f}")
        print(f"     타임스탑: {signal['time_stop']:2d}일 | 헤지비율: {signal['hedge_ratio']:6.3f}")
        print()
    
    print(f"\n[WATCH LIST] 관찰 대상 ({len(watch_list)}개):")
    print("-" * 65)
    for i, signal in enumerate(watch_list, 1):
        print(f"{i:2d}. {signal['pair']:20s} | s-Score: {signal['current_zscore']:6.2f}")
        print(f"     κ: {signal['kappa_avg']:6.4f} | Half-Life: {signal['half_life_avg']:4.1f}D")
        print(f"     동적임계값: {signal['dynamic_threshold']:6.2f} | Anti-chop: {signal['anti_chop_passed']}")
        print(f"     타임스탑: {signal['time_stop']:2d}일 | 비용비율: {signal['cost_ratio']:5.1f}")
        print()

if __name__ == "__main__":
    main()