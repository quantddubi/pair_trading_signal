"""
2) 공적분(Engle-Granger) 기반 페어트레이딩 - 장기 균형이 있는 페어
핵심: 두 가격 간 장기 균형(공적분)이 있으면 잔차가 정상적(평균회귀)일 가능성↑
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
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
calculate_hedge_ratio_ols = common_utils.calculate_hedge_ratio_ols
calculate_transaction_cost_ratio = common_utils.calculate_transaction_cost_ratio
class CointegrationPairTrading:
    def __init__(self, formation_window: int = 252, signal_window: int = 60,
                 enter_threshold: float = 2.0, exit_threshold: float = 0.5,
                 stop_loss: float = 3.0, min_half_life: int = 5, max_half_life: int = 60,
                 min_cost_ratio: float = 5.0, max_pvalue: float = 0.05):
        """
        공적분 기반 페어트레이딩 파라미터
        
        Args:
            formation_window: 페어 선정 기간 (영업일)
            signal_window: z-스코어 계산 롤링 윈도우  
            enter_threshold: 진입 z-스코어 임계값
            exit_threshold: 청산 z-스코어 임계값
            stop_loss: 손절 z-스코어 임계값
            min_half_life: 최소 반감기 (영업일)
            max_half_life: 최대 반감기 (영업일)
            min_cost_ratio: 최소 1σ/거래비용 비율
            max_pvalue: 최대 ADF 테스트 p값 (공적분 유의성)
        """
        self.formation_window = formation_window
        self.signal_window = signal_window
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss = stop_loss
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.min_cost_ratio = min_cost_ratio
        self.max_pvalue = max_pvalue
        
    def test_cointegration(self, price1: pd.Series, price2: pd.Series) -> Tuple[float, float, pd.Series]:
        """
        Engle-Granger 공적분 테스트
        
        Args:
            price1: 첫 번째 자산 가격
            price2: 두 번째 자산 가격
            
        Returns:
            (hedge_ratio, p_value, residuals): 헤지비율, ADF p값, 잔차
        """
        try:
            # 공통 인덱스 확보
            common_idx = price1.index.intersection(price2.index)
            if len(common_idx) < 30:
                return 1.0, 1.0, pd.Series()
            
            p1_common = price1[common_idx].dropna()
            p2_common = price2[common_idx].dropna()
            
            if len(p1_common) != len(p2_common) or len(p1_common) < 30:
                return 1.0, 1.0, pd.Series()
            
            # OLS 회귀: price1 = α + β * price2 + ε
            X = np.column_stack([np.ones(len(p2_common)), p2_common.values])
            y = p1_common.values
            
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            hedge_ratio = beta[1]
            
            # 잔차 계산
            residuals = y - (beta[0] + beta[1] * p2_common.values)
            residuals_series = pd.Series(residuals, index=p1_common.index)
            
            # ADF 테스트 (잔차의 정상성 검정)
            adf_result = adfuller(residuals, autolag='AIC', maxlag=int(len(residuals)/4))
            p_value = adf_result[1]
            
            return hedge_ratio, p_value, residuals_series
            
        except Exception as e:
            return 1.0, 1.0, pd.Series()
    
    def select_pairs(self, prices: pd.DataFrame, n_pairs: int = 20) -> List[Dict]:
        """
        공적분 기반 페어 선정
        
        Args:
            prices: 가격 데이터 (최근 formation_window 일)
            n_pairs: 선정할 페어 개수
            
        Returns:
            선정된 페어 정보 리스트
        """
        # 최근 formation_window 기간 데이터 추출
        formation_data = prices.tail(self.formation_window)
        
        # 결측치가 많은 자산 제외 (90% 이상 데이터 있어야 함)
        valid_assets = []
        for col in formation_data.columns:
            if formation_data[col].notna().sum() >= self.formation_window * 0.9:
                valid_assets.append(col)
        
        if len(valid_assets) < 2:
            return []
            
        formation_data = formation_data[valid_assets].fillna(method='ffill')
        
        # 자산 수가 너무 많으면 샘플링
        if len(valid_assets) > 30:
            # 변동성이 높은 상위 30개 자산만 선택
            volatilities = {}
            for asset in valid_assets:
                returns = formation_data[asset].pct_change().dropna()
                if len(returns) > 30:
                    volatilities[asset] = returns.std()
            
            sorted_assets = sorted(volatilities.keys(), key=lambda x: volatilities[x], reverse=True)
            valid_assets = sorted_assets[:30]
        
        # 모든 자산 쌍에 대해 공적분 테스트
        cointegration_results = []
        total_pairs = len(valid_assets) * (len(valid_assets) - 1) // 2
        
        print(f"공적분 분석 대상: {len(valid_assets)}개 자산, {total_pairs}개 페어")
        
        for i, asset1 in enumerate(valid_assets):
            for j, asset2 in enumerate(valid_assets):
                if i >= j:  # 중복 방지
                    continue
                    
                # 공적분 테스트
                hedge_ratio, p_value, residuals = self.test_cointegration(
                    formation_data[asset1], formation_data[asset2]
                )
                
                # 유의한 공적분 관계가 있는 경우만
                if p_value <= self.max_pvalue and len(residuals) > 0:
                    
                    # Half-life 계산
                    half_life = calculate_half_life(residuals)
                    
                    # 거래비용 대비 수익성 계산
                    cost_ratio = calculate_transaction_cost_ratio(residuals)
                    
                    # 품질 필터
                    if (self.min_half_life <= half_life <= self.max_half_life and 
                        cost_ratio >= self.min_cost_ratio):
                        
                        cointegration_results.append({
                            'asset1': asset1,
                            'asset2': asset2,
                            'hedge_ratio': hedge_ratio,
                            'p_value': p_value,
                            'half_life': half_life,
                            'cost_ratio': cost_ratio,
                            'residuals': residuals,
                            'method': 'cointegration'
                        })
        
        # p값이 낮은 순으로 정렬 (공적분 관계가 강한 순)
        cointegration_results.sort(key=lambda x: x['p_value'])
        
        # 중복 없는 페어 선정
        selected_pairs = []
        used_assets = set()
        
        for result in cointegration_results:
            if len(selected_pairs) >= n_pairs:
                break
                
            asset1, asset2 = result['asset1'], result['asset2']
            if asset1 not in used_assets and asset2 not in used_assets:
                selected_pairs.append(result)
                used_assets.add(asset1)
                used_assets.add(asset2)
        
        return selected_pairs
    
    def check_cointegration_stability(self, prices: pd.DataFrame, pair_info: Dict) -> float:
        """
        최근 3개월 공적분 관계 안정성 재검정
        
        Returns:
            recent_p_value: 최근 기간 ADF p값
        """
        asset1, asset2 = pair_info['asset1'], pair_info['asset2']
        
        # 최근 3개월(60일) 데이터
        recent_data = prices[[asset1, asset2]].tail(60).fillna(method='ffill')
        
        if len(recent_data) < 30:
            return 1.0  # 불안정으로 간주
            
        _, recent_p_value, _ = self.test_cointegration(
            recent_data[asset1], recent_data[asset2]
        )
        
        return recent_p_value
    
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
        
        # 최근 공적분 안정성 체크
        recent_p_value = self.check_cointegration_stability(prices, pair_info)
        if recent_p_value > self.max_pvalue:
            return {'status': 'cointegration_broken', 'recent_p_value': recent_p_value}
        
        # 최근 데이터 확보
        recent_data = prices[[asset1, asset2]].tail(self.signal_window * 2).fillna(method='ffill')
        
        if len(recent_data) < self.signal_window:
            return {'status': 'insufficient_data'}
        
        # 스프레드 계산 (공적분 헤지비율 사용)
        spread = calculate_spread(
            recent_data[asset1],
            recent_data[asset2],
            hedge_ratio=pair_info['hedge_ratio']
        )
        
        # Z-스코어 계산
        zscore = calculate_zscore(spread, window=self.signal_window)
        
        # 현재 z-스코어
        current_zscore = zscore.iloc[-1] if not zscore.empty else 0
        
        # 신호 생성
        signals = generate_trading_signals(
            zscore, 
            enter_threshold=self.enter_threshold,
            exit_threshold=self.exit_threshold,
            stop_loss=self.stop_loss
        )
        
        current_signal = signals.iloc[-1] if not signals.empty else 0
        
        # 신호 해석
        if current_signal == 1:
            signal_type = "ENTER_LONG"  # asset1 롱, asset2 숏
            direction = f"Long {asset1}, Short {asset2} (ratio: {pair_info['hedge_ratio']:.3f})"
        elif current_signal == -1:
            signal_type = "ENTER_SHORT"  # asset1 숏, asset2 롱
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
            'p_value': pair_info['p_value'],
            'recent_p_value': recent_p_value,
            'hedge_ratio': pair_info['hedge_ratio'],
            'half_life': pair_info['half_life'],
            'cost_ratio': pair_info['cost_ratio'],
            'method': 'cointegration'
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
        
        # p값이 낮은 순으로 정렬 (공적분 관계가 강한 순)
        enter_signals.sort(key=lambda x: x['p_value'])
        watch_signals.sort(key=lambda x: x['p_value'])
        
        return enter_signals[:n_pairs], watch_signals[:n_pairs]

def main():
    """
    공적분 기반 페어트레이딩 실행 예제
    """
    # 데이터 로딩
    file_path = "/Users/a/PycharmProjects/pair_trading_signal/data/MU Price(BBG).csv"
    prices = load_data(file_path)
    
    # 공적분 기반 페어트레이딩 객체 생성
    cointegration_trader = CointegrationPairTrading(
        formation_window=252,   # 1년
        signal_window=60,       # 3개월
        enter_threshold=2.0,
        exit_threshold=0.5,
        stop_loss=3.0,
        min_half_life=5,
        max_half_life=60,
        min_cost_ratio=5.0,
        max_pvalue=0.05         # 5% 유의수준
    )
    
    # 페어 스크리닝
    enter_list, watch_list = cointegration_trader.screen_pairs(prices, n_pairs=10)
    
    print("=" * 65)
    print("공적분 기반 페어트레이딩 신호 (Engle-Granger)")
    print("=" * 65)
    
    print(f"\n📈 진입 신호 ({len(enter_list)}개):")
    print("-" * 55)
    for i, signal in enumerate(enter_list, 1):
        print(f"{i:2d}. {signal['pair']:20s} | {signal['direction']:30s}")
        print(f"     Z-Score: {signal['current_zscore']:6.2f} | Half-Life: {signal['half_life']:4.1f}D")
        print(f"     P-Value: {signal['p_value']:.4f} | Recent P-Value: {signal['recent_p_value']:.4f}")
        print(f"     Cost Ratio: {signal['cost_ratio']:5.1f} | Hedge Ratio: {signal['hedge_ratio']:6.3f}")
        print()
    
    print(f"\n👀 관찰 대상 ({len(watch_list)}개):")
    print("-" * 55)
    for i, signal in enumerate(watch_list, 1):
        print(f"{i:2d}. {signal['pair']:20s} | Z-Score: {signal['current_zscore']:6.2f}")
        print(f"     P-Value: {signal['p_value']:.4f} | Half-Life: {signal['half_life']:4.1f}D")
        print(f"     Cost Ratio: {signal['cost_ratio']:5.1f} | Hedge Ratio: {signal['hedge_ratio']:6.3f}")
        print()

if __name__ == "__main__":
    main()