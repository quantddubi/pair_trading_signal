"""
3) 롤링 상관 "레짐 전환" 스크리닝 - 합·분 현상 포착
핵심: 장기 상관 대비 단기 상관 변화(Δcorr)가 큰 쌍을 찾아 레짐 전환 신호로 활용
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
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
calculate_correlation_regime_change = common_utils.calculate_correlation_regime_change

class CorrelationRegimePairTrading:
    def __init__(self, formation_window: int = 252, signal_window: int = 60,
                 long_corr_window: int = 252, short_corr_window: int = 60,
                 enter_threshold: float = 2.0, exit_threshold: float = 0.5,
                 stop_loss: float = 3.0, min_half_life: int = 5, max_half_life: int = 60,
                 min_cost_ratio: float = 5.0, min_delta_corr: float = 0.3):
        """
        상관관계 레짐 전환 기반 페어트레이딩 파라미터
        
        Args:
            formation_window: 페어 선정 기간 (영업일)
            signal_window: z-스코어 계산 롤링 윈도우
            long_corr_window: 장기 상관관계 계산 윈도우 (12개월)
            short_corr_window: 단기 상관관계 계산 윈도우 (3개월)
            enter_threshold: 진입 z-스코어 임계값
            exit_threshold: 청산 z-스코어 임계값
            stop_loss: 손절 z-스코어 임계값
            min_half_life: 최소 반감기 (영업일)
            max_half_life: 최대 반감기 (영업일)
            min_cost_ratio: 최소 1σ/거래비용 비율
            min_delta_corr: 최소 상관관계 변화 임계값
        """
        self.formation_window = formation_window
        self.signal_window = signal_window
        self.long_corr_window = long_corr_window
        self.short_corr_window = short_corr_window
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss = stop_loss
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.min_cost_ratio = min_cost_ratio
        self.min_delta_corr = min_delta_corr
    
    def calculate_rolling_correlations(self, price1: pd.Series, price2: pd.Series) -> Tuple[float, float, float]:
        """
        장기 vs 단기 상관관계 및 변화량 계산
        
        Returns:
            (long_corr, short_corr, delta_corr): 장기 상관, 단기 상관, 변화량
        """
        # 수익률 계산
        returns1 = price1.pct_change().dropna()
        returns2 = price2.pct_change().dropna()
        
        # 공통 인덱스
        common_idx = returns1.index.intersection(returns2.index)
        if len(common_idx) < max(self.long_corr_window, self.short_corr_window):
            return 0, 0, 0
            
        returns1_common = returns1[common_idx]
        returns2_common = returns2[common_idx]
        
        # 장기 상관관계 (최근 12개월)
        if len(returns1_common) >= self.long_corr_window:
            long_corr = returns1_common.tail(self.long_corr_window).corr(
                returns2_common.tail(self.long_corr_window)
            )
        else:
            long_corr = returns1_common.corr(returns2_common)
        
        # 단기 상관관계 (최근 3개월)
        if len(returns1_common) >= self.short_corr_window:
            short_corr = returns1_common.tail(self.short_corr_window).corr(
                returns2_common.tail(self.short_corr_window)
            )
        else:
            short_corr = returns1_common.corr(returns2_common)
        
        # 변화량
        delta_corr = abs(short_corr - long_corr) if not np.isnan(short_corr) and not np.isnan(long_corr) else 0
        
        return long_corr if not np.isnan(long_corr) else 0, \
               short_corr if not np.isnan(short_corr) else 0, \
               delta_corr
    
    def classify_regime_type(self, long_corr: float, short_corr: float) -> str:
        """
        레짐 전환 유형 분류
        
        Returns:
            regime_type: "convergence" (합류), "divergence" (분화), "stable" (안정)
        """
        if abs(short_corr - long_corr) < 0.1:
            return "stable"
        elif short_corr > long_corr:
            return "convergence"  # 최근 들어 더 높은 상관관계
        else:
            return "divergence"   # 최근 들어 낮은 상관관계
    
    def select_pairs_by_sector(self, prices: pd.DataFrame, sectors: Dict[str, List[str]] = None) -> List[str]:
        """
        섹터별 자산 그룹 내에서만 페어 후보 선정 (선택사항)
        
        Args:
            prices: 가격 데이터
            sectors: {'sector_name': [asset_list]} 형태의 섹터 분류
            
        Returns:
            같은 섹터 내 자산 쌍 리스트
        """
        if sectors is None:
            # 기본: 모든 자산
            return [(col1, col2) for i, col1 in enumerate(prices.columns) 
                    for col2 in prices.columns[i+1:]]
        
        sector_pairs = []
        for sector_name, asset_list in sectors.items():
            # 해당 섹터 자산들이 데이터에 있는지 확인
            available_assets = [asset for asset in asset_list if asset in prices.columns]
            if len(available_assets) >= 2:
                for i, asset1 in enumerate(available_assets):
                    for asset2 in available_assets[i+1:]:
                        sector_pairs.append((asset1, asset2))
        
        return sector_pairs if sector_pairs else [(col1, col2) for i, col1 in enumerate(prices.columns) 
                                                  for col2 in prices.columns[i+1:]]
    
    def select_pairs(self, prices: pd.DataFrame, n_pairs: int = 20, sectors: Dict[str, List[str]] = None) -> List[Dict]:
        """
        상관관계 레짐 전환 기반 페어 선정
        
        Args:
            prices: 가격 데이터
            n_pairs: 선정할 페어 개수
            sectors: 섹터별 자산 그룹 (선택)
            
        Returns:
            선정된 페어 정보 리스트
        """
        # 최근 formation_window 기간 데이터 추출
        formation_data = prices.tail(self.formation_window)
        
        # 결측치가 많은 자산 제외
        valid_assets = []
        for col in formation_data.columns:
            if formation_data[col].notna().sum() >= self.formation_window * 0.8:
                valid_assets.append(col)
        
        if len(valid_assets) < 2:
            return []
            
        formation_data = formation_data[valid_assets].fillna(method='ffill')
        
        # 섹터별 페어 후보 (또는 전체)
        pair_candidates = self.select_pairs_by_sector(formation_data, sectors)
        
        # 상관관계 레짐 변화 분석
        regime_results = []
        
        for asset1, asset2 in pair_candidates:
            if asset1 not in valid_assets or asset2 not in valid_assets:
                continue
                
            # 상관관계 변화 계산
            long_corr, short_corr, delta_corr = self.calculate_rolling_correlations(
                formation_data[asset1], formation_data[asset2]
            )
            
            # 최소 변화 임계값 필터
            if delta_corr < self.min_delta_corr:
                continue
            
            # 레짐 타입 분류
            regime_type = self.classify_regime_type(long_corr, short_corr)
            
            # 스프레드 계산 (1:1 비율로 시작)
            spread = calculate_spread(
                formation_data[asset1], 
                formation_data[asset2], 
                hedge_ratio=1.0
            )
            
            # Half-life 계산
            half_life = calculate_half_life(spread)
            
            # 거래비용 대비 수익성 계산
            cost_ratio = calculate_transaction_cost_ratio(spread)
            
            # 품질 필터
            if (self.min_half_life <= half_life <= self.max_half_life and 
                cost_ratio >= self.min_cost_ratio):
                
                regime_results.append({
                    'asset1': asset1,
                    'asset2': asset2,
                    'long_corr': long_corr,
                    'short_corr': short_corr,
                    'delta_corr': delta_corr,
                    'regime_type': regime_type,
                    'half_life': half_life,
                    'cost_ratio': cost_ratio,
                    'hedge_ratio': 1.0,
                    'method': 'correlation_regime'
                })
        
        # delta_corr이 큰 순으로 정렬
        regime_results.sort(key=lambda x: x['delta_corr'], reverse=True)
        
        # 중복 없는 페어 선정
        selected_pairs = []
        used_assets = set()
        
        for result in regime_results:
            if len(selected_pairs) >= n_pairs:
                break
                
            asset1, asset2 = result['asset1'], result['asset2']
            if asset1 not in used_assets and asset2 not in used_assets:
                selected_pairs.append(result)
                used_assets.add(asset1)
                used_assets.add(asset2)
        
        return selected_pairs
    
    def generate_signals(self, prices: pd.DataFrame, pair_info: Dict) -> Dict:
        """
        특정 페어에 대한 트레이딩 신호 생성
        - 레짐 전환 + z-스코어 동시 조건 확인
        
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
        
        # 현재 상관관계 레짐 상태 재확인
        current_long_corr, current_short_corr, current_delta_corr = self.calculate_rolling_correlations(
            recent_data[asset1], recent_data[asset2]
        )
        current_regime = self.classify_regime_type(current_long_corr, current_short_corr)
        
        # 신호 생성 (레짐 전환 + z-스코어 동시 조건)
        signals = generate_trading_signals(
            zscore, 
            enter_threshold=self.enter_threshold,
            exit_threshold=self.exit_threshold,
            stop_loss=self.stop_loss
        )
        
        current_signal = signals.iloc[-1] if not signals.empty else 0
        
        # 레짐 전환 상황에서만 진입 신호 유효
        regime_filter = current_delta_corr >= self.min_delta_corr
        
        if not regime_filter:
            current_signal = 0  # 레짐 전환이 없으면 대기
        
        # 신호 해석
        if current_signal == 1:
            signal_type = "ENTER_LONG"
            direction = f"Long {asset1}, Short {asset2} (Regime: {current_regime})"
        elif current_signal == -1:
            signal_type = "ENTER_SHORT"
            direction = f"Short {asset1}, Long {asset2} (Regime: {current_regime})"
        else:
            signal_type = "EXIT_OR_WAIT"
            direction = f"Exit or Wait (Regime: {current_regime})"
        
        return {
            'status': 'success',
            'pair': f"{asset1}-{asset2}",
            'signal_type': signal_type,
            'direction': direction,
            'current_zscore': current_zscore,
            'long_corr': pair_info['long_corr'],
            'short_corr': pair_info['short_corr'],
            'delta_corr': pair_info['delta_corr'],
            'current_delta_corr': current_delta_corr,
            'regime_type': pair_info['regime_type'],
            'current_regime': current_regime,
            'half_life': pair_info['half_life'],
            'cost_ratio': pair_info['cost_ratio'],
            'method': 'correlation_regime'
        }
    
    def screen_pairs(self, prices: pd.DataFrame, n_pairs: int = 10, 
                    sectors: Dict[str, List[str]] = None) -> Tuple[List[Dict], List[Dict]]:
        """
        전체 페어 스크리닝 및 신호 생성
        
        Returns:
            (enter_signals, watch_signals): 진입 신호와 관찰 대상 리스트
        """
        # 페어 선정
        selected_pairs = self.select_pairs(prices, n_pairs * 2, sectors)
        
        enter_signals = []
        watch_signals = []
        
        for pair_info in selected_pairs:
            signal_result = self.generate_signals(prices, pair_info)
            
            if signal_result['status'] != 'success':
                continue
                
            current_z = abs(signal_result['current_zscore'])
            
            # 진입 신호 (|z| >= 2.0 & 레짐 전환)
            if current_z >= self.enter_threshold and signal_result['signal_type'] != 'EXIT_OR_WAIT':
                enter_signals.append(signal_result)
            # 관찰 대상 (1.5 <= |z| < 2.0)
            elif 1.5 <= current_z < self.enter_threshold:
                watch_signals.append(signal_result)
        
        # delta_corr이 큰 순으로 정렬 (레짐 변화가 큰 순)
        enter_signals.sort(key=lambda x: x['current_delta_corr'], reverse=True)
        watch_signals.sort(key=lambda x: x['current_delta_corr'], reverse=True)
        
        return enter_signals[:n_pairs], watch_signals[:n_pairs]

def main():
    """
    상관관계 레짐 전환 기반 페어트레이딩 실행 예제
    """
    # 데이터 로딩
    file_path = "/Users/a/PycharmProjects/pair_trading_signal/data/MU Price(BBG).csv"
    prices = load_data(file_path)
    
    # 간단한 섹터 분류 예제 (실제로는 더 정교하게 분류)
    sample_sectors = {
        'US_Indices': ['SPX Index', 'NDX Index', 'INDU Index', 'RTY Index'],
        'EU_Indices': ['SX5E Index', 'DAX Index', 'CAC Index', 'UKX Index'],
        'Asian_Indices': ['NKY Index', 'HSI Index', 'KOSPI Index', 'TWSE Index'],
        'Bonds': ['TY1 Comdty', 'FV1 Comdty', 'TU1 Comdty', 'RX1 Comdty'],
        'FX': ['EURUSD Curncy', 'JPYUSD Curncy', 'GBPUSD Curncy', 'CHFUSD Curncy'],
        'Commodities': ['CL1 Comdty', 'GC1 Comdty', 'HG1 Comdty', 'NG1 Comdty']
    }
    
    # 상관관계 레짐 전환 기반 페어트레이딩 객체 생성
    regime_trader = CorrelationRegimePairTrading(
        formation_window=252,      # 1년
        signal_window=60,          # 3개월
        long_corr_window=252,      # 장기 상관관계: 12개월
        short_corr_window=60,      # 단기 상관관계: 3개월
        enter_threshold=2.0,
        exit_threshold=0.5,
        stop_loss=3.0,
        min_half_life=5,
        max_half_life=60,
        min_cost_ratio=5.0,
        min_delta_corr=0.3         # 30% 이상 상관관계 변화
    )
    
    # 페어 스크리닝 (섹터별)
    enter_list, watch_list = regime_trader.screen_pairs(prices, n_pairs=10, sectors=sample_sectors)
    
    print("=" * 70)
    print("상관관계 레짐 전환 기반 페어트레이딩 신호")
    print("=" * 70)
    
    print(f"\n📈 진입 신호 ({len(enter_list)}개):")
    print("-" * 60)
    for i, signal in enumerate(enter_list, 1):
        print(f"{i:2d}. {signal['pair']:20s} | {signal['direction']:35s}")
        print(f"     Z-Score: {signal['current_zscore']:6.2f} | Half-Life: {signal['half_life']:4.1f}D")
        print(f"     Long Corr: {signal['long_corr']:6.3f} | Short Corr: {signal['short_corr']:6.3f}")
        print(f"     Delta Corr: {signal['delta_corr']:6.3f} | Current ΔCorr: {signal['current_delta_corr']:6.3f}")
        print(f"     Cost Ratio: {signal['cost_ratio']:5.1f} | Regime: {signal['regime_type']}")
        print()
    
    print(f"\n👀 관찰 대상 ({len(watch_list)}개):")
    print("-" * 60)
    for i, signal in enumerate(watch_list, 1):
        print(f"{i:2d}. {signal['pair']:20s} | Z-Score: {signal['current_zscore']:6.2f}")
        print(f"     Delta Corr: {signal['current_delta_corr']:6.3f} | Regime: {signal['current_regime']}")
        print(f"     Cost Ratio: {signal['cost_ratio']:5.1f} | Half-Life: {signal['half_life']:4.1f}D")
        print()

if __name__ == "__main__":
    main()