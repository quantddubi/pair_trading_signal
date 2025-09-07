"""
1) 거리(유클리드) 기반 페어트레이딩 - 가장 단순하고 빠른 벤치마크
핵심: 표준화 가격 경로가 가장 비슷한 쌍을 후보로 선정
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
    utils_path = os.path.join(os.path.dirname(current_dir), 'utils', 'common_utils.py')
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
euclidean_distance_matrix = common_utils.euclidean_distance_matrix
get_non_overlapping_pairs = common_utils.get_non_overlapping_pairs
calculate_transaction_cost_ratio = common_utils.calculate_transaction_cost_ratio

class EuclideanDistancePairTrading:
    def __init__(self, formation_window: int = 756, signal_window: int = 60,
                 enter_threshold: float = 2.0, exit_threshold: float = 0.5,
                 stop_loss: float = 3.0, min_half_life: int = 5, max_half_life: int = 60,
                 min_cost_ratio: float = 5.0, transaction_cost: float = 0.0001):
        """
        거리 기반 페어트레이딩 파라미터
        
        Args:
            formation_window: 페어 선정 기간 (영업일, 기본값: 756일 = 3년)
            signal_window: z-스코어 계산 롤링 윈도우
            enter_threshold: 진입 z-스코어 임계값
            exit_threshold: 청산 z-스코어 임계값 (사용안함)
            stop_loss: 손절 z-스코어 임계값 (사용안함)
            min_half_life: 최소 반감기 (영업일)
            max_half_life: 최대 반감기 (영업일)
            min_cost_ratio: 최소 1σ/거래비용 비율
            transaction_cost: 거래비용 (1bp = 0.0001)
        """
        self.formation_window = formation_window
        self.signal_window = signal_window
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss = stop_loss
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.min_cost_ratio = min_cost_ratio
        self.transaction_cost = transaction_cost
        
    def select_pairs(self, prices: pd.DataFrame, n_pairs: int = 20) -> List[Dict]:
        """
        거리 기반 페어 선정
        
        Args:
            prices: 가격 데이터 (최근 formation_window 일)
            n_pairs: 선정할 페어 개수
            
        Returns:
            선정된 페어 정보 리스트
        """
        # 최근 formation_window 기간 데이터 추출
        formation_data = prices.tail(self.formation_window)
        
        # 결측치가 많은 자산 제외 (80% 이상 데이터 있어야 함)
        valid_assets = []
        for col in formation_data.columns:
            if formation_data[col].notna().sum() >= self.formation_window * 0.8:
                valid_assets.append(col)
        
        if len(valid_assets) < 2:
            return []
            
        formation_data = formation_data[valid_assets].fillna(method='ffill')
        
        # 가격 표준화 (리베이스: 첫날 = 1)
        normalized_prices = normalize_prices(formation_data, method='rebase')
        
        # 유클리드 거리 행렬 계산
        distance_matrix = euclidean_distance_matrix(normalized_prices)
        
        # 모든 가능한 페어 생성 (중복 허용)
        all_pairs = []
        n_assets = len(valid_assets)
        
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                distance = distance_matrix[i][j]
                all_pairs.append((valid_assets[i], valid_assets[j], distance))
        
        # 거리 기준 오름차순 정렬 (가까운 거리부터)
        selected_pairs = sorted(all_pairs, key=lambda x: x[2])[:n_pairs * 3]  # 필터링 고려해서 여유롭게
        
        # 페어 품질 필터링
        qualified_pairs = []
        for asset1, asset2, distance in selected_pairs:
            if len(qualified_pairs) >= n_pairs:
                break
                
            # 스프레드 계산 (1:1 비율)
            spread = calculate_spread(
                normalized_prices[asset1], 
                normalized_prices[asset2], 
                hedge_ratio=1.0
            )
            
            # Half-life 계산
            half_life = calculate_half_life(spread)
            
            # 거래비용 대비 수익성 계산 (1bp = 0.0001)
            spread_std = spread.std()
            cost_ratio = spread_std / self.transaction_cost if self.transaction_cost > 0 else float('inf')
            
            # 필터링 조건 (Cost Ratio 조건 제거, Half-Life만 검사)
            if (self.min_half_life <= half_life <= self.max_half_life):
                
                qualified_pairs.append({
                    'asset1': asset1,
                    'asset2': asset2,
                    'distance': distance,
                    'half_life': half_life,
                    'cost_ratio': cost_ratio,
                    'hedge_ratio': 1.0,  # 거리 기반은 1:1 비율
                    'method': 'euclidean_distance'
                })
        
        return qualified_pairs
    
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
        
        # 페어 선정 기간과 동일한 데이터 확보 (신호 생성용)
        recent_data = prices[[asset1, asset2]].tail(self.formation_window).fillna(method='ffill')
        
        if len(recent_data) < self.formation_window * 0.8:  # 80% 이상 데이터 필요
            return {'status': 'insufficient_data'}
        
        # 표준화
        normalized_recent = normalize_prices(recent_data, method='rebase')
        
        # 스프레드 계산
        spread = calculate_spread(
            normalized_recent[asset1],
            normalized_recent[asset2],
            hedge_ratio=pair_info['hedge_ratio']
        )
        
        # Z-스코어 계산 (적절한 롤링 윈도우 사용)
        zscore_window = min(self.signal_window, len(spread)//4, 60)  # 최대 60일 윈도우
        zscore = calculate_zscore(spread, window=zscore_window)
        
        # 현재 z-스코어
        current_zscore = zscore.iloc[-1] if not zscore.empty else 0
        
        # 진입 신호만 생성 (청산/손절 신호 제거)
        if abs(current_zscore) >= self.enter_threshold:
            if current_zscore > 0:
                signal_type = "ENTER_LONG"  # asset1 롱, asset2 숏
                direction = f"Long {asset1}, Short {asset2}"
            else:
                signal_type = "ENTER_SHORT"  # asset1 숏, asset2 롱  
                direction = f"Short {asset1}, Long {asset2}"
        else:
            signal_type = "WATCH"
            direction = "Watch for Entry"
        
        return {
            'status': 'success',
            'pair': f"{asset1}-{asset2}",
            'signal_type': signal_type,
            'direction': direction,
            'current_zscore': current_zscore,
            'distance_rank': pair_info['distance'],
            'half_life': pair_info['half_life'],
            'cost_ratio': pair_info['cost_ratio'],
            'method': 'euclidean_distance'
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
        
        # z-스코어 절댓값 기준으로 정렬
        enter_signals.sort(key=lambda x: abs(x['current_zscore']), reverse=True)
        watch_signals.sort(key=lambda x: abs(x['current_zscore']), reverse=True)
        
        return enter_signals[:n_pairs], watch_signals[:n_pairs]

def main():
    """
    거리 기반 페어트레이딩 실행 예제
    """
    # 데이터 로딩
    file_path = "/Users/a/PycharmProjects/pair_trading_signal/data/MU Price(BBG).csv"
    prices = load_data(file_path)
    
    # 거리 기반 페어트레이딩 객체 생성
    euclidean_trader = EuclideanDistancePairTrading(
        formation_window=756,     # 3년 (252 * 3)
        signal_window=60,         # 3개월
        enter_threshold=2.0,
        exit_threshold=0.5,       # 사용안함
        stop_loss=3.0,            # 사용안함
        min_half_life=5,
        max_half_life=60,
        min_cost_ratio=5.0,       # 1σ/거래비용 >= 5배
        transaction_cost=0.0001   # 1bp = 0.01%
    )
    
    # 페어 스크리닝
    enter_list, watch_list = euclidean_trader.screen_pairs(prices, n_pairs=10)
    
    print("=" * 60)
    print("거리 기반 페어트레이딩 신호 (Euclidean Distance)")
    print("=" * 60)
    
    print(f"\n📈 진입 신호 ({len(enter_list)}개):")
    print("-" * 50)
    for i, signal in enumerate(enter_list, 1):
        print(f"{i:2d}. {signal['pair']:20s} | {signal['direction']:25s}")
        print(f"     Z-Score: {signal['current_zscore']:6.2f} | Half-Life: {signal['half_life']:4.1f}D")
        print(f"     Cost Ratio: {signal['cost_ratio']:5.1f} | Distance: {signal['distance_rank']:.3f}")
        print()
    
    print(f"\n👀 관찰 대상 ({len(watch_list)}개):")
    print("-" * 50)
    for i, signal in enumerate(watch_list, 1):
        print(f"{i:2d}. {signal['pair']:20s} | Z-Score: {signal['current_zscore']:6.2f}")
        print(f"     Half-Life: {signal['half_life']:4.1f}D | Cost Ratio: {signal['cost_ratio']:5.1f}")
        print()

if __name__ == "__main__":
    main()