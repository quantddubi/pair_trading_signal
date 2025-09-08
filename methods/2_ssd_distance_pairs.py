"""
8) SSD(Sum of Squared Deviations) 거리 기반 페어트레이딩
핵심: 12개월 누적수익률 정규화 후 제곱편차 합이 최소인 페어 선정
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
calculate_transaction_cost_ratio = common_utils.calculate_transaction_cost_ratio

class SSDDistancePairTrading:
    def __init__(self, formation_window: int = 252, signal_window: int = 60,
                 enter_threshold: float = 2.0, exit_threshold: float = 0.5,
                 stop_loss: float = 3.0, min_half_life: int = 5, max_half_life: int = 60,
                 min_cost_ratio: float = 5.0, transaction_cost: float = 0.0001):
        """
        SSD 거리 기반 페어트레이딩 파라미터
        
        Args:
            formation_window: 페어 선정 기간 (영업일, 기본값: 252일 = 12개월)
            signal_window: z-스코어 계산 롤링 윈도우
            enter_threshold: 진입 z-스코어 임계값 (2σ)
            exit_threshold: 청산 z-스코어 임계값
            stop_loss: 손절 z-스코어 임계값
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
        
    def calculate_cumulative_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        12개월 누적수익률 계산 (배당재투자 가정)
        
        Args:
            prices: 가격 데이터
            
        Returns:
            누적수익률 인덱스 (첫날 = 1)
        """
        # 숫자형 데이터만 선택
        numeric_prices = prices.select_dtypes(include=[np.number])
        
        # 일간 수익률 계산
        returns = numeric_prices.pct_change().fillna(0)
        
        # 누적수익률 인덱스 계산 (1부터 시작)
        cumulative_returns = (1 + returns).cumprod()
        
        return cumulative_returns
    
    def calculate_ssd(self, series1: pd.Series, series2: pd.Series) -> float:
        """
        정규화된 두 시계열 간 제곱편차 합(SSD) 계산
        
        Args:
            series1: 첫 번째 자산의 정규화된 가격
            series2: 두 번째 자산의 정규화된 가격
            
        Returns:
            SSD 값
        """
        try:
            # 숫자형으로 변환
            s1_numeric = pd.to_numeric(series1, errors='coerce')
            s2_numeric = pd.to_numeric(series2, errors='coerce')
            
            # 공통 인덱스 확보
            common_idx = s1_numeric.index.intersection(s2_numeric.index)
            if len(common_idx) < self.formation_window * 0.8:
                return float('inf')
            
            s1_common = s1_numeric[common_idx].dropna()
            s2_common = s2_numeric[common_idx].dropna()
            
            # 길이가 다르면 짧은 쪽에 맞춤
            min_len = min(len(s1_common), len(s2_common))
            if min_len < self.formation_window * 0.8:
                return float('inf')
            
            s1_common = s1_common.iloc[:min_len]
            s2_common = s2_common.iloc[:min_len]
            
            # 정규화 (첫날 = 1)
            if s1_common.iloc[0] != 0 and s2_common.iloc[0] != 0:
                s1_normalized = s1_common / s1_common.iloc[0]
                s2_normalized = s2_common / s2_common.iloc[0]
            else:
                return float('inf')
            
            # SSD 계산
            ssd = np.sum((s1_normalized - s2_normalized) ** 2)
            
            return ssd if not np.isnan(ssd) else float('inf')
            
        except Exception as e:
            return float('inf')
    
    def find_minimum_ssd_pairs(self, cumulative_returns: pd.DataFrame) -> List[Tuple[str, str, float]]:
        """
        각 종목에 대해 SSD가 최소가 되는 페어 찾기
        
        Args:
            cumulative_returns: 누적수익률 데이터
            
        Returns:
            (asset1, asset2, ssd) 튜플 리스트
        """
        assets = cumulative_returns.columns.tolist()
        n_assets = len(assets)
        
        # SSD 행렬 계산
        ssd_matrix = np.full((n_assets, n_assets), float('inf'))
        
        for i in range(n_assets):
            for j in range(n_assets):
                if i != j:
                    ssd = self.calculate_ssd(
                        cumulative_returns.iloc[:, i],
                        cumulative_returns.iloc[:, j]
                    )
                    ssd_matrix[i][j] = ssd
        
        # 각 자산에 대해 최소 SSD 페어 찾기
        pairs_dict = {}
        for i in range(n_assets):
            if np.all(np.isinf(ssd_matrix[i])):
                continue
            min_j = np.argmin(ssd_matrix[i])
            min_ssd = ssd_matrix[i][min_j]
            
            if not np.isinf(min_ssd):
                asset1 = assets[i]
                asset2 = assets[min_j]
                # 중복 방지를 위해 정렬된 키 사용
                key = tuple(sorted([asset1, asset2]))
                if key not in pairs_dict or min_ssd < pairs_dict[key][2]:
                    pairs_dict[key] = (asset1, asset2, min_ssd)
        
        # 딕셔너리를 리스트로 변환
        return list(pairs_dict.values())
    
    def select_pairs(self, prices: pd.DataFrame, n_pairs: int = 20) -> List[Dict]:
        """
        SSD 기반 페어 선정
        
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
        
        # 문자열 컬럼을 숫자형으로 변환 시도
        for col in valid_assets:
            formation_data[col] = pd.to_numeric(formation_data[col], errors='coerce')
            
        formation_data = formation_data[valid_assets].fillna(method='ffill')
        
        # 12개월 누적수익률 계산
        cumulative_returns = self.calculate_cumulative_returns(formation_data)
        
        # SSD 최소 페어 찾기
        ssd_pairs = self.find_minimum_ssd_pairs(cumulative_returns)
        
        # SSD 기준 오름차순 정렬 (가까운 거리부터)
        ssd_pairs.sort(key=lambda x: x[2])
        
        # 페어 품질 필터링
        qualified_pairs = []
        for asset1, asset2, ssd in ssd_pairs[:n_pairs * 3]:  # 필터링 고려해서 여유롭게
            if len(qualified_pairs) >= n_pairs:
                break
                
            # 정규화된 가격으로 스프레드 계산
            normalized_data = normalize_prices(formation_data[[asset1, asset2]], method='rebase')
            spread = calculate_spread(
                normalized_data[asset1], 
                normalized_data[asset2], 
                hedge_ratio=1.0
            )
            
            # 형성기간 동안의 스프레드 표준편차 계산
            spread_std = spread.std()
            
            # Half-life 계산
            half_life = calculate_half_life(spread)
            
            # 거래비용 대비 수익성 계산
            cost_ratio = spread_std / self.transaction_cost if self.transaction_cost > 0 else float('inf')
            
            # 필터링 조건
            if (self.min_half_life <= half_life <= self.max_half_life):
                qualified_pairs.append({
                    'asset1': asset1,
                    'asset2': asset2,
                    'ssd': ssd,
                    'half_life': half_life,
                    'cost_ratio': cost_ratio,
                    'hedge_ratio': 1.0,  # SSD 기반은 1:1 비율
                    'formation_std': spread_std,  # 형성기간 표준편차 저장
                    'method': 'ssd_distance'
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
        
        # 페어 선정 기간과 동일한 데이터 확보
        recent_data = prices[[asset1, asset2]].tail(self.formation_window).fillna(method='ffill')
        
        # 숫자형으로 변환
        for col in [asset1, asset2]:
            recent_data[col] = pd.to_numeric(recent_data[col], errors='coerce')
        
        recent_data = recent_data.dropna()
        
        if len(recent_data) < self.formation_window * 0.8:
            return {'status': 'insufficient_data'}
        
        # 누적수익률 기반 정규화
        cumulative_returns = self.calculate_cumulative_returns(recent_data)
        
        # 스프레드 계산 (누적수익률 차이)
        spread = cumulative_returns[asset1] - cumulative_returns[asset2]
        
        # 현재 스프레드와 형성기간 통계 비교
        current_spread = spread.iloc[-1]
        spread_mean = spread.mean()
        
        # 형성기간 표준편차 사용 (트리거 조건)
        formation_std = pair_info.get('formation_std', spread.std())
        
        # 현재 편차 (표준편차 단위)
        current_deviation = (current_spread - spread_mean) / formation_std if formation_std > 0 else 0
        
        # 트리거 조건: 2σ 이상 벌어졌을 때
        if abs(current_deviation) >= self.enter_threshold:
            if current_deviation > 0:
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
            'current_deviation': current_deviation,  # σ 단위
            'ssd_distance': pair_info['ssd'],
            'half_life': pair_info['half_life'],
            'cost_ratio': pair_info['cost_ratio'],
            'method': 'ssd_distance'
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
                
            current_dev = abs(signal_result['current_deviation'])
            
            # 진입 신호 (|deviation| >= 2.0σ)
            if current_dev >= self.enter_threshold:
                enter_signals.append(signal_result)
            # 관찰 대상 (1.5σ <= |deviation| < 2.0σ)
            elif 1.5 <= current_dev < self.enter_threshold:
                watch_signals.append(signal_result)
        
        # 편차 절댓값 기준으로 정렬
        enter_signals.sort(key=lambda x: abs(x['current_deviation']), reverse=True)
        watch_signals.sort(key=lambda x: abs(x['current_deviation']), reverse=True)
        
        return enter_signals[:n_pairs], watch_signals[:n_pairs]

def main():
    """
    SSD 거리 기반 페어트레이딩 실행 예제
    """
    # 데이터 로딩
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    file_path = os.path.join(project_root, "data/MU Price(BBG).csv")
    prices = load_data(file_path)
    
    # SSD 거리 기반 페어트레이딩 객체 생성
    ssd_trader = SSDDistancePairTrading(
        formation_window=252,     # 12개월 (1년)
        signal_window=60,         # 3개월
        enter_threshold=2.0,      # 2σ 트리거
        exit_threshold=0.5,       
        stop_loss=3.0,            
        min_half_life=5,
        max_half_life=60,
        min_cost_ratio=5.0,       
        transaction_cost=0.0001   # 1bp = 0.01%
    )
    
    # 페어 스크리닝
    enter_list, watch_list = ssd_trader.screen_pairs(prices, n_pairs=10)
    
    print("=" * 60)
    print("SSD 거리 기반 페어트레이딩 신호 (Sum of Squared Deviations)")
    print("=" * 60)
    
    print(f"\n📈 진입 신호 ({len(enter_list)}개):")
    print("-" * 50)
    for i, signal in enumerate(enter_list, 1):
        print(f"{i:2d}. {signal['pair']:20s} | {signal['direction']:25s}")
        print(f"     Deviation: {signal['current_deviation']:6.2f}σ | Half-Life: {signal['half_life']:4.1f}D")
        print(f"     Cost Ratio: {signal['cost_ratio']:5.1f} | SSD: {signal['ssd_distance']:.3f}")
        print()
    
    print(f"\n👀 관찰 대상 ({len(watch_list)}개):")
    print("-" * 50)
    for i, signal in enumerate(watch_list, 1):
        print(f"{i:2d}. {signal['pair']:20s} | Deviation: {signal['current_deviation']:6.2f}σ")
        print(f"     Half-Life: {signal['half_life']:4.1f}D | Cost Ratio: {signal['cost_ratio']:5.1f}")
        print()

if __name__ == "__main__":
    main()