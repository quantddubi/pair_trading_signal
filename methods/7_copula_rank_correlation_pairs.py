"""
6) 코퓰라·순위상관(간단 버전) 기반 - 비선형·꼬리의존 반영
핵심: 선형 상관이 아니어도 함께 움직이는 꼬리 동조를 포착
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.stats import kendalltau, spearmanr
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
calculate_rank_correlation = common_utils.calculate_rank_correlation
calculate_tail_dependence = common_utils.calculate_tail_dependence

class CopulaRankCorrelationPairTrading:
    def __init__(self, formation_window: int = 252, signal_window: int = 60,
                 long_window: int = 252, short_window: int = 60,
                 enter_threshold: float = 2.0, exit_threshold: float = 0.5,
                 stop_loss: float = 3.0, min_half_life: int = 5, max_half_life: int = 60,
                 min_cost_ratio: float = 5.0, min_rank_corr: float = 0.3,
                 min_rank_corr_change: float = 0.2, tail_quantile: float = 0.1):
        """
        코퓰라·순위상관 기반 페어트레이딩 파라미터
        
        Args:
            formation_window: 페어 선정 기간 (영업일)
            signal_window: z-스코어 계산 롤링 윈도우
            long_window: 장기 순위상관 계산 윈도우
            short_window: 단기 순위상관 계산 윈도우
            enter_threshold: 진입 z-스코어 임계값
            exit_threshold: 청산 z-스코어 임계값
            stop_loss: 손절 z-스코어 임계값
            min_half_life: 최소 반감기 (영업일)
            max_half_life: 최대 반감기 (영업일)
            min_cost_ratio: 최소 1σ/거래비용 비율
            min_rank_corr: 최소 장기 순위상관 임계값
            min_rank_corr_change: 최소 순위상관 변화 임계값
            tail_quantile: 꼬리 의존성 계산용 분위수 (상/하위 10%)
        """
        self.formation_window = formation_window
        self.signal_window = signal_window
        self.long_window = long_window
        self.short_window = short_window
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss = stop_loss
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.min_cost_ratio = min_cost_ratio
        self.min_rank_corr = min_rank_corr
        self.min_rank_corr_change = min_rank_corr_change
        self.tail_quantile = tail_quantile
    
    def calculate_rolling_rank_correlations(self, price1: pd.Series, price2: pd.Series, 
                                         method: str = 'kendall') -> Tuple[float, float, float]:
        """
        장기 vs 단기 순위상관 및 변화량 계산
        
        Args:
            price1: 첫 번째 자산 가격
            price2: 두 번째 자산 가격
            method: 'kendall' 또는 'spearman'
            
        Returns:
            (long_corr, short_corr, delta_corr): 장기 상관, 단기 상관, 변화량
        """
        # 수익률 계산
        returns1 = price1.pct_change().dropna()
        returns2 = price2.pct_change().dropna()
        
        # 공통 인덱스
        common_idx = returns1.index.intersection(returns2.index)
        if len(common_idx) < max(self.long_window, self.short_window):
            return 0, 0, 0
            
        returns1_common = returns1[common_idx]
        returns2_common = returns2[common_idx]
        
        # 장기 순위상관
        if len(returns1_common) >= self.long_window:
            r1_long = returns1_common.tail(self.long_window)
            r2_long = returns2_common.tail(self.long_window)
        else:
            r1_long = returns1_common
            r2_long = returns2_common
            
        # 단기 순위상관
        if len(returns1_common) >= self.short_window:
            r1_short = returns1_common.tail(self.short_window)
            r2_short = returns2_common.tail(self.short_window)
        else:
            r1_short = returns1_common
            r2_short = returns2_common
        
        try:
            if method == 'kendall':
                long_corr, _ = kendalltau(r1_long, r2_long)
                short_corr, _ = kendalltau(r1_short, r2_short)
            elif method == 'spearman':
                long_corr, _ = spearmanr(r1_long, r2_long)
                short_corr, _ = spearmanr(r1_short, r2_short)
            else:
                raise ValueError("Method must be 'kendall' or 'spearman'")
                
            # NaN 값 처리
            long_corr = long_corr if not np.isnan(long_corr) else 0
            short_corr = short_corr if not np.isnan(short_corr) else 0
            delta_corr = abs(short_corr - long_corr)
            
            return long_corr, short_corr, delta_corr
            
        except Exception:
            return 0, 0, 0
    
    def calculate_enhanced_tail_dependence(self, price1: pd.Series, price2: pd.Series) -> Dict[str, float]:
        """
        향상된 꼬리 의존성 계산
        
        Returns:
            꼬리 의존성 지표들
        """
        returns1 = price1.pct_change().dropna()
        returns2 = price2.pct_change().dropna()
        
        # 공통 인덱스
        common_idx = returns1.index.intersection(returns2.index)
        if len(common_idx) < 100:  # 최소 표본 수
            return {'lower_tail': 0, 'upper_tail': 0, 'total_tail': 0, 'asymmetry': 0}
            
        returns1_common = returns1[common_idx]
        returns2_common = returns2[common_idx]
        
        # 임계값 계산
        threshold1_low = returns1_common.quantile(self.tail_quantile)
        threshold1_high = returns1_common.quantile(1 - self.tail_quantile)
        threshold2_low = returns2_common.quantile(self.tail_quantile)
        threshold2_high = returns2_common.quantile(1 - self.tail_quantile)
        
        # 하방 꼬리 의존성 (동시 극단 하락)
        joint_lower = ((returns1_common <= threshold1_low) & 
                      (returns2_common <= threshold2_low)).sum()
        expected_lower = len(returns1_common) * self.tail_quantile * self.tail_quantile
        lower_tail_dep = joint_lower / expected_lower if expected_lower > 0 else 0
        
        # 상방 꼬리 의존성 (동시 극단 상승)
        joint_upper = ((returns1_common >= threshold1_high) & 
                      (returns2_common >= threshold2_high)).sum()
        expected_upper = len(returns1_common) * self.tail_quantile * self.tail_quantile
        upper_tail_dep = joint_upper / expected_upper if expected_upper > 0 else 0
        
        # 전체 꼬리 의존성
        total_tail_dep = (joint_lower + joint_upper) / \
                        (2 * len(returns1_common) * self.tail_quantile * self.tail_quantile) \
                        if len(returns1_common) > 0 else 0
        
        # 비대칭성 (상방 vs 하방 의존성 차이)
        asymmetry = abs(upper_tail_dep - lower_tail_dep)
        
        return {
            'lower_tail': lower_tail_dep,
            'upper_tail': upper_tail_dep, 
            'total_tail': total_tail_dep,
            'asymmetry': asymmetry
        }
    
    def calculate_concordance_measures(self, price1: pd.Series, price2: pd.Series) -> Dict[str, float]:
        """
        일치성(Concordance) 측정 지표들 계산
        
        Returns:
            일치성 관련 지표들
        """
        returns1 = price1.pct_change().dropna()
        returns2 = price2.pct_change().dropna()
        
        # 공통 인덱스
        common_idx = returns1.index.intersection(returns2.index)
        if len(common_idx) < 30:
            return {'concordance_ratio': 0, 'kendall_tau': 0, 'spearman_rho': 0}
            
        returns1_common = returns1[common_idx]
        returns2_common = returns2[common_idx]
        
        # 기본 방향 일치 비율
        same_direction = ((returns1_common > 0) & (returns2_common > 0)) | \
                        ((returns1_common < 0) & (returns2_common < 0))
        concordance_ratio = same_direction.sum() / len(returns1_common)
        
        # Kendall's τ
        try:
            kendall_tau, _ = kendalltau(returns1_common, returns2_common)
            kendall_tau = kendall_tau if not np.isnan(kendall_tau) else 0
        except:
            kendall_tau = 0
        
        # Spearman's ρ
        try:
            spearman_rho, _ = spearmanr(returns1_common, returns2_common)
            spearman_rho = spearman_rho if not np.isnan(spearman_rho) else 0
        except:
            spearman_rho = 0
        
        return {
            'concordance_ratio': concordance_ratio,
            'kendall_tau': kendall_tau,
            'spearman_rho': spearman_rho
        }
    
    def select_pairs(self, prices: pd.DataFrame, n_pairs: int = 20) -> List[Dict]:
        """
        코퓰라·순위상관 기반 페어 선정
        
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
            if formation_data[col].notna().sum() >= self.formation_window * 0.8:
                valid_assets.append(col)
        
        if len(valid_assets) < 2:
            return []
            
        formation_data = formation_data[valid_assets].fillna(method='ffill')
        
        # 순위상관 및 꼬리 의존성 분석
        copula_results = []
        
        for i, asset1 in enumerate(valid_assets):
            for j, asset2 in enumerate(valid_assets):
                if i >= j:  # 중복 방지
                    continue
                
                # 장기/단기 순위상관 계산 (Kendall's τ)
                tau_long, tau_short, delta_tau = self.calculate_rolling_rank_correlations(
                    formation_data[asset1], formation_data[asset2], method='kendall'
                )
                
                # Spearman ρ 계산
                rho_long, rho_short, delta_rho = self.calculate_rolling_rank_correlations(
                    formation_data[asset1], formation_data[asset2], method='spearman'  
                )
                
                # 장기 순위상관이 충분히 높은 경우만 고려
                if abs(tau_long) < self.min_rank_corr and abs(rho_long) < self.min_rank_corr:
                    continue
                
                # 순위상관 변화가 충분히 큰 경우만 고려
                if delta_tau < self.min_rank_corr_change and delta_rho < self.min_rank_corr_change:
                    continue
                
                # 꼬리 의존성 계산
                tail_deps = self.calculate_enhanced_tail_dependence(
                    formation_data[asset1], formation_data[asset2]
                )
                
                # 일치성 측정
                concordance = self.calculate_concordance_measures(
                    formation_data[asset1], formation_data[asset2]
                )
                
                # 스프레드 품질 검사
                spread = calculate_spread(
                    formation_data[asset1], 
                    formation_data[asset2], 
                    hedge_ratio=1.0
                )
                
                half_life = calculate_half_life(spread)
                cost_ratio = calculate_transaction_cost_ratio(spread)
                
                # 품질 필터
                if (self.min_half_life <= half_life <= self.max_half_life and 
                    cost_ratio >= self.min_cost_ratio):
                    
                    # 코퓰라 품질 점수 계산
                    copula_score = self.calculate_copula_quality_score(
                        tau_long, tau_short, delta_tau, rho_long, rho_short, delta_rho,
                        tail_deps, concordance
                    )
                    
                    copula_results.append({
                        'asset1': asset1,
                        'asset2': asset2,
                        'tau_long': tau_long,
                        'tau_short': tau_short,
                        'delta_tau': delta_tau,
                        'rho_long': rho_long,
                        'rho_short': rho_short,
                        'delta_rho': delta_rho,
                        'tail_lower': tail_deps['lower_tail'],
                        'tail_upper': tail_deps['upper_tail'],
                        'tail_total': tail_deps['total_tail'],
                        'tail_asymmetry': tail_deps['asymmetry'],
                        'concordance_ratio': concordance['concordance_ratio'],
                        'half_life': half_life,
                        'cost_ratio': cost_ratio,
                        'copula_score': copula_score,
                        'hedge_ratio': 1.0,
                        'method': 'copula_rank_correlation'
                    })
        
        # 코퓰라 품질 점수 기준 정렬
        copula_results.sort(key=lambda x: x['copula_score'], reverse=True)
        
        # 중복 없는 페어 선정
        selected_pairs = []
        used_assets = set()
        
        for result in copula_results:
            if len(selected_pairs) >= n_pairs:
                break
                
            asset1, asset2 = result['asset1'], result['asset2']
            if asset1 not in used_assets and asset2 not in used_assets:
                selected_pairs.append(result)
                used_assets.add(asset1)
                used_assets.add(asset2)
        
        return selected_pairs
    
    def calculate_copula_quality_score(self, tau_long: float, tau_short: float, delta_tau: float,
                                     rho_long: float, rho_short: float, delta_rho: float,
                                     tail_deps: Dict, concordance: Dict) -> float:
        """
        코퓰라 기반 페어 품질 점수 계산
        
        Returns:
            품질 점수 (0~100)
        """
        score = 0
        
        # 1. 장기 순위상관 강도 (30%)
        avg_long_corr = (abs(tau_long) + abs(rho_long)) / 2
        corr_score = min(100, avg_long_corr * 100)
        score += corr_score * 0.3
        
        # 2. 순위상관 변화 크기 (25%) - 레짐 전환 포착
        avg_delta_corr = (delta_tau + delta_rho) / 2
        change_score = min(100, avg_delta_corr * 200)
        score += change_score * 0.25
        
        # 3. 꼬리 의존성 (25%)
        tail_score = min(100, tail_deps['total_tail'] * 100)
        score += tail_score * 0.25
        
        # 4. 전반적 일치성 (20%)
        concordance_score = concordance['concordance_ratio'] * 100
        score += concordance_score * 0.2
        
        return score
    
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
        
        # 현재 순위상관 상태 재확인
        current_tau_long, current_tau_short, current_delta_tau = \
            self.calculate_rolling_rank_correlations(recent_data[asset1], recent_data[asset2], 'kendall')
        
        current_rho_long, current_rho_short, current_delta_rho = \
            self.calculate_rolling_rank_correlations(recent_data[asset1], recent_data[asset2], 'spearman')
        
        # 스프레드 계산
        spread = calculate_spread(
            recent_data[asset1],
            recent_data[asset2],
            hedge_ratio=pair_info['hedge_ratio']
        )
        
        # Z-스코어 계산
        zscore = calculate_zscore(spread, window=self.signal_window)
        current_zscore = zscore.iloc[-1] if not zscore.empty else 0
        
        # 순위상관 필터 (변화가 지속되는 경우에만 진입)
        rank_corr_filter = (current_delta_tau >= self.min_rank_corr_change or 
                           current_delta_rho >= self.min_rank_corr_change)
        
        # 신호 생성
        signals = generate_trading_signals(
            zscore, 
            enter_threshold=self.enter_threshold,
            exit_threshold=self.exit_threshold,
            stop_loss=self.stop_loss
        )
        
        current_signal = signals.iloc[-1] if not signals.empty else 0
        
        # 순위상관 필터 적용
        if not rank_corr_filter:
            current_signal = 0
        
        # 신호 해석
        if current_signal == 1:
            signal_type = "ENTER_LONG"
            direction = f"Long {asset1}, Short {asset2}"
        elif current_signal == -1:
            signal_type = "ENTER_SHORT"
            direction = f"Short {asset1}, Long {asset2}"
        else:
            signal_type = "EXIT_OR_WAIT"
            direction = "Exit or Wait"
        
        return {
            'status': 'success',
            'pair': f"{asset1}-{asset2}",
            'signal_type': signal_type,
            'direction': direction,
            'current_zscore': current_zscore,
            'tau_long': pair_info['tau_long'],
            'tau_short': pair_info['tau_short'],
            'delta_tau': pair_info['delta_tau'],
            'current_delta_tau': current_delta_tau,
            'current_delta_rho': current_delta_rho,
            'tail_total': pair_info['tail_total'],
            'concordance_ratio': pair_info['concordance_ratio'],
            'copula_score': pair_info['copula_score'],
            'half_life': pair_info['half_life'],
            'cost_ratio': pair_info['cost_ratio'],
            'method': 'copula_rank_correlation'
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
            
            # 진입 신호 (|z| >= 2.0 & 순위상관 변화 지속)
            if current_z >= self.enter_threshold and signal_result['signal_type'] != 'EXIT_OR_WAIT':
                enter_signals.append(signal_result)
            # 관찰 대상 (1.5 <= |z| < 2.0)
            elif 1.5 <= current_z < self.enter_threshold:
                watch_signals.append(signal_result)
        
        # 코퓰라 품질 점수 기준 정렬
        enter_signals.sort(key=lambda x: x['copula_score'], reverse=True)
        watch_signals.sort(key=lambda x: x['copula_score'], reverse=True)
        
        return enter_signals[:n_pairs], watch_signals[:n_pairs]

def main():
    """
    코퓰라·순위상관 기반 페어트레이딩 실행 예제
    """
    # 데이터 로딩
    file_path = "/Users/a/PycharmProjects/pair_trading_signal/data/MU Price(BBG).csv"
    prices = load_data(file_path)
    
    # 코퓰라·순위상관 기반 페어트레이딩 객체 생성
    copula_trader = CopulaRankCorrelationPairTrading(
        formation_window=252,        # 1년
        signal_window=60,            # 3개월
        long_window=252,             # 장기 순위상관: 12개월
        short_window=60,             # 단기 순위상관: 3개월
        enter_threshold=2.0,
        exit_threshold=0.5,
        stop_loss=3.0,
        min_half_life=5,
        max_half_life=60,
        min_cost_ratio=5.0,
        min_rank_corr=0.3,           # 최소 순위상관 30%
        min_rank_corr_change=0.2,    # 최소 순위상관 변화 20%
        tail_quantile=0.1            # 상/하위 10% 꼬리
    )
    
    # 페어 스크리닝
    enter_list, watch_list = copula_trader.screen_pairs(prices, n_pairs=10)
    
    print("=" * 80)
    print("코퓰라·순위상관 기반 페어트레이딩 신호")
    print("=" * 80)
    
    print(f"\n📈 진입 신호 ({len(enter_list)}개):")
    print("-" * 70)
    for i, signal in enumerate(enter_list, 1):
        print(f"{i:2d}. {signal['pair']:20s} | {signal['direction']:25s}")
        print(f"     Z-Score: {signal['current_zscore']:6.2f} | Half-Life: {signal['half_life']:4.1f}D")
        print(f"     Kendall τ: {signal['tau_long']:6.3f}→{signal['tau_short']:6.3f} (Δ{signal['delta_tau']:6.3f})")
        print(f"     현재 Δτ: {signal['current_delta_tau']:6.3f} | 현재 Δρ: {signal['current_delta_rho']:6.3f}")
        print(f"     꼬리의존: {signal['tail_total']:6.3f} | 일치율: {signal['concordance_ratio']:6.3f}")
        print(f"     코퓰라점수: {signal['copula_score']:5.1f} | 비용비율: {signal['cost_ratio']:5.1f}")
        print()
    
    print(f"\n👀 관찰 대상 ({len(watch_list)}개):")
    print("-" * 70)
    for i, signal in enumerate(watch_list, 1):
        print(f"{i:2d}. {signal['pair']:20s} | Z-Score: {signal['current_zscore']:6.2f}")
        print(f"     Kendall Δτ: {signal['current_delta_tau']:6.3f} | Spearman Δρ: {signal['current_delta_rho']:6.3f}")
        print(f"     코퓰라점수: {signal['copula_score']:5.1f} | Half-Life: {signal['half_life']:4.1f}D")
        print()

if __name__ == "__main__":
    main()