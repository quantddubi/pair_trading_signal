"""
상관관계 레짐 진입 신호 생성 단계별 디버깅
"""
import pandas as pd
import numpy as np
import os
import sys
import importlib.util

# 공통 유틸리티 import
current_dir = os.path.dirname(os.path.abspath(__file__))
utils_path = os.path.join(current_dir, "utils", "common_utils.py")
spec = importlib.util.spec_from_file_location("common_utils", utils_path)
common_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common_utils)

# methods 디렉토리에서 상관관계 레짐 클래스 import
methods_path = os.path.join(current_dir, "methods", "4_correlation_regime_pairs.py")
spec = importlib.util.spec_from_file_location("correlation_regime", methods_path)
correlation_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(correlation_module)

def analyze_signal_generation():
    """진입 신호 생성 과정 상세 분석"""
    
    print("=" * 80)
    print("상관관계 레짐 진입 신호 생성 디버깅")
    print("=" * 80)
    
    # 데이터 로딩
    file_path = os.path.join(current_dir, "data", "MU Price(BBG).csv")
    prices = common_utils.load_data(file_path)
    
    # 상관관계 레짐 전환 객체 생성
    regime_trader = correlation_module.CorrelationRegimePairTrading(
        formation_window=252,
        signal_window=60,
        long_corr_window=252,
        short_corr_window=60,
        enter_threshold=2.0,
        exit_threshold=0.5,
        stop_loss=3.0,
        min_half_life=5,
        max_half_life=60,
        min_cost_ratio=5.0,
        min_delta_corr=0.3
    )
    
    # 페어 선정
    selected_pairs = regime_trader.select_pairs(prices, n_pairs=20)
    print(f"선정된 페어 수: {len(selected_pairs)}")
    
    if len(selected_pairs) == 0:
        print("❌ 선정된 페어가 없습니다!")
        return
    
    # 상위 10개 페어에 대해 신호 생성 과정 분석
    print(f"\n상위 10개 페어 신호 생성 과정:")
    print("-" * 80)
    
    enter_signals = []
    watch_signals = []
    
    for i, pair_info in enumerate(selected_pairs[:10]):
        pair_name = f"{pair_info['asset1']}-{pair_info['asset2']}"
        print(f"\n페어 {i+1}: {pair_name} (Delta Corr: {pair_info['delta_corr']:.3f})")
        
        # 신호 생성
        signal_result = regime_trader.generate_signals(prices, pair_info)
        
        if signal_result['status'] != 'success':
            print(f"  ❌ 신호 생성 실패: {signal_result['status']}")
            continue
        
        current_z = signal_result['current_zscore']
        abs_z = abs(current_z)
        
        print(f"  현재 z-score: {current_z:.3f} (|z|={abs_z:.3f})")
        print(f"  진입 임계값: {regime_trader.enter_threshold}")
        print(f"  레짐 필터: {signal_result['current_delta_corr']:.3f} >= {regime_trader.min_delta_corr} = {signal_result['current_delta_corr'] >= regime_trader.min_delta_corr}")
        print(f"  신호 타입: {signal_result['signal_type']}")
        print(f"  현재 레짐: {signal_result['current_regime']}")
        
        # 진입 조건 체크
        z_condition = abs_z >= regime_trader.enter_threshold
        regime_condition = signal_result['current_delta_corr'] >= regime_trader.min_delta_corr
        not_exit = signal_result['signal_type'] != 'EXIT_OR_WAIT'
        
        print(f"  Z-score 조건: {z_condition} (|{current_z:.3f}| >= {regime_trader.enter_threshold})")
        print(f"  레짐 조건: {regime_condition}")
        print(f"  진입 신호: {not_exit}")
        
        # 분류
        if z_condition and not_exit and regime_condition:
            print(f"  ✅ 진입 신호 조건 만족!")
            enter_signals.append(signal_result)
        elif 1.5 <= abs_z < regime_trader.enter_threshold:
            print(f"  👀 관찰 대상 (Z-score 범위)")
            watch_signals.append(signal_result)
        else:
            print(f"  ❌ 조건 미달")
    
    print(f"\n" + "=" * 80)
    print(f"결과 요약:")
    print(f"진입 신호: {len(enter_signals)}개")
    print(f"관찰 대상: {len(watch_signals)}개")
    
    # Z-score 분포 분석
    print(f"\n선정된 모든 페어의 Z-score 분포:")
    print("-" * 50)
    
    z_scores = []
    delta_corrs = []
    
    for pair_info in selected_pairs:
        signal_result = regime_trader.generate_signals(prices, pair_info)
        if signal_result['status'] == 'success':
            z_scores.append(abs(signal_result['current_zscore']))
            delta_corrs.append(signal_result['current_delta_corr'])
    
    z_scores = np.array(z_scores)
    delta_corrs = np.array(delta_corrs)
    
    print(f"Z-score 통계:")
    print(f"  평균: {z_scores.mean():.3f}")
    print(f"  중앙값: {np.median(z_scores):.3f}")
    print(f"  최대값: {z_scores.max():.3f}")
    print(f"  최소값: {z_scores.min():.3f}")
    print(f"  >= 2.0인 페어: {(z_scores >= 2.0).sum()}개")
    print(f"  >= 1.5인 페어: {(z_scores >= 1.5).sum()}개")
    
    print(f"\nDelta Correlation 통계:")
    print(f"  평균: {delta_corrs.mean():.3f}")
    print(f"  중앙값: {np.median(delta_corrs):.3f}")
    print(f"  최대값: {delta_corrs.max():.3f}")
    print(f"  최소값: {delta_corrs.min():.3f}")
    print(f"  >= 0.3인 페어: {(delta_corrs >= 0.3).sum()}개")
    
    # 조건별 페어 분석
    print(f"\n조건별 분석:")
    high_z = z_scores >= 2.0
    high_delta = delta_corrs >= 0.3
    
    print(f"  Z-score >= 2.0: {high_z.sum()}개")
    print(f"  Delta corr >= 0.3: {high_delta.sum()}개")
    print(f"  둘 다 만족: {(high_z & high_delta).sum()}개")

if __name__ == "__main__":
    analyze_signal_generation()