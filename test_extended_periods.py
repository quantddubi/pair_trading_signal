"""
확장된 기간으로 상관관계 레짐 테스트
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

def test_extended_periods():
    """확장된 기간으로 분석"""
    
    print("=" * 80)
    print("확장된 기간 상관관계 레짐 분석 (Formation: 3년, Signal: 1년)")
    print("=" * 80)
    
    # 데이터 로딩
    file_path = os.path.join(current_dir, "data", "MU Price(BBG).csv")
    prices = common_utils.load_data(file_path)
    
    print(f"전체 데이터 기간: {prices.index[0]} ~ {prices.index[-1]} ({len(prices)} 포인트)")
    
    # 확장된 파라미터로 객체 생성
    regime_trader = correlation_module.CorrelationRegimePairTrading(
        formation_window=756,      # 3년
        signal_window=252,         # 1년
        long_corr_window=504,      # 장기: 2년
        short_corr_window=252,     # 단기: 1년
        enter_threshold=2.0,
        exit_threshold=0.5,
        stop_loss=3.0,
        min_half_life=5,
        max_half_life=120,
        min_cost_ratio=5.0,
        min_delta_corr=0.2         # 20%로 완화
    )
    
    # Formation 데이터 확인
    formation_data = prices.tail(regime_trader.formation_window)
    print(f"Formation 데이터: {formation_data.index[0]} ~ {formation_data.index[-1]} ({len(formation_data)} 포인트)")
    
    # 페어 선정 과정
    print(f"\n페어 선정 과정:")
    selected_pairs = regime_trader.select_pairs(prices, n_pairs=20)
    print(f"선정된 페어: {len(selected_pairs)}개")
    
    if len(selected_pairs) == 0:
        print("❌ 선정된 페어가 없습니다!")
        return
    
    # 상위 10개 페어 분석
    print(f"\n상위 10개 페어 신호 분석:")
    print("-" * 80)
    
    enter_count = 0
    watch_count = 0
    
    for i, pair_info in enumerate(selected_pairs[:10]):
        pair_name = f"{pair_info['asset1']}-{pair_info['asset2']}"
        print(f"\n{i+1}. {pair_name}")
        print(f"   Formation 델타상관: {pair_info['delta_corr']:.3f}")
        print(f"   Half-life: {pair_info['half_life']:.1f}일")
        print(f"   Cost ratio: {pair_info['cost_ratio']:.1f}")
        
        # 신호 생성
        signal_result = regime_trader.generate_signals(prices, pair_info)
        
        if signal_result['status'] != 'success':
            print(f"   ❌ 신호 생성 실패: {signal_result['status']}")
            continue
        
        current_z = signal_result['current_zscore']
        abs_z = abs(current_z)
        current_delta = signal_result['current_delta_corr']
        
        print(f"   현재 Z-score: {current_z:.3f} (|z|={abs_z:.3f})")
        print(f"   현재 델타상관: {current_delta:.3f}")
        print(f"   현재 레짐: {signal_result['current_regime']}")
        print(f"   신호 타입: {signal_result['signal_type']}")
        
        # 조건 체크
        z_condition = abs_z >= regime_trader.enter_threshold
        regime_condition = current_delta >= regime_trader.min_delta_corr
        not_exit = signal_result['signal_type'] != 'EXIT_OR_WAIT'
        
        print(f"   조건: Z≥2.0({z_condition}) & Δ≥0.2({regime_condition}) & Signal({not_exit})")
        
        if z_condition and regime_condition and not_exit:
            print(f"   ✅ 진입 신호!")
            enter_count += 1
        elif 1.5 <= abs_z < regime_trader.enter_threshold:
            print(f"   👀 관찰 대상")
            watch_count += 1
        else:
            print(f"   ❌ 조건 미달")
    
    # 통계 분석
    print(f"\n" + "=" * 80)
    print(f"신호 생성 통계:")
    
    all_z_scores = []
    all_delta_corrs = []
    
    for pair_info in selected_pairs:
        signal_result = regime_trader.generate_signals(prices, pair_info)
        if signal_result['status'] == 'success':
            all_z_scores.append(abs(signal_result['current_zscore']))
            all_delta_corrs.append(signal_result['current_delta_corr'])
    
    if all_z_scores:
        z_scores = np.array(all_z_scores)
        delta_corrs = np.array(all_delta_corrs)
        
        print(f"\nZ-score 분포 ({len(z_scores)}개 페어):")
        print(f"  평균: {z_scores.mean():.3f}")
        print(f"  중앙값: {np.median(z_scores):.3f}")
        print(f"  최대: {z_scores.max():.3f}")
        print(f"  최소: {z_scores.min():.3f}")
        print(f"  ≥2.0: {(z_scores >= 2.0).sum()}개")
        print(f"  ≥1.5: {(z_scores >= 1.5).sum()}개")
        
        print(f"\nDelta Correlation 분포:")
        print(f"  평균: {delta_corrs.mean():.3f}")
        print(f"  중앙값: {np.median(delta_corrs):.3f}")
        print(f"  최대: {delta_corrs.max():.3f}")
        print(f"  최소: {delta_corrs.min():.3f}")
        print(f"  ≥0.2: {(delta_corrs >= 0.2).sum()}개")
        print(f"  ≥0.1: {(delta_corrs >= 0.1).sum()}개")
        
        # 결합 조건
        high_z = z_scores >= 2.0
        high_delta = delta_corrs >= 0.2
        
        print(f"\n결합 조건:")
        print(f"  Z-score ≥ 2.0: {high_z.sum()}개")
        print(f"  Delta corr ≥ 0.2: {high_delta.sum()}개")
        print(f"  둘 다 만족: {(high_z & high_delta).sum()}개")
        
        # 더 완화된 조건 테스트
        print(f"\n완화된 조건 테스트:")
        for z_thresh in [1.8, 1.5, 1.2]:
            for d_thresh in [0.15, 0.1, 0.05]:
                count = ((z_scores >= z_thresh) & (delta_corrs >= d_thresh)).sum()
                print(f"  Z≥{z_thresh}, Δ≥{d_thresh}: {count}개")

if __name__ == "__main__":
    test_extended_periods()