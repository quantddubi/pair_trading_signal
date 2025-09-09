"""
상관관계 레짐 전환 방법론 디버깅 - 각 단계별 필터링 원인 분석
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

def analyze_correlation_regime_filtering():
    """각 필터링 단계별로 얼마나 많은 페어가 제외되는지 분석"""
    
    print("=" * 80)
    print("상관관계 레짐 전환 방법론 디버깅 분석")
    print("=" * 80)
    
    # 데이터 로딩
    file_path = os.path.join(current_dir, "data", "MU Price(BBG).csv")
    prices = common_utils.load_data(file_path)
    
    print(f"전체 자산 수: {len(prices.columns)}")
    print(f"데이터 기간: {prices.index[0]} ~ {prices.index[-1]}")
    print(f"총 데이터 포인트: {len(prices)}")
    print()
    
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
    
    # 1단계: Formation 데이터 준비
    formation_data = prices.tail(regime_trader.formation_window)
    print(f"1단계 - Formation 데이터 기간: {formation_data.index[0]} ~ {formation_data.index[-1]}")
    print(f"Formation 데이터 길이: {len(formation_data)}")
    
    # 2단계: 유효 자산 필터링 (80% 이상 데이터 있는 자산)
    valid_assets = []
    asset_data_completeness = {}
    
    for col in formation_data.columns:
        completeness = formation_data[col].notna().sum() / len(formation_data)
        asset_data_completeness[col] = completeness
        if completeness >= 0.8:
            valid_assets.append(col)
    
    print(f"2단계 - 유효 자산 필터링:")
    print(f"  전체 자산: {len(formation_data.columns)}")
    print(f"  80% 이상 데이터 있는 자산: {len(valid_assets)}")
    print(f"  제외된 자산: {len(formation_data.columns) - len(valid_assets)}")
    
    # 데이터 완성도가 낮은 상위 10개 자산 표시
    low_completeness = sorted(asset_data_completeness.items(), key=lambda x: x[1])[:10]
    print("  데이터 완성도가 낮은 자산 (상위 10개):")
    for asset, completeness in low_completeness:
        print(f"    {asset}: {completeness:.2%}")
    print()
    
    if len(valid_assets) < 2:
        print("ERROR: 유효 자산이 2개 미만입니다.")
        return
    
    formation_data = formation_data[valid_assets].fillna(method='ffill')
    
    # 3단계: 모든 가능한 페어 생성
    all_pairs = []
    for i, asset1 in enumerate(valid_assets):
        for asset2 in valid_assets[i+1:]:
            all_pairs.append((asset1, asset2))
    
    print(f"3단계 - 전체 페어 후보: {len(all_pairs)}개")
    
    # 4단계: 상관관계 레짐 분석 및 필터링
    step_results = {
        'total_pairs': len(all_pairs),
        'correlation_calculated': 0,
        'min_delta_corr_passed': 0,
        'half_life_passed': 0,
        'cost_ratio_passed': 0,
        'final_selected': 0
    }
    
    detailed_results = []
    
    for i, (asset1, asset2) in enumerate(all_pairs):
        if i < 20:  # 처음 20개 페어만 상세 분석
            print(f"\n페어 {i+1}: {asset1} - {asset2}")
        
        # 상관관계 변화 계산
        long_corr, short_corr, delta_corr = regime_trader.calculate_rolling_correlations(
            formation_data[asset1], formation_data[asset2]
        )
        
        if long_corr == 0 and short_corr == 0 and delta_corr == 0:
            if i < 20:
                print(f"  ❌ 상관관계 계산 실패 (데이터 부족)")
            continue
        
        step_results['correlation_calculated'] += 1
        
        if i < 20:
            print(f"  장기 상관: {long_corr:.3f}, 단기 상관: {short_corr:.3f}")
            print(f"  델타 상관: {delta_corr:.3f} (기준: {regime_trader.min_delta_corr})")
        
        # 최소 변화 임계값 필터
        if delta_corr < regime_trader.min_delta_corr:
            if i < 20:
                print(f"  ❌ 델타 상관 기준 미달")
            continue
        
        step_results['min_delta_corr_passed'] += 1
        if i < 20:
            print(f"  ✅ 델타 상관 기준 통과")
        
        # 스프레드 계산
        spread = common_utils.calculate_spread(
            formation_data[asset1], 
            formation_data[asset2], 
            hedge_ratio=1.0
        )
        
        # Half-life 계산
        half_life = common_utils.calculate_half_life(spread)
        
        if i < 20:
            print(f"  Half-life: {half_life:.1f} (기준: {regime_trader.min_half_life}-{regime_trader.max_half_life})")
        
        if not (regime_trader.min_half_life <= half_life <= regime_trader.max_half_life):
            if i < 20:
                print(f"  ❌ Half-life 기준 미달")
            continue
        
        step_results['half_life_passed'] += 1
        if i < 20:
            print(f"  ✅ Half-life 기준 통과")
        
        # 거래비용 대비 수익성 계산
        cost_ratio = common_utils.calculate_transaction_cost_ratio(spread)
        
        if i < 20:
            print(f"  Cost ratio: {cost_ratio:.1f} (기준: {regime_trader.min_cost_ratio})")
        
        if cost_ratio < regime_trader.min_cost_ratio:
            if i < 20:
                print(f"  ❌ Cost ratio 기준 미달")
            continue
        
        step_results['cost_ratio_passed'] += 1
        step_results['final_selected'] += 1
        
        if i < 20:
            print(f"  ✅ Cost ratio 기준 통과")
            print(f"  ✅ 최종 선정!")
        
        # 결과 저장
        detailed_results.append({
            'pair': f"{asset1}-{asset2}",
            'long_corr': long_corr,
            'short_corr': short_corr,
            'delta_corr': delta_corr,
            'half_life': half_life,
            'cost_ratio': cost_ratio,
            'regime_type': regime_trader.classify_regime_type(long_corr, short_corr)
        })
    
    # 결과 요약
    print("\n" + "=" * 80)
    print("필터링 단계별 결과 요약")
    print("=" * 80)
    print(f"1. 전체 페어 후보:           {step_results['total_pairs']:,}")
    print(f"2. 상관관계 계산 성공:       {step_results['correlation_calculated']:,} ({step_results['correlation_calculated']/step_results['total_pairs']:.1%})")
    print(f"3. 델타 상관 기준 통과:      {step_results['min_delta_corr_passed']:,} ({step_results['min_delta_corr_passed']/step_results['total_pairs']:.1%})")
    print(f"4. Half-life 기준 통과:      {step_results['half_life_passed']:,} ({step_results['half_life_passed']/step_results['total_pairs']:.1%})")
    print(f"5. Cost ratio 기준 통과:     {step_results['cost_ratio_passed']:,} ({step_results['cost_ratio_passed']/step_results['total_pairs']:.1%})")
    print(f"6. 최종 선정 페어:           {step_results['final_selected']:,} ({step_results['final_selected']/step_results['total_pairs']:.1%})")
    
    # 가장 큰 병목 지점 식별
    bottlenecks = []
    prev_count = step_results['total_pairs']
    
    for step_name, count in [
        ('상관관계 계산', step_results['correlation_calculated']),
        ('델타 상관 필터', step_results['min_delta_corr_passed']),
        ('Half-life 필터', step_results['half_life_passed']),
        ('Cost ratio 필터', step_results['cost_ratio_passed'])
    ]:
        filtered_out = prev_count - count
        if filtered_out > 0:
            bottlenecks.append((step_name, filtered_out, filtered_out/step_results['total_pairs']))
        prev_count = count
    
    print(f"\n주요 병목 지점:")
    for step_name, filtered_count, percentage in sorted(bottlenecks, key=lambda x: x[1], reverse=True)[:3]:
        print(f"  {step_name}: {filtered_count:,}개 제외 ({percentage:.1%})")
    
    # 최종 선정된 페어들 정보
    if detailed_results:
        print(f"\n최종 선정된 페어들 ({len(detailed_results)}개):")
        print("-" * 80)
        for result in sorted(detailed_results, key=lambda x: x['delta_corr'], reverse=True):
            print(f"  {result['pair']:25} | ΔCorr: {result['delta_corr']:.3f} | HL: {result['half_life']:5.1f} | CR: {result['cost_ratio']:5.1f}")
    else:
        print("\n❌ 최종 선정된 페어가 없습니다!")
        
        # 각 기준을 완화했을 때의 시뮬레이션
        print("\n기준 완화 시뮬레이션:")
        print("-" * 40)
        
        relaxed_tests = [
            ('min_delta_corr', 0.2, '델타 상관 기준'),
            ('min_delta_corr', 0.1, '델타 상관 기준'),
            ('min_half_life', 1, 'Half-life 최소값'),
            ('max_half_life', 120, 'Half-life 최대값'),
            ('min_cost_ratio', 3.0, 'Cost ratio 기준'),
            ('min_cost_ratio', 2.0, 'Cost ratio 기준')
        ]
        
        for param_name, new_value, description in relaxed_tests:
            # 임시로 기준 완화
            original_value = getattr(regime_trader, param_name)
            setattr(regime_trader, param_name, new_value)
            
            # 다시 스크리닝
            test_pairs = regime_trader.select_pairs(prices, n_pairs=20)
            
            print(f"  {description} {original_value} → {new_value}: {len(test_pairs)}개 페어")
            
            # 원래 값으로 복원
            setattr(regime_trader, param_name, original_value)

if __name__ == "__main__":
    analyze_correlation_regime_filtering()