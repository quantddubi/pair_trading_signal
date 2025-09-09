"""
상관관계 계산 함수 문제 진단
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

def test_correlation_calculation():
    """상관관계 계산 함수 문제 진단"""
    
    print("상관관계 계산 함수 디버깅")
    print("=" * 50)
    
    # 데이터 로딩
    file_path = os.path.join(current_dir, "data", "MU Price(BBG).csv")
    prices = common_utils.load_data(file_path)
    
    # 상관관계 레짐 전환 객체 생성
    regime_trader = correlation_module.CorrelationRegimePairTrading(
        long_corr_window=252,
        short_corr_window=60
    )
    
    # Formation 데이터 준비
    formation_data = prices.tail(252)
    print(f"Formation 데이터 크기: {formation_data.shape}")
    print(f"Formation 데이터 기간: {formation_data.index[0]} ~ {formation_data.index[-1]}")
    
    # 첫 번째 두 자산 선택
    asset1 = formation_data.columns[0]  # SPX Index
    asset2 = formation_data.columns[1]  # RTY Index
    
    print(f"\n테스트 페어: {asset1} - {asset2}")
    
    price1 = formation_data[asset1]
    price2 = formation_data[asset2]
    
    print(f"Asset1 데이터 길이: {len(price1)}")
    print(f"Asset2 데이터 길이: {len(price2)}")
    print(f"Asset1 결측치: {price1.isna().sum()}")
    print(f"Asset2 결측치: {price2.isna().sum()}")
    
    # 수익률 계산
    returns1 = price1.pct_change().dropna()
    returns2 = price2.pct_change().dropna()
    
    print(f"\nReturns1 길이: {len(returns1)}")
    print(f"Returns2 길이: {len(returns2)}")
    print(f"Returns1 결측치: {returns1.isna().sum()}")
    print(f"Returns2 결측치: {returns2.isna().sum()}")
    
    # 공통 인덱스 확인
    common_idx = returns1.index.intersection(returns2.index)
    print(f"공통 인덱스 길이: {len(common_idx)}")
    print(f"필요한 최소 길이: {max(regime_trader.long_corr_window, regime_trader.short_corr_window)}")
    
    if len(common_idx) < max(regime_trader.long_corr_window, regime_trader.short_corr_window):
        print("❌ 공통 인덱스가 필요한 윈도우보다 짧습니다!")
        print(f"   공통 인덱스: {len(common_idx)}")
        print(f"   장기 윈도우 필요: {regime_trader.long_corr_window}")
        print(f"   단기 윈도우 필요: {regime_trader.short_corr_window}")
        
        # 실제 공통 인덱스 확인
        print(f"\nReturns1 인덱스 범위: {returns1.index[0]} ~ {returns1.index[-1]}")
        print(f"Returns2 인덱스 범위: {returns2.index[0]} ~ {returns2.index[-1]}")
        print(f"공통 인덱스 범위: {common_idx[0] if len(common_idx) > 0 else 'None'} ~ {common_idx[-1] if len(common_idx) > 0 else 'None'}")
    else:
        print("✅ 공통 인덱스가 충분합니다!")
    
    # 직접 상관관계 계산 테스트
    returns1_common = returns1[common_idx]
    returns2_common = returns2[common_idx]
    
    if len(returns1_common) > 0:
        full_corr = returns1_common.corr(returns2_common)
        print(f"\n전체 기간 상관관계: {full_corr:.6f}")
        
        # 장기 상관관계 계산 시도
        if len(returns1_common) >= regime_trader.long_corr_window:
            long_corr = returns1_common.tail(regime_trader.long_corr_window).corr(
                returns2_common.tail(regime_trader.long_corr_window)
            )
            print(f"장기 상관관계 (252일): {long_corr:.6f}")
        else:
            print(f"❌ 장기 상관관계 계산 불가 (데이터 {len(returns1_common)}일 < 필요 {regime_trader.long_corr_window}일)")
        
        # 단기 상관관계 계산 시도
        if len(returns1_common) >= regime_trader.short_corr_window:
            short_corr = returns1_common.tail(regime_trader.short_corr_window).corr(
                returns2_common.tail(regime_trader.short_corr_window)
            )
            print(f"단기 상관관계 (60일): {short_corr:.6f}")
        else:
            print(f"❌ 단기 상관관계 계산 불가 (데이터 {len(returns1_common)}일 < 필요 {regime_trader.short_corr_window}일)")
    
    # 실제 calculate_rolling_correlations 함수 호출 테스트
    print(f"\n실제 함수 호출 결과:")
    long_corr, short_corr, delta_corr = regime_trader.calculate_rolling_correlations(price1, price2)
    print(f"Long corr: {long_corr}")
    print(f"Short corr: {short_corr}")
    print(f"Delta corr: {delta_corr}")
    
    # 다른 페어들도 테스트
    print(f"\n다른 페어들 테스트:")
    for i in range(min(5, len(formation_data.columns)-1)):
        asset_a = formation_data.columns[i]
        asset_b = formation_data.columns[i+1]
        
        long_c, short_c, delta_c = regime_trader.calculate_rolling_correlations(
            formation_data[asset_a], formation_data[asset_b]
        )
        print(f"{asset_a[:15]:15} - {asset_b[:15]:15}: Long={long_c:.3f}, Short={short_c:.3f}, Delta={delta_c:.3f}")

if __name__ == "__main__":
    test_correlation_calculation()