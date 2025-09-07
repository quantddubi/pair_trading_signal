"""
빠른 페어트레이딩 스크리너 - 실행시간이 빠른 방법론들만 사용
"""
import pandas as pd
import numpy as np
from typing import List, Dict

import sys
import importlib.util

from utils.common_utils import load_data

# 숫자로 시작하는 모듈 동적 import
def import_module_from_file(file_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

euclidean_module = import_module_from_file("methods/1_euclidean_distance_pairs.py", "euclidean_distance_pairs")
EuclideanDistancePairTrading = euclidean_module.EuclideanDistancePairTrading

def main():
    """
    빠른 페어트레이딩 스크리너 실행
    """
    print("=" * 80)
    print("⚡ 빠른 페어트레이딩 스크리너")
    print("=" * 80)
    
    # 데이터 로딩
    file_path = "/Users/a/PycharmProjects/pair_trading_signal/data/MU Price(BBG).csv"
    prices = load_data(file_path)
    
    print(f"📊 데이터 로딩 완료: {len(prices.columns)}개 자산, {len(prices)}일 데이터")
    print(f"📅 기간: {prices.index[0].strftime('%Y-%m-%d')} ~ {prices.index[-1].strftime('%Y-%m-%d')}")
    
    # 1. 거리 기반 방법 (가장 빠름)
    print(f"\n🔍 1. 거리 기반 페어 분석...")
    euclidean_trader = EuclideanDistancePairTrading(
        formation_window=252,
        signal_window=60,
        enter_threshold=2.0,
        exit_threshold=0.5,
        stop_loss=3.0,
        min_half_life=5,
        max_half_life=60,
        min_cost_ratio=5.0
    )
    
    euclidean_enter, euclidean_watch = euclidean_trader.screen_pairs(prices, n_pairs=15)
    
    print(f"\n📈 진입 가능 페어 ({len(euclidean_enter)}개):")
    print("-" * 70)
    
    if euclidean_enter:
        for i, signal in enumerate(euclidean_enter, 1):
            print(f"{i:2d}. {signal['pair']:25s} | Z-Score: {signal['current_zscore']:6.2f}")
            print(f"     {signal['direction']:40s}")
            print(f"     Half-Life: {signal['half_life']:4.1f}일 | Cost Ratio: {signal['cost_ratio']:5.1f}")
            print(f"     거리 랭크: {signal['distance_rank']:.4f}")
            print()
    else:
        print("     현재 진입 가능한 페어가 없습니다.")
    
    print(f"\n👀 관찰 대상 ({len(euclidean_watch)}개):")
    print("-" * 70)
    
    if euclidean_watch:
        for i, signal in enumerate(euclidean_watch, 1):
            print(f"{i:2d}. {signal['pair']:25s} | Z-Score: {signal['current_zscore']:6.2f}")
            print(f"     Half-Life: {signal['half_life']:4.1f}일 | Distance: {signal['distance_rank']:.4f}")
    else:
        print("     현재 관찰 대상이 없습니다.")
    
    # 요약 통계
    print(f"\n📋 요약:")
    print(f"   • 총 분석 자산: {len(prices.columns)}개")
    print(f"   • 진입 신호: {len(euclidean_enter)}개 페어")
    print(f"   • 관찰 대상: {len(euclidean_watch)}개 페어")
    
    if euclidean_enter:
        avg_z_score = np.mean([abs(s['current_zscore']) for s in euclidean_enter])
        avg_half_life = np.mean([s['half_life'] for s in euclidean_enter])
        print(f"   • 평균 Z-Score: {avg_z_score:.2f}")
        print(f"   • 평균 Half-Life: {avg_half_life:.1f}일")
    
    print("\n" + "=" * 80)
    print("✅ 분석 완료! 위 결과를 참고하여 트레이딩 전략을 수립하세요.")
    print("📌 주의: 실제 트레이딩 전 추가적인 리스크 관리와 검증이 필요합니다.")

if __name__ == "__main__":
    main()