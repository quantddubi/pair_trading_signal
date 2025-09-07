"""
공적분 분석 결과 사전 계산 및 캐시 생성
주요 파라미터 조합에 대해 미리 계산하여 JSON 파일로 저장
"""
import json
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import importlib.util

# 동적 모듈 import
def import_module_from_file(file_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# 모듈 import
common_utils = import_module_from_file("utils/common_utils.py", "common_utils")
cointegration_module = import_module_from_file("methods/2_cointegration_pairs.py", "cointegration_pairs")

load_data = common_utils.load_data
CointegrationPairTrading = cointegration_module.CointegrationPairTrading

def serialize_result(result):
    """결과를 JSON 직렬화 가능한 형태로 변환"""
    serializable = {}
    for key, value in result.items():
        if isinstance(value, (pd.Series, pd.DataFrame)):
            continue  # Skip pandas objects
        elif isinstance(value, np.floating):
            serializable[key] = float(value)
        elif isinstance(value, np.integer):
            serializable[key] = int(value)
        elif isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        else:
            serializable[key] = value
    return serializable

def generate_cointegration_cache():
    """주요 파라미터 조합에 대한 공적분 분석 결과 생성"""
    print("🔍 공적분 분석 캐시 생성 시작...")
    
    # 데이터 로딩
    file_path = "/Users/a/PycharmProjects/pair_trading_signal/data/MU Price(BBG).csv"
    prices = load_data(file_path)
    print(f"✅ 데이터 로딩 완료: {len(prices.columns)}개 자산, {len(prices)}일")
    
    # 주요 파라미터 조합 정의
    parameter_combinations = [
        # (formation_days, max_pvalue, n_pairs)
        (252, 0.05, 8),   # 1년, 5% 유의수준
        (252, 0.01, 8),   # 1년, 1% 유의수준  
        (504, 0.05, 8),   # 2년, 5% 유의수준
        (756, 0.05, 8),   # 3년, 5% 유의수준
        (756, 0.01, 8),   # 3년, 1% 유의수준
        (1008, 0.05, 8),  # 4년, 5% 유의수준
        (1260, 0.05, 8),  # 5년, 5% 유의수준
    ]
    
    cache_data = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'data_period': f"{prices.index[0]} to {prices.index[-1]}",
            'total_assets': len(prices.columns),
            'description': 'Pre-computed cointegration analysis results'
        },
        'results': {}
    }
    
    total_combinations = len(parameter_combinations)
    
    for i, (formation_days, max_pvalue, n_pairs) in enumerate(parameter_combinations, 1):
        print(f"\n📊 진행률: {i}/{total_combinations} - 분석 기간: {formation_days}일, p-value < {max_pvalue}")
        
        try:
            # 공적분 분석 실행
            trader = CointegrationPairTrading(
                formation_window=formation_days,
                signal_window=60,  # 고정
                enter_threshold=2.0,  # 고정
                exit_threshold=0.5,
                stop_loss=3.0,
                min_half_life=5,
                max_half_life=60,
                min_cost_ratio=5.0,
                max_pvalue=max_pvalue
            )
            
            enter_list, watch_list = trader.screen_pairs(prices, n_pairs=n_pairs)
            
            # 결과 직렬화
            cache_key = f"f{formation_days}_p{max_pvalue}_n{n_pairs}"
            cache_data['results'][cache_key] = {
                'parameters': {
                    'formation_days': formation_days,
                    'max_pvalue': max_pvalue,
                    'n_pairs': n_pairs,
                    'signal_window': 60
                },
                'enter_list': [serialize_result(r) for r in enter_list],
                'watch_list': [serialize_result(r) for r in watch_list],
                'stats': {
                    'enter_count': len(enter_list),
                    'watch_count': len(watch_list),
                    'avg_pvalue': np.mean([r.get('p_value', 0) for r in enter_list]) if enter_list else 0
                }
            }
            
            print(f"   ✅ 완료: {len(enter_list)}개 진입 신호, {len(watch_list)}개 관찰 대상")
            
        except Exception as e:
            print(f"   ❌ 오류: {str(e)}")
            cache_data['results'][cache_key] = {
                'error': str(e),
                'parameters': {
                    'formation_days': formation_days,
                    'max_pvalue': max_pvalue,
                    'n_pairs': n_pairs
                }
            }
    
    # 캐시 파일 저장
    cache_file = "/Users/a/PycharmProjects/pair_trading_signal/cointegration_cache.json"
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 캐시 생성 완료!")
    print(f"📁 파일 위치: {cache_file}")
    print(f"📈 총 {len([k for k in cache_data['results'] if 'error' not in cache_data['results'][k]])}개 조합 성공")

if __name__ == "__main__":
    generate_cointegration_cache()