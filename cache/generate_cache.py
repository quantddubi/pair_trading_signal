#!/usr/bin/env python3
"""
각 방법론별 분석 결과를 사전 계산하여 캐시 파일로 저장하는 스크립트

이제 config.default_parameters에서 중앙화된 파라미터를 사용합니다.
Single Source of Truth를 유지하여 cache_utils.py와 항상 일치합니다.
"""
import os
import sys
import pickle
import json
from datetime import datetime
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 루트 경로 추가 (cache 폴더에서 상위 디렉토리로)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # 상위 디렉토리가 프로젝트 루트
sys.path.insert(0, project_root)

# 중앙화된 파라미터 import
from config.default_parameters import get_default_parameters

# 각 방법론 모듈 import
import importlib.util

def import_module_from_file(file_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# 공통 유틸리티
common_utils = import_module_from_file(os.path.join(project_root, "utils/common_utils.py"), "common_utils")

# 각 방법론 모듈
euclidean_module = import_module_from_file(os.path.join(project_root, "methods/1_euclidean_distance_pairs.py"), "euclidean_distance_pairs")
ssd_module = import_module_from_file(os.path.join(project_root, "methods/2_ssd_distance_pairs.py"), "ssd_distance_pairs")
cointegration_module = import_module_from_file(os.path.join(project_root, "methods/3_cointegration_pairs.py"), "cointegration_pairs")
regime_module = import_module_from_file(os.path.join(project_root, "methods/4_correlation_regime_pairs.py"), "correlation_regime_pairs")
ou_module = import_module_from_file(os.path.join(project_root, "methods/5_ou_mean_reversion_pairs.py"), "ou_mean_reversion_pairs")
clustering_module = import_module_from_file(os.path.join(project_root, "methods/6_clustering_pairs.py"), "clustering_pairs")
copula_module = import_module_from_file(os.path.join(project_root, "methods/7_copula_rank_correlation_pairs.py"), "copula_rank_correlation_pairs")

def get_data_last_date():
    """데이터의 마지막 날짜 확인"""
    file_path = os.path.join(project_root, "data/MU Price(BBG).csv")
    prices = common_utils.load_data(file_path)
    return prices.index[-1].strftime('%Y%m%d')

def generate_euclidean_cache():
    """유클리드 거리 방법론 캐시 생성"""
    print("유클리드 거리 방법론 캐시 생성 중...")

    file_path = os.path.join(project_root, "data/MU Price(BBG).csv")
    prices = common_utils.load_data(file_path)

    # ✅ config에서 기본 파라미터 가져오기 (Single Source of Truth)
    params = get_default_parameters('euclidean')
    print(f"  파라미터: formation_window={params['formation_window']}, signal_window={params['signal_window']}")

    # 파라미터로 분석
    trader = euclidean_module.EuclideanDistancePairTrading(**params)

    enter_list, watch_list = trader.screen_pairs(prices, n_pairs=10)

    cache_data = {
        'enter_signals': enter_list,
        'watch_signals': watch_list,
        'parameters': params,  # ✅ 동일한 파라미터 저장
        'generated_at': datetime.now().isoformat(),
        'data_date': get_data_last_date(),
        'method': 'euclidean'
    }

    # 캐시 데이터 디렉토리 생성 (cache/data/)
    cache_data_dir = os.path.join(current_dir, "data")
    if not os.path.exists(cache_data_dir):
        os.makedirs(cache_data_dir)

    # 캐시 저장
    cache_file = os.path.join(cache_data_dir, "euclidean_default.pkl")
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)

    print(f"완료: 유클리드 거리 캐시 저장 완료: {len(enter_list)}개 진입신호, {len(watch_list)}개 관찰대상")
    return len(enter_list), len(watch_list)

def generate_ssd_cache():
    """SSD 거리 방법론 캐시 생성"""
    print("- SSD 거리 방법론 캐시 생성 중...")

    file_path = os.path.join(project_root, "data/MU Price(BBG).csv")
    prices = common_utils.load_data(file_path)

    # ✅ config에서 기본 파라미터 가져오기
    params = get_default_parameters('ssd')
    print(f"  파라미터: formation_window={params['formation_window']}, signal_window={params['signal_window']}")

    trader = ssd_module.SSDDistancePairTrading(**params)

    enter_list, watch_list = trader.screen_pairs(prices, n_pairs=5)

    cache_data = {
        'enter_signals': enter_list,
        'watch_signals': watch_list,
        'parameters': params,  # ✅ 동일한 파라미터 저장
        'generated_at': datetime.now().isoformat(),
        'data_last_date': get_data_last_date()
    }

    # 캐시 데이터 디렉토리 생성 (cache/data/)
    cache_data_dir = os.path.join(current_dir, "data")
    if not os.path.exists(cache_data_dir):
        os.makedirs(cache_data_dir)

    # 캐시 저장
    cache_file = os.path.join(cache_data_dir, "ssd_default.pkl")
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)

    print(f"완료: SSD 거리 캐시 저장 완료: {len(enter_list)}개 진입신호, {len(watch_list)}개 관찰대상")
    return len(enter_list), len(watch_list)

def generate_cointegration_cache():
    """공적분 방법론 캐시 생성"""
    print("- 공적분 방법론 캐시 생성 중...")

    file_path = os.path.join(project_root, "data/MU Price(BBG).csv")
    prices = common_utils.load_data(file_path)

    # ✅ config에서 기본 파라미터 가져오기
    params = get_default_parameters('cointegration')
    print(f"  파라미터: formation_window={params['formation_window']}, signal_window={params['signal_window']}")

    trader = cointegration_module.CointegrationPairTrading(**params)

    enter_list, watch_list = trader.screen_pairs(prices, n_pairs=5)

    cache_data = {
        'enter_signals': enter_list,
        'watch_signals': watch_list,
        'generated_at': datetime.now().isoformat(),
        'data_date': get_data_last_date(),
        'parameters': params  # ✅ 동일한 파라미터 저장
    }

    # 캐시 데이터 디렉토리 생성 (cache/data/)
    cache_data_dir = os.path.join(current_dir, "data")
    if not os.path.exists(cache_data_dir):
        os.makedirs(cache_data_dir)

    cache_file = os.path.join(cache_data_dir, "cointegration_default.pkl")
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)

    print(f"완료: 공적분 캐시 완료: {len(enter_list)}개 진입 + {len(watch_list)}개 관찰")

def generate_regime_cache():
    """상관관계 레짐 방법론 캐시 생성"""
    print("- 상관관계 레짐 방법론 캐시 생성 중...")

    file_path = os.path.join(project_root, "data/MU Price(BBG).csv")
    prices = common_utils.load_data(file_path)

    # ✅ config에서 기본 파라미터 가져오기
    params = get_default_parameters('regime')
    print(f"  파라미터: formation_window={params['formation_window']}, long_corr_window={params['long_corr_window']}")

    trader = regime_module.CorrelationRegimePairTrading(**params)

    enter_list, watch_list = trader.screen_pairs(prices, n_pairs=5)

    cache_data = {
        'enter_signals': enter_list,
        'watch_signals': watch_list,
        'generated_at': datetime.now().isoformat(),
        'data_date': get_data_last_date(),
        'parameters': params  # ✅ 동일한 파라미터 저장
    }

    # 캐시 데이터 디렉토리 생성 (cache/data/)
    cache_data_dir = os.path.join(current_dir, "data")
    if not os.path.exists(cache_data_dir):
        os.makedirs(cache_data_dir)

    cache_file = os.path.join(cache_data_dir, "regime_default.pkl")
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)

    print(f"완료: 레짐 캐시 완료: {len(enter_list)}개 진입 + {len(watch_list)}개 관찰")

def generate_ou_cache():
    """OU 평균회귀 방법론 캐시 생성"""
    print("- OU 평균회귀 방법론 캐시 생성 중...")

    file_path = os.path.join(project_root, "data/MU Price(BBG).csv")
    prices = common_utils.load_data(file_path)

    # ✅ config에서 기본 파라미터 가져오기
    params = get_default_parameters('ou')
    print(f"  파라미터: formation_window={params['formation_window']}, rolling_window={params['rolling_window']}")

    trader = ou_module.OUMeanReversionPairTrading(**params)

    selected_pairs = trader.select_pairs(prices, n_pairs=10)

    # 진입 신호와 관찰 대상 분리
    enter_list = [pair for pair in selected_pairs if pair.get('signal_type') == 'ENTRY']
    watch_list = [pair for pair in selected_pairs if pair.get('signal_type') == 'WATCH']

    cache_data = {
        'enter_signals': enter_list,
        'watch_signals': watch_list,
        'selected_pairs': selected_pairs,
        'generated_at': datetime.now().isoformat(),
        'data_date': get_data_last_date(),
        'parameters': params  # ✅ 동일한 파라미터 저장
    }

    # 캐시 데이터 디렉토리 생성 (cache/data/)
    cache_data_dir = os.path.join(current_dir, "data")
    if not os.path.exists(cache_data_dir):
        os.makedirs(cache_data_dir)

    cache_file = os.path.join(cache_data_dir, "ou_default.pkl")
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)

    print(f"완료: OU 캐시 완료: {len(selected_pairs)}개 선별 ({len(enter_list)}개 진입 + {len(watch_list)}개 관찰)")

def generate_clustering_cache():
    """클러스터링 방법론 캐시 생성"""
    print("- 클러스터링 방법론 캐시 생성 중...")

    file_path = os.path.join(project_root, "data/MU Price(BBG).csv")
    prices = common_utils.load_data(file_path)

    # ✅ config에서 기본 파라미터 가져오기
    params = get_default_parameters('clustering')
    print(f"  파라미터: formation_window={params['formation_window']}, n_clusters={params['n_clusters']}")

    trader = clustering_module.ClusteringPairTrading(**params)

    enter_list, watch_list = trader.screen_pairs(prices, n_pairs=5)

    # 클러스터 요약 정보도 추가
    cluster_summary = trader.get_cluster_summary(prices)

    cache_data = {
        'enter_signals': enter_list,
        'watch_signals': watch_list,
        'cluster_summary': cluster_summary,
        'generated_at': datetime.now().isoformat(),
        'data_date': get_data_last_date(),
        'parameters': params  # ✅ 동일한 파라미터 저장
    }

    # 캐시 데이터 디렉토리 생성 (cache/data/)
    cache_data_dir = os.path.join(current_dir, "data")
    if not os.path.exists(cache_data_dir):
        os.makedirs(cache_data_dir)

    cache_file = os.path.join(cache_data_dir, "clustering_default.pkl")
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)

    print(f"완료: 클러스터링 캐시 완료: {len(enter_list)}개 진입 + {len(watch_list)}개 관찰")

def generate_copula_cache():
    """코퓰라 실시간 스크리닝 방법론 캐시 생성"""
    print("- 코퓰라 실시간 스크리닝 방법론 캐시 생성 중...")

    file_path = os.path.join(project_root, "data/MU Price(BBG).csv")
    prices = common_utils.load_data(file_path)

    # ✅ config에서 기본 파라미터 가져오기
    params = get_default_parameters('copula')
    print(f"  파라미터: formation_window={params['formation_window']}, min_tail_dependence={params['min_tail_dependence']}")

    screener = copula_module.CopulaBasedPairScreening(**params)

    selected_pairs = screener.select_pairs(prices, n_pairs=10)

    # 신호 타입별 분리
    enter_list = [pair for pair in selected_pairs if pair.get('signal_type') in ['LONG', 'SHORT']]
    watch_list = [pair for pair in selected_pairs if pair.get('signal_type') == 'NEUTRAL']

    cache_data = {
        'enter_signals': enter_list,
        'watch_signals': watch_list,
        'selected_pairs': selected_pairs,
        'generated_at': datetime.now().isoformat(),
        'data_date': get_data_last_date(),
        'parameters': params  # ✅ 동일한 파라미터 저장
    }

    # 캐시 데이터 디렉토리 생성 (cache/data/)
    cache_data_dir = os.path.join(current_dir, "data")
    if not os.path.exists(cache_data_dir):
        os.makedirs(cache_data_dir)

    cache_file = os.path.join(cache_data_dir, "copula_default.pkl")
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)

    print(f"완료: 코퓰라 캐시 완료: {len(selected_pairs)}개 선별 ({len(enter_list)}개 진입 + {len(watch_list)}개 중성)")

def main():
    """모든 방법론 캐시 생성"""
    print("=" * 60)
    print("페어트레이딩 방법론별 캐시 생성 시작")
    print("=" * 60)
    
    data_date = get_data_last_date()
    print(f"데이터 기준일: {data_date}")
    
    try:
        # 각 방법론별 캐시 생성 (시간이 오래 걸리므로 순차적으로)
        generate_euclidean_cache()
        generate_ssd_cache()
        generate_cointegration_cache()
        generate_regime_cache() 
        generate_ou_cache()
        generate_clustering_cache()
        generate_copula_cache()
        
        print("\n" + "=" * 60)
        print("🎉 모든 캐시 생성 완료!")
        print("=" * 60)
        print(f"캐시 위치: {os.path.join(current_dir, 'data')}")
        
    except Exception as e:
        print(f"❌ 캐시 생성 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    main()