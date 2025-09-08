#!/usr/bin/env python3
"""
각 방법론별 분석 결과를 사전 계산하여 캐시 파일로 저장하는 스크립트
"""
import os
import sys
import pickle
import json
from datetime import datetime
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 각 방법론 모듈 import
import importlib.util

def import_module_from_file(file_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# 공통 유틸리티
common_utils = import_module_from_file("utils/common_utils.py", "common_utils")

# 각 방법론 모듈
euclidean_module = import_module_from_file("methods/1_euclidean_distance_pairs.py", "euclidean_distance_pairs")
ssd_module = import_module_from_file("methods/2_ssd_distance_pairs.py", "ssd_distance_pairs")
cointegration_module = import_module_from_file("methods/3_cointegration_pairs.py", "cointegration_pairs")
regime_module = import_module_from_file("methods/4_correlation_regime_pairs.py", "correlation_regime_pairs")
ou_module = import_module_from_file("methods/5_ou_mean_reversion_pairs.py", "ou_mean_reversion_pairs")
clustering_module = import_module_from_file("methods/6_clustering_pairs.py", "clustering_pairs")
copula_module = import_module_from_file("methods/7_copula_rank_correlation_pairs.py", "copula_rank_correlation_pairs")

def get_data_last_date():
    """데이터의 마지막 날짜 확인"""
    file_path = "data/MU Price(BBG).csv"
    prices = common_utils.load_data(file_path)
    return prices.index[-1].strftime('%Y%m%d')

def generate_euclidean_cache():
    """유클리드 거리 방법론 캐시 생성"""
    print("유클리드 거리 방법론 캐시 생성 중...")
    
    file_path = "data/MU Price(BBG).csv"
    prices = common_utils.load_data(file_path)
    
    # 기본 파라미터로 분석
    trader = euclidean_module.EuclideanDistancePairTrading(
        formation_window=756,
        signal_window=756,
        enter_threshold=2.0,
        exit_threshold=0.5,
        stop_loss=3.0,
        min_half_life=5,
        max_half_life=60,
        min_cost_ratio=5.0,
        transaction_cost=0.0001
    )
    
    enter_list, watch_list = trader.screen_pairs(prices, n_pairs=10)
    
    cache_data = {
        'enter_signals': enter_list,
        'watch_signals': watch_list,
        'parameters': {
            'formation_window': 756,
            'signal_window': 756,
            'enter_threshold': 2.0,
            'exit_threshold': 0.5,
            'stop_loss': 3.0,
            'min_half_life': 5,
            'max_half_life': 60,
            'min_cost_ratio': 5.0,
            'transaction_cost': 0.0001
        },
        'generated_at': datetime.now().isoformat(),
        'data_date': get_data_last_date(),
        'method': 'euclidean'
    }
    
    # 캐시 디렉토리 생성
    cache_dir = "cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    # 캐시 저장
    cache_file = os.path.join(cache_dir, "euclidean_default.pkl")
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f"✅ 유클리드 거리 캐시 저장 완료: {len(enter_list)}개 진입신호, {len(watch_list)}개 관찰대상")
    return len(enter_list), len(watch_list)

def generate_ssd_cache():
    """SSD 거리 방법론 캐시 생성"""
    print("🔄 SSD 거리 방법론 캐시 생성 중...")
    
    file_path = "data/MU Price(BBG).csv"
    prices = common_utils.load_data(file_path)
    
    # 기본 파라미터로 분석
    trader = ssd_module.SSDDistancePairTrading(
        formation_window=252,
        signal_window=252,
        enter_threshold=2.0,
        exit_threshold=0.5,
        stop_loss=3.0,
        min_half_life=5,
        max_half_life=60,
        min_cost_ratio=5.0,
        transaction_cost=0.0001
    )
    
    enter_list, watch_list = trader.screen_pairs(prices, n_pairs=5)
    
    cache_data = {
        'enter_signals': enter_list,
        'watch_signals': watch_list,
        'parameters': {
            'formation_window': 252,
            'signal_window': 252,
            'enter_threshold': 2.0,
            'exit_threshold': 0.5,
            'stop_loss': 3.0,
            'min_half_life': 5,
            'max_half_life': 60,
            'min_cost_ratio': 5.0,
            'transaction_cost': 0.0001
        },
        'generated_at': datetime.now().isoformat(),
        'data_last_date': get_data_last_date()
    }
    
    # 캐시 디렉토리 생성
    cache_dir = "cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    # 캐시 저장
    cache_file = os.path.join(cache_dir, "ssd_default.pkl")
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f"✅ SSD 거리 캐시 저장 완료: {len(enter_list)}개 진입신호, {len(watch_list)}개 관찰대상")
    return len(enter_list), len(watch_list)

def generate_cointegration_cache():
    """공적분 방법론 캐시 생성"""
    print("🔄 공적분 방법론 캐시 생성 중...")
    
    file_path = "data/MU Price(BBG).csv"
    prices = common_utils.load_data(file_path)
    
    # 기본 파라미터로 분석
    trader = cointegration_module.CointegrationPairTrading(
        formation_window=252,
        signal_window=60,
        enter_threshold=2.0,
        exit_threshold=0.5,
        stop_loss=3.0,
        min_half_life=5,
        max_half_life=60,
        min_cost_ratio=5.0,
        max_pvalue=0.05
    )
    
    enter_list, watch_list = trader.screen_pairs(prices, n_pairs=5)
    
    cache_data = {
        'enter_signals': enter_list,
        'watch_signals': watch_list,
        'generated_at': datetime.now().isoformat(),
        'data_date': get_data_last_date(),
        'parameters': {
            'formation_window': 252,
            'signal_window': 60,
            'enter_threshold': 2.0,
            'exit_threshold': 0.5,
            'stop_loss': 3.0,
            'min_half_life': 5,
            'max_half_life': 60,
            'min_cost_ratio': 5.0,
            'max_pvalue': 0.05
        }
    }
    
    with open('cache/cointegration_default.pkl', 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f"✅ 공적분 캐시 완료: {len(enter_list)}개 진입 + {len(watch_list)}개 관찰")

def generate_regime_cache():
    """상관관계 레짐 방법론 캐시 생성"""
    print("🔄 상관관계 레짐 방법론 캐시 생성 중...")
    
    file_path = "data/MU Price(BBG).csv"
    prices = common_utils.load_data(file_path)
    
    trader = regime_module.CorrelationRegimePairTrading(
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
    
    enter_list, watch_list = trader.screen_pairs(prices, n_pairs=5)
    
    cache_data = {
        'enter_signals': enter_list,
        'watch_signals': watch_list,
        'generated_at': datetime.now().isoformat(),
        'data_date': get_data_last_date(),
        'parameters': {
            'formation_window': 252,
            'signal_window': 60,
            'long_corr_window': 252,
            'short_corr_window': 60,
            'enter_threshold': 2.0,
            'exit_threshold': 0.5,
            'stop_loss': 3.0,
            'min_half_life': 5,
            'max_half_life': 60,
            'min_cost_ratio': 5.0,
            'min_delta_corr': 0.3
        }
    }
    
    with open('cache/regime_default.pkl', 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f"✅ 레짐 캐시 완료: {len(enter_list)}개 진입 + {len(watch_list)}개 관찰")

def generate_ou_cache():
    """OU 평균회귀 방법론 캐시 생성"""
    print("🔄 OU 평균회귀 방법론 캐시 생성 중...")
    
    file_path = "data/MU Price(BBG).csv"
    prices = common_utils.load_data(file_path)
    
    trader = ou_module.OUMeanReversionPairTrading(
        formation_window=252,
        signal_window=60,
        enter_threshold=2.0,
        exit_threshold=0.5,
        stop_loss=3.0,
        min_half_life=5,
        max_half_life=60,
        min_cost_ratio=5.0,
        min_mean_reversion_speed=0.01
    )
    
    enter_list, watch_list = trader.screen_pairs(prices, n_pairs=5)
    
    cache_data = {
        'enter_signals': enter_list,
        'watch_signals': watch_list,
        'generated_at': datetime.now().isoformat(),
        'data_date': get_data_last_date(),
        'parameters': {
            'formation_window': 252,
            'signal_window': 60,
            'enter_threshold': 2.0,
            'exit_threshold': 0.5,
            'stop_loss': 3.0,
            'min_half_life': 5,
            'max_half_life': 60,
            'min_cost_ratio': 5.0,
            'min_mean_reversion_speed': 0.01
        }
    }
    
    with open('cache/ou_default.pkl', 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f"✅ OU 캐시 완료: {len(enter_list)}개 진입 + {len(watch_list)}개 관찰")

def generate_clustering_cache():
    """클러스터링 방법론 캐시 생성"""
    print("🔄 클러스터링 방법론 캐시 생성 중...")
    
    file_path = "data/MU Price(BBG).csv"
    prices = common_utils.load_data(file_path)
    
    trader = clustering_module.ClusteringPairTrading(
        formation_window=252,
        signal_window=60,
        enter_threshold=2.0,
        exit_threshold=0.5,
        stop_loss=3.0,
        min_half_life=5,
        max_half_life=60,
        min_cost_ratio=5.0,
        n_clusters=8,
        clustering_method='kmeans'
    )
    
    enter_list, watch_list = trader.screen_pairs(prices, n_pairs=5)
    
    # 클러스터 요약 정보도 추가
    cluster_summary = trader.get_cluster_summary(prices)
    
    cache_data = {
        'enter_signals': enter_list,
        'watch_signals': watch_list,
        'cluster_summary': cluster_summary,
        'generated_at': datetime.now().isoformat(),
        'data_date': get_data_last_date(),
        'parameters': {
            'formation_window': 252,
            'signal_window': 60,
            'enter_threshold': 2.0,
            'exit_threshold': 0.5,
            'stop_loss': 3.0,
            'min_half_life': 5,
            'max_half_life': 60,
            'min_cost_ratio': 5.0,
            'n_clusters': 8,
            'clustering_method': 'kmeans'
        }
    }
    
    with open('cache/clustering_default.pkl', 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f"✅ 클러스터링 캐시 완료: {len(enter_list)}개 진입 + {len(watch_list)}개 관찰")

def generate_copula_cache():
    """코퓰라 순위상관 방법론 캐시 생성"""
    print("🔄 코퓰라 순위상관 방법론 캐시 생성 중...")
    
    file_path = "data/MU Price(BBG).csv"
    prices = common_utils.load_data(file_path)
    
    trader = copula_module.CopulaRankCorrelationPairTrading(
        formation_window=252,
        signal_window=60,
        long_window=252,
        short_window=60,
        enter_threshold=2.0,
        exit_threshold=0.5,
        stop_loss=3.0,
        min_half_life=5,
        max_half_life=60,
        min_cost_ratio=5.0,
        min_rank_corr=0.3,
        min_rank_corr_change=0.2,
        tail_quantile=0.1
    )
    
    enter_list, watch_list = trader.screen_pairs(prices, n_pairs=5)
    
    cache_data = {
        'enter_signals': enter_list,
        'watch_signals': watch_list,
        'generated_at': datetime.now().isoformat(),
        'data_date': get_data_last_date(),
        'parameters': {
            'formation_window': 252,
            'signal_window': 60,
            'long_window': 252,
            'short_window': 60,
            'enter_threshold': 2.0,
            'exit_threshold': 0.5,
            'stop_loss': 3.0,
            'min_half_life': 5,
            'max_half_life': 60,
            'min_cost_ratio': 5.0,
            'min_rank_corr': 0.3,
            'min_rank_corr_change': 0.2,
            'tail_quantile': 0.1
        }
    }
    
    with open('cache/copula_default.pkl', 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f"✅ 코퓰라 캐시 완료: {len(enter_list)}개 진입 + {len(watch_list)}개 관찰")

def main():
    """모든 방법론 캐시 생성"""
    print("=" * 60)
    print("🚀 페어트레이딩 방법론별 캐시 생성 시작")
    print("=" * 60)
    
    data_date = get_data_last_date()
    print(f"📅 데이터 기준일: {data_date}")
    
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
        print(f"캐시 위치: {os.path.abspath('cache')}")
        
    except Exception as e:
        print(f"❌ 캐시 생성 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    main()