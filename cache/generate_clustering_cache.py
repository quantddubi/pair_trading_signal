#!/usr/bin/env python3
"""
Clustering 방법론 캐시 생성 스크립트
"""
import os
import sys
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 루트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# 필요한 모듈 import
from config.default_parameters import get_default_parameters
import importlib.util

def import_module_from_file(file_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# 공통 유틸리티 및 방법론 모듈
common_utils = import_module_from_file(os.path.join(project_root, "utils/common_utils.py"), "common_utils")
clustering_module = import_module_from_file(os.path.join(project_root, "methods/6_clustering_pairs.py"), "clustering_pairs")

def get_data_last_date():
    """데이터의 마지막 날짜 확인"""
    file_path = os.path.join(project_root, "data/MU Price(BBG).csv")
    prices = common_utils.load_data(file_path)
    return prices.index[-1].strftime('%Y%m%d')

def main():
    """Clustering 방법론 캐시 생성"""
    print("=" * 60)
    print("Clustering 방법론 캐시 생성")
    print("=" * 60)

    data_date = get_data_last_date()
    print(f"데이터 기준일: {data_date}")
    print()

    try:
        print("클러스터링 방법론 캐시 생성 중...")

        file_path = os.path.join(project_root, "data/MU Price(BBG).csv")
        prices = common_utils.load_data(file_path)

        # config에서 기본 파라미터 가져오기
        params = get_default_parameters('clustering')
        print(f"  파라미터: formation_window={params['formation_window']}, n_clusters={params['n_clusters']}")

        # 파라미터로 분석
        trader = clustering_module.ClusteringPairTrading(**params)
        enter_list, watch_list = trader.screen_pairs(prices, n_pairs=5)

        # 클러스터 요약 정보도 추가
        cluster_summary = trader.get_cluster_summary(prices)

        cache_data = {
            'enter_signals': enter_list,
            'watch_signals': watch_list,
            'cluster_summary': cluster_summary,
            'generated_at': datetime.now().isoformat(),
            'data_date': data_date,
            'parameters': params
        }

        # 캐시 데이터 디렉토리 생성
        cache_data_dir = os.path.join(current_dir, "data")
        if not os.path.exists(cache_data_dir):
            os.makedirs(cache_data_dir)

        # 캐시 저장
        cache_file = os.path.join(cache_data_dir, "clustering_default.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)

        print(f"완료: 클러스터링 캐시 저장 완료")
        print(f"  - 진입신호: {len(enter_list)}개")
        print(f"  - 관찰대상: {len(watch_list)}개")
        print(f"  - 저장위치: {cache_file}")
        print()
        print("=" * 60)
        print("[SUCCESS] 캐시 생성 완료!")
        print("=" * 60)

    except Exception as e:
        print(f"[ERROR] 캐시 생성 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
