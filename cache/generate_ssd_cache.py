#!/usr/bin/env python3
"""
SSD Distance 방법론 캐시 생성 스크립트
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
ssd_module = import_module_from_file(os.path.join(project_root, "methods/2_ssd_distance_pairs.py"), "ssd_distance_pairs")

def get_data_last_date():
    """데이터의 마지막 날짜 확인"""
    file_path = os.path.join(project_root, "data/MU Price(BBG).csv")
    prices = common_utils.load_data(file_path)
    return prices.index[-1].strftime('%Y%m%d')

def main():
    """SSD Distance 방법론 캐시 생성"""
    print("=" * 60)
    print("SSD Distance 방법론 캐시 생성")
    print("=" * 60)

    data_date = get_data_last_date()
    print(f"데이터 기준일: {data_date}")
    print()

    try:
        print("SSD 거리 방법론 캐시 생성 중...")

        file_path = os.path.join(project_root, "data/MU Price(BBG).csv")
        prices = common_utils.load_data(file_path)

        # config에서 기본 파라미터 가져오기
        params = get_default_parameters('ssd')
        print(f"  파라미터: formation_window={params['formation_window']}, signal_window={params['signal_window']}")

        # 파라미터로 분석
        trader = ssd_module.SSDDistancePairTrading(**params)
        enter_list, watch_list = trader.screen_pairs(prices, n_pairs=5)

        cache_data = {
            'enter_signals': enter_list,
            'watch_signals': watch_list,
            'parameters': params,
            'generated_at': datetime.now().isoformat(),
            'data_last_date': data_date
        }

        # 캐시 데이터 디렉토리 생성
        cache_data_dir = os.path.join(current_dir, "data")
        if not os.path.exists(cache_data_dir):
            os.makedirs(cache_data_dir)

        # 캐시 저장
        cache_file = os.path.join(cache_data_dir, "ssd_default.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)

        print(f"완료: SSD 거리 캐시 저장 완료")
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
