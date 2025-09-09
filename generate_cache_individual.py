"""
개별 방법론 캐시 생성 스크립트
각 방법론별로 독립적으로 캐시를 생성할 수 있음
"""
import os
import sys
import pickle
import argparse
from datetime import datetime
import importlib.util

def import_module_from_file(file_path, module_name):
    """동적 모듈 import"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# 공통 유틸리티 import
common_utils = import_module_from_file("utils/common_utils.py", "common_utils")
cache_utils = import_module_from_file("utils/cache_utils.py", "cache_utils")

def generate_euclidean_cache():
    """유클리드 거리 방법론 캐시 생성"""
    print("🔍 유클리드 거리 분석 캐시 생성 시작...")
    
    try:
        # 모듈 import
        euclidean_module = import_module_from_file("methods/1_euclidean_distance_pairs.py", "euclidean_pairs")
        EuclideanDistancePairTrading = euclidean_module.EuclideanDistancePairTrading
        
        # 데이터 로딩
        file_path = "data/MU Price(BBG).csv"
        prices = common_utils.load_data(file_path)
        print(f"✅ 데이터 로딩 완료: {len(prices.columns)}개 자산, {len(prices)}일")
        
        # 기본 파라미터로 분석 실행
        default_params = cache_utils.get_default_parameters('euclidean')
        
        trader = EuclideanDistancePairTrading(
            formation_window=default_params['formation_window'],
            signal_window=default_params['signal_window'], 
            enter_threshold=default_params['enter_threshold'],
            exit_threshold=default_params['exit_threshold'],
            stop_loss=default_params['stop_loss'],
            min_half_life=default_params['min_half_life'],
            max_half_life=default_params['max_half_life'],
            min_cost_ratio=default_params['min_cost_ratio'],
            transaction_cost=default_params['transaction_cost']
        )
        
        enter_list, watch_list = trader.screen_pairs(prices, n_pairs=20)
        
        # 캐시 데이터 생성
        cache_data = {
            'generated_at': datetime.now().isoformat(),
            'data_date': prices.index[-1].isoformat(),
            'parameters': default_params,
            'enter_signals': enter_list,
            'watch_signals': watch_list,
            'method': 'euclidean'
        }
        
        # 캐시 파일 저장
        cache_dir = "cache"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "euclidean_default.pkl")
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"✅ 유클리드 거리 캐시 생성 완료!")
        print(f"📁 파일: {cache_file}")
        print(f"📈 진입 신호: {len(enter_list)}개, 관찰 대상: {len(watch_list)}개")
        
    except Exception as e:
        print(f"❌ 유클리드 거리 캐시 생성 실패: {str(e)}")

def generate_ssd_cache():
    """SSD 방법론 캐시 생성"""
    print("🔍 SSD 분석 캐시 생성 시작...")
    
    try:
        # 모듈 import  
        ssd_module = import_module_from_file("methods/2_ssd_pairs.py", "ssd_pairs")
        SSDPairTrading = ssd_module.SSDPairTrading
        
        # 데이터 로딩
        file_path = "data/MU Price(BBG).csv"
        prices = common_utils.load_data(file_path)
        print(f"✅ 데이터 로딩 완료: {len(prices.columns)}개 자산, {len(prices)}일")
        
        # 기본 파라미터로 분석 실행
        default_params = cache_utils.get_default_parameters('ssd')
        
        trader = SSDPairTrading(
            formation_window=default_params['formation_window'],
            signal_window=default_params['signal_window'],
            enter_threshold=default_params['enter_threshold'],
            exit_threshold=default_params['exit_threshold'], 
            stop_loss=default_params['stop_loss'],
            min_half_life=default_params['min_half_life'],
            max_half_life=default_params['max_half_life'],
            min_cost_ratio=default_params['min_cost_ratio'],
            transaction_cost=default_params['transaction_cost']
        )
        
        enter_list, watch_list = trader.screen_pairs(prices, n_pairs=20)
        
        # 캐시 데이터 생성
        cache_data = {
            'generated_at': datetime.now().isoformat(),
            'data_date': prices.index[-1].isoformat(),
            'parameters': default_params,
            'enter_signals': enter_list,
            'watch_signals': watch_list,
            'method': 'ssd'
        }
        
        # 캐시 파일 저장
        cache_dir = "cache"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "ssd_default.pkl")
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"✅ SSD 캐시 생성 완료!")
        print(f"📁 파일: {cache_file}")
        print(f"📈 진입 신호: {len(enter_list)}개, 관찰 대상: {len(watch_list)}개")
        
    except Exception as e:
        print(f"❌ SSD 캐시 생성 실패: {str(e)}")

def generate_cointegration_cache():
    """공적분 방법론 캐시 생성"""
    print("🔍 공적분 분석 캐시 생성 시작...")
    
    try:
        # 모듈 import
        cointegration_module = import_module_from_file("methods/3_cointegration_pairs.py", "cointegration_pairs")
        CointegrationPairTrading = cointegration_module.CointegrationPairTrading
        
        # 데이터 로딩
        file_path = "data/MU Price(BBG).csv"
        prices = common_utils.load_data(file_path)
        print(f"✅ 데이터 로딩 완료: {len(prices.columns)}개 자산, {len(prices)}일")
        
        # 기본 파라미터로 분석 실행
        default_params = cache_utils.get_default_parameters('cointegration')
        
        trader = CointegrationPairTrading(
            formation_window=default_params['formation_window'],
            signal_window=default_params['signal_window'],
            enter_threshold=default_params['enter_threshold'],
            exit_threshold=default_params['exit_threshold'],
            stop_loss=default_params['stop_loss'],
            min_half_life=default_params['min_half_life'],
            max_half_life=default_params['max_half_life'],
            min_cost_ratio=default_params['min_cost_ratio'],
            max_pvalue=default_params['max_pvalue']
        )
        
        enter_list, watch_list = trader.screen_pairs(prices, n_pairs=20)
        
        # 캐시 데이터 생성
        cache_data = {
            'generated_at': datetime.now().isoformat(),
            'data_date': prices.index[-1].isoformat(),
            'parameters': default_params,
            'enter_signals': enter_list,
            'watch_signals': watch_list,
            'method': 'cointegration'
        }
        
        # 캐시 파일 저장
        cache_dir = "cache"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "cointegration_default.pkl")
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"✅ 공적분 캐시 생성 완료!")
        print(f"📁 파일: {cache_file}")
        print(f"📈 진입 신호: {len(enter_list)}개, 관찰 대상: {len(watch_list)}개")
        
    except Exception as e:
        print(f"❌ 공적분 캐시 생성 실패: {str(e)}")

def generate_regime_cache():
    """상관관계 레짐 방법론 캐시 생성"""
    print("🔍 상관관계 레짐 분석 캐시 생성 시작...")
    
    try:
        # 모듈 import
        regime_module = import_module_from_file("methods/4_correlation_regime_pairs.py", "correlation_regime_pairs")
        CorrelationRegimePairTrading = regime_module.CorrelationRegimePairTrading
        
        # 데이터 로딩
        file_path = "data/MU Price(BBG).csv"
        prices = common_utils.load_data(file_path)
        print(f"✅ 데이터 로딩 완료: {len(prices.columns)}개 자산, {len(prices)}일")
        
        # 기본 파라미터로 분석 실행
        default_params = cache_utils.get_default_parameters('regime')
        
        trader = CorrelationRegimePairTrading(
            formation_window=default_params['formation_window'],
            signal_window=default_params['signal_window'],
            long_corr_window=default_params['long_corr_window'],
            short_corr_window=default_params['short_corr_window'],
            enter_threshold=default_params['enter_threshold'],
            exit_threshold=default_params['exit_threshold'],
            stop_loss=default_params['stop_loss'],
            min_half_life=default_params['min_half_life'],
            max_half_life=default_params['max_half_life'],
            min_cost_ratio=default_params['min_cost_ratio'],
            min_delta_corr=default_params['min_delta_corr']
        )
        
        enter_list, watch_list = trader.screen_pairs(prices, n_pairs=20)
        
        # 캐시 데이터 생성
        cache_data = {
            'generated_at': datetime.now().isoformat(),
            'data_date': prices.index[-1].isoformat(),
            'parameters': default_params,
            'enter_signals': enter_list,
            'watch_signals': watch_list,
            'method': 'regime'
        }
        
        # 캐시 파일 저장
        cache_dir = "cache"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "regime_default.pkl")
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"✅ 상관관계 레짐 캐시 생성 완료!")
        print(f"📁 파일: {cache_file}")
        print(f"📈 진입 신호: {len(enter_list)}개, 관찰 대상: {len(watch_list)}개")
        
    except Exception as e:
        print(f"❌ 상관관계 레짐 캐시 생성 실패: {str(e)}")

def generate_ou_cache():
    """OU 평균회귀 방법론 캐시 생성"""
    print("🔍 OU 평균회귀 분석 캐시 생성 시작...")
    
    try:
        # 모듈 import
        ou_module = import_module_from_file("methods/5_ou_mean_reversion_pairs.py", "ou_pairs")
        OUMeanReversionPairTrading = ou_module.OUMeanReversionPairTrading
        
        # 데이터 로딩
        file_path = "data/MU Price(BBG).csv"
        prices = common_utils.load_data(file_path)
        print(f"✅ 데이터 로딩 완료: {len(prices.columns)}개 자산, {len(prices)}일")
        
        # 기본 파라미터로 분석 실행
        default_params = cache_utils.get_default_parameters('ou')
        
        trader = OUMeanReversionPairTrading(
            formation_window=default_params['formation_window'],
            signal_window=default_params['signal_window'],
            enter_threshold=default_params['enter_threshold'],
            exit_threshold=default_params['exit_threshold'],
            stop_loss=default_params['stop_loss'],
            min_half_life=default_params['min_half_life'],
            max_half_life=default_params['max_half_life'],
            min_cost_ratio=default_params['min_cost_ratio'],
            min_mean_reversion_speed=default_params['min_mean_reversion_speed']
        )
        
        enter_list, watch_list = trader.screen_pairs(prices, n_pairs=20)
        
        # 캐시 데이터 생성
        cache_data = {
            'generated_at': datetime.now().isoformat(),
            'data_date': prices.index[-1].isoformat(),
            'parameters': default_params,
            'enter_signals': enter_list,
            'watch_signals': watch_list,
            'method': 'ou'
        }
        
        # 캐시 파일 저장
        cache_dir = "cache"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "ou_default.pkl")
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"✅ OU 평균회귀 캐시 생성 완료!")
        print(f"📁 파일: {cache_file}")
        print(f"📈 진입 신호: {len(enter_list)}개, 관찰 대상: {len(watch_list)}개")
        
    except Exception as e:
        print(f"❌ OU 평균회귀 캐시 생성 실패: {str(e)}")

def generate_clustering_cache():
    """클러스터링 방법론 캐시 생성"""
    print("🔍 클러스터링 분석 캐시 생성 시작...")
    
    try:
        # 모듈 import
        clustering_module = import_module_from_file("methods/6_clustering_pairs.py", "clustering_pairs")
        ClusteringPairTrading = clustering_module.ClusteringPairTrading
        
        # 데이터 로딩
        file_path = "data/MU Price(BBG).csv"
        prices = common_utils.load_data(file_path)
        print(f"✅ 데이터 로딩 완료: {len(prices.columns)}개 자산, {len(prices)}일")
        
        # 기본 파라미터로 분석 실행
        default_params = cache_utils.get_default_parameters('clustering')
        
        trader = ClusteringPairTrading(
            formation_window=default_params['formation_window'],
            signal_window=default_params['signal_window'],
            enter_threshold=default_params['enter_threshold'],
            exit_threshold=default_params['exit_threshold'],
            stop_loss=default_params['stop_loss'],
            min_half_life=default_params['min_half_life'],
            max_half_life=default_params['max_half_life'],
            min_cost_ratio=default_params['min_cost_ratio'],
            n_clusters=default_params['n_clusters'],
            clustering_method=default_params['clustering_method']
        )
        
        enter_list, watch_list = trader.screen_pairs(prices, n_pairs=20)
        
        # 캐시 데이터 생성
        cache_data = {
            'generated_at': datetime.now().isoformat(),
            'data_date': prices.index[-1].isoformat(),
            'parameters': default_params,
            'enter_signals': enter_list,
            'watch_signals': watch_list,
            'method': 'clustering'
        }
        
        # 캐시 파일 저장
        cache_dir = "cache"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "clustering_default.pkl")
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"✅ 클러스터링 캐시 생성 완료!")
        print(f"📁 파일: {cache_file}")
        print(f"📈 진입 신호: {len(enter_list)}개, 관찰 대상: {len(watch_list)}개")
        
    except Exception as e:
        print(f"❌ 클러스터링 캐시 생성 실패: {str(e)}")

def generate_copula_cache():
    """코퓰라 방법론 캐시 생성"""
    print("🔍 코퓰라 분석 캐시 생성 시작...")
    
    try:
        # 모듈 import
        copula_module = import_module_from_file("methods/7_copula_rank_correlation_pairs.py", "copula_pairs")
        CopulaBasedPairScreening = copula_module.CopulaBasedPairScreening
        
        # 데이터 로딩
        file_path = "data/MU Price(BBG).csv"
        prices = common_utils.load_data(file_path)
        print(f"✅ 데이터 로딩 완료: {len(prices.columns)}개 자산, {len(prices)}일")
        
        # 기본 파라미터로 분석 실행
        default_params = cache_utils.get_default_parameters('copula')
        
        # 새로운 관대한 설정으로 CopulaBasedPairScreening 사용
        screener = CopulaBasedPairScreening()  # 기본값 사용 (이미 관대하게 설정됨)
        
        enter_list = screener.select_pairs(prices, n_pairs=20)
        watch_list = []  # CopulaBasedPairScreening에서는 watch_list 분리 로직이 다름
        
        # 캐시 데이터 생성
        cache_data = {
            'generated_at': datetime.now().isoformat(),
            'data_date': prices.index[-1].isoformat(),
            'parameters': default_params,
            'enter_signals': enter_list,
            'watch_signals': watch_list,
            'method': 'copula'
        }
        
        # 캐시 파일 저장
        cache_dir = "cache"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "copula_default.pkl")
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"✅ 코퓰라 캐시 생성 완료!")
        print(f"📁 파일: {cache_file}")
        print(f"📈 진입 신호: {len(enter_list)}개, 관찰 대상: {len(watch_list)}개")
        
    except Exception as e:
        print(f"❌ 코퓰라 캐시 생성 실패: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='개별 방법론 캐시 생성')
    parser.add_argument('method', nargs='?', 
                       choices=['euclidean', 'ssd', 'cointegration', 'regime', 'ou', 'clustering', 'copula', 'all'],
                       help='생성할 캐시 방법론 (all: 모든 방법론)')
    
    args = parser.parse_args()
    
    if not args.method:
        print("🔥 개별 방법론 캐시 생성 도구")
        print("-" * 40)
        print("사용법: python generate_cache_individual.py [METHOD]")
        print()
        print("METHOD 옵션:")
        print("  euclidean     - 유클리드 거리 방법론")
        print("  ssd           - SSD 방법론") 
        print("  cointegration - 공적분 방법론")
        print("  regime        - 상관관계 레짐 방법론")
        print("  ou            - OU 평균회귀 방법론")
        print("  clustering    - 클러스터링 방법론")
        print("  copula        - 코퓰라 방법론")
        print("  all           - 모든 방법론 (순차 실행)")
        print()
        print("예시:")
        print("  python generate_cache_individual.py regime")
        print("  python generate_cache_individual.py all")
        return
    
    print(f"🚀 {args.method} 방법론 캐시 생성 시작")
    print("=" * 50)
    
    start_time = datetime.now()
    
    if args.method == 'euclidean':
        generate_euclidean_cache()
    elif args.method == 'ssd':
        generate_ssd_cache()
    elif args.method == 'cointegration':
        generate_cointegration_cache()
    elif args.method == 'regime':
        generate_regime_cache()
    elif args.method == 'ou':
        generate_ou_cache()
    elif args.method == 'clustering':
        generate_clustering_cache()
    elif args.method == 'copula':
        generate_copula_cache()
    elif args.method == 'all':
        print("📦 모든 방법론 캐시 순차 생성...")
        methods = ['euclidean', 'ssd', 'cointegration', 'regime', 'ou', 'clustering', 'copula']
        
        for i, method in enumerate(methods, 1):
            print(f"\n{'='*50}")
            print(f"🔄 {i}/{len(methods)}: {method.upper()} 방법론")
            print(f"{'='*50}")
            
            method_start = datetime.now()
            
            if method == 'euclidean':
                generate_euclidean_cache()
            elif method == 'ssd':
                generate_ssd_cache()
            elif method == 'cointegration':
                generate_cointegration_cache()
            elif method == 'regime':
                generate_regime_cache()
            elif method == 'ou':
                generate_ou_cache()
            elif method == 'clustering':
                generate_clustering_cache()
            elif method == 'copula':
                generate_copula_cache()
            
            method_end = datetime.now()
            method_duration = method_end - method_start
            print(f"⏱️ {method} 완료 시간: {method_duration.total_seconds():.1f}초")
    
    end_time = datetime.now()
    total_duration = end_time - start_time
    
    print(f"\n🎉 전체 작업 완료!")
    print(f"⏱️ 총 소요 시간: {total_duration.total_seconds():.1f}초")
    
    # 생성된 캐시 파일 정보 표시
    cache_info = cache_utils.get_cache_info()
    print(f"\n📊 생성된 캐시 파일 현황:")
    for method, info in cache_info.items():
        if info['exists']:
            print(f"  ✅ {method}: {info['enter_count']}개 진입, {info['watch_count']}개 관찰")
        else:
            print(f"  ❌ {method}: 파일 없음")

if __name__ == "__main__":
    main()