"""
상관관계 레짐 방법론 전용 캐시 생성 스크립트
"""
import os
import sys
import pickle
from datetime import datetime
import importlib.util

def import_module_from_file(file_path, module_name):
    """동적 모듈 import"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def main():
    print("🔍 상관관계 레짐 방법론 캐시 생성")
    print("=" * 50)
    
    start_time = datetime.now()
    
    try:
        # 모듈 import
        common_utils = import_module_from_file("utils/common_utils.py", "common_utils")
        cache_utils = import_module_from_file("utils/cache_utils.py", "cache_utils")
        regime_module = import_module_from_file("methods/4_correlation_regime_pairs.py", "correlation_regime_pairs")
        
        # 데이터 로딩
        file_path = "data/MU Price(BBG).csv"
        prices = common_utils.load_data(file_path)
        print(f"✅ 데이터 로딩 완료: {len(prices.columns)}개 자산, {len(prices)}일")
        
        # 기본 파라미터로 분석 실행
        default_params = cache_utils.get_default_parameters('regime')
        print(f"📊 사용 파라미터:")
        for key, value in default_params.items():
            print(f"   {key}: {value}")
        
        trader = regime_module.CorrelationRegimePairTrading(
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
        
        print(f"🔄 페어 분석 실행 중...")
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
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"✅ 상관관계 레짐 캐시 생성 완료!")
        print(f"📁 파일: {cache_file}")
        print(f"📈 진입 신호: {len(enter_list)}개")
        print(f"👀 관찰 대상: {len(watch_list)}개")
        print(f"⏱️ 소요 시간: {duration.total_seconds():.1f}초")
        
        # 진입 신호 상세 정보 표시
        if enter_list:
            print(f"\n📈 진입 신호 상세:")
            for i, signal in enumerate(enter_list[:5], 1):  # 상위 5개만 표시
                print(f"   {i}. {signal['pair']}: Z={signal['current_zscore']:.2f}, ΔCorr={signal.get('current_delta_corr', 0):.3f}")
        
        if watch_list:
            print(f"\n👀 관찰 대상 상세:")
            for i, signal in enumerate(watch_list[:5], 1):  # 상위 5개만 표시
                print(f"   {i}. {signal['pair']}: Z={signal['current_zscore']:.2f}, ΔCorr={signal.get('current_delta_corr', 0):.3f}")
        
    except Exception as e:
        print(f"❌ 상관관계 레짐 캐시 생성 실패: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()