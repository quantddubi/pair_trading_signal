"""
캐시 관리 유틸리티 함수들
"""
import os
import pickle
from datetime import datetime
from typing import Dict, Optional, Tuple, List

def load_cache(method_name: str, parameters: Dict = None) -> Optional[Dict]:
    """
    캐시 파일 로드
    
    Args:
        method_name: 방법론 이름 (cointegration, regime, ou, clustering, copula)
        parameters: 파라미터 딕셔너리 (기본값과 비교용)
        
    Returns:
        캐시 데이터 또는 None
    """
    # 절대 경로로 캐시 파일 지정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    cache_file = os.path.join(project_root, "cache", f"{method_name}_default.pkl")
    
    if not os.path.exists(cache_file):
        return None
    
    try:
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        # 파라미터가 제공된 경우 기본값과 비교
        if parameters and 'parameters' in cache_data:
            if parameters != cache_data['parameters']:
                # 파라미터가 다르면 캐시를 사용하지 않음
                return None
        
        return cache_data
        
    except Exception as e:
        print(f"캐시 로드 오류: {str(e)}")
        return None

def is_cache_valid(method_name: str, max_age_hours: int = 24) -> bool:
    """
    캐시가 유효한지 확인 (생성 시간 기준)
    
    Args:
        method_name: 방법론 이름
        max_age_hours: 최대 허용 나이 (시간)
        
    Returns:
        캐시 유효성 여부
    """
    cache_data = load_cache(method_name)
    
    if not cache_data or 'generated_at' not in cache_data:
        return False
    
    try:
        generated_at = datetime.fromisoformat(cache_data['generated_at'])
        age_hours = (datetime.now() - generated_at).total_seconds() / 3600
        
        return age_hours <= max_age_hours
        
    except Exception:
        return False

def get_default_parameters(method_name: str) -> Dict:
    """각 방법론의 기본 파라미터 반환"""
    
    defaults = {
        'euclidean': {
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
        'ssd': {
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
        'cointegration': {
            'formation_window': 252,
            'signal_window': 60,
            'enter_threshold': 2.0,
            'exit_threshold': 0.5,
            'stop_loss': 3.0,
            'min_half_life': 5,
            'max_half_life': 60,
            'min_cost_ratio': 5.0,
            'max_pvalue': 0.05
        },
        'regime': {
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
        },
        'ou': {
            'formation_window': 252,
            'signal_window': 60,
            'enter_threshold': 2.0,
            'exit_threshold': 0.5,
            'stop_loss': 3.0,
            'min_half_life': 5,
            'max_half_life': 60,
            'min_cost_ratio': 5.0,
            'min_mean_reversion_speed': 0.01
        },
        'clustering': {
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
        },
        'copula': {
            'formation_window': 252,
            'signal_window': 60,
            'enter_threshold': 2.0,
            'exit_threshold': 0.5,
            'stop_loss': 3.0,
            'min_half_life': 5,
            'max_half_life': 60,
            'min_cost_ratio': 5.0,
            'min_rank_correlation': 0.7,
            'min_tail_dependence': 0.2
        }
    }
    
    return defaults.get(method_name, {})

def parameters_match_default(method_name: str, user_params: Dict) -> bool:
    """사용자 파라미터가 기본값과 일치하는지 확인"""
    default_params = get_default_parameters(method_name)
    
    # 모든 기본 파라미터가 사용자 파라미터와 일치하는지 확인
    for key, default_value in default_params.items():
        if user_params.get(key) != default_value:
            return False
    
    return True

def get_cache_info() -> Dict:
    """모든 캐시 파일의 정보 반환"""
    cache_info = {}
    methods = ['euclidean', 'ssd', 'cointegration', 'regime', 'ou', 'clustering', 'copula']
    
    for method in methods:
        try:
            cache_data = load_cache(method)
            if cache_data:
                cache_info[method] = {
                    'exists': True,
                    'generated_at': cache_data.get('generated_at'),
                    'data_date': cache_data.get('data_date'),
                    'enter_count': len(cache_data.get('enter_signals', [])),
                    'watch_count': len(cache_data.get('watch_signals', []))
                }
            else:
                cache_info[method] = {
                    'exists': False
                }
        except Exception as e:
            print(f"Error loading cache info for {method}: {str(e)}")
            cache_info[method] = {
                'exists': False
            }
    
    return cache_info