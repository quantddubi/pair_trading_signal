"""
캐시 관리 유틸리티 함수들
"""
import os
import sys
import pickle
from datetime import datetime
from typing import Dict, Optional, Tuple, List

# config 모듈 import
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from config.default_parameters import get_default_parameters as get_config_params

def load_cache(method_name: str, parameters: Dict = None) -> Optional[Dict]:
    """
    캐시 파일 로드
    
    Args:
        method_name: 방법론 이름 (cointegration, regime, ou, clustering, copula)
        parameters: 파라미터 딕셔너리 (기본값과 비교용)
        
    Returns:
        캐시 데이터 또는 None
    """
    # 절대 경로로 캐시 파일 지정 (cache/data/ 하위 폴더)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    cache_file = os.path.join(project_root, "cache", "data", f"{method_name}_default.pkl")
    
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
    """
    각 방법론의 기본 파라미터 반환

    이제 config.default_parameters에서 중앙화된 파라미터를 가져옵니다.
    Single Source of Truth를 유지합니다.

    Args:
        method_name: 방법론 이름

    Returns:
        해당 방법론의 기본 파라미터 딕셔너리
    """
    return get_config_params(method_name)

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