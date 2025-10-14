#!/usr/bin/env python3
"""
파라미터 일관성 자동 테스트

generate_cache.py와 cache_utils.py가 모두 config/default_parameters.py를 사용하는지 검증합니다.
Single Source of Truth가 유지되고 있는지 확인합니다.
"""
import os
import sys
import pytest

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config.default_parameters import get_default_parameters, DEFAULT_PARAMETERS
from utils.cache_utils import get_default_parameters as cache_get_params


class TestParameterConsistency:
    """파라미터 일관성 테스트"""

    METHODS = ['euclidean', 'ssd', 'cointegration', 'regime', 'ou', 'clustering', 'copula']

    def test_config_has_all_methods(self):
        """config에 모든 방법론의 파라미터가 정의되어 있는지 확인"""
        for method in self.METHODS:
            assert method in DEFAULT_PARAMETERS, f"{method} 파라미터가 config에 정의되지 않음"
            assert len(DEFAULT_PARAMETERS[method]) > 0, f"{method} 파라미터가 비어있음"

    def test_cache_utils_uses_config(self):
        """cache_utils.py가 config를 사용하는지 확인"""
        for method in self.METHODS:
            config_params = get_default_parameters(method)
            cache_params = cache_get_params(method)

            assert config_params == cache_params, \
                f"{method}: cache_utils 파라미터가 config와 다름\nconfig: {config_params}\ncache_utils: {cache_params}"

    def test_no_hardcoded_parameters_in_cache_utils(self):
        """cache_utils.py에 하드코딩된 파라미터가 없는지 확인"""
        cache_utils_path = os.path.join(project_root, "utils", "cache_utils.py")

        with open(cache_utils_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # get_default_parameters 함수가 config를 import하는지 확인
        assert "from config.default_parameters import" in content or \
               "get_default_parameters as get_config_params" in content, \
               "cache_utils.py가 config를 import하지 않음"

        # get_default_parameters 함수 내에 하드코딩된 파라미터 딕셔너리가 없는지 확인
        function_start = content.find("def get_default_parameters(")
        if function_start != -1:
            # 다음 함수 시작까지 또는 파일 끝까지
            next_def = content.find("\ndef ", function_start + 1)
            if next_def == -1:
                function_body = content[function_start:]
            else:
                function_body = content[function_start:next_def]

            # 함수 내에 formation_window 같은 파라미터 키워드가 직접 정의되어 있지 않은지 확인
            # (단, 주석은 제외)
            lines = function_body.split('\n')
            for line in lines:
                if '#' in line:
                    line = line[:line.index('#')]  # 주석 제거

                # 'formation_window': 같은 패턴이 있으면 하드코딩
                if "'formation_window':" in line or '"formation_window":' in line:
                    pytest.fail("cache_utils.py의 get_default_parameters에 하드코딩된 파라미터 발견")

    def test_generate_cache_imports_config(self):
        """generate_cache.py가 config를 import하는지 확인"""
        generate_cache_path = os.path.join(project_root, "cache", "generate_cache.py")

        with open(generate_cache_path, 'r', encoding='utf-8') as f:
            content = f.read()

        assert "from config.default_parameters import get_default_parameters" in content, \
               "generate_cache.py가 config를 import하지 않음"

    def test_generate_cache_uses_config_in_functions(self):
        """generate_cache.py의 각 함수가 config를 사용하는지 확인"""
        generate_cache_path = os.path.join(project_root, "cache", "generate_cache.py")

        with open(generate_cache_path, 'r', encoding='utf-8') as f:
            content = f.read()

        for method in self.METHODS:
            # 각 generate 함수 찾기
            function_pattern = f"def generate_{method}_cache():"
            assert function_pattern in content, f"{method}의 generate 함수를 찾을 수 없음"

            # 해당 함수 내에서 get_default_parameters 사용 확인
            func_start = content.find(function_pattern)
            next_func = content.find("\ndef ", func_start + 1)
            if next_func == -1:
                func_body = content[func_start:]
            else:
                func_body = content[func_start:next_func]

            assert "get_default_parameters(" in func_body, \
                   f"generate_{method}_cache()가 get_default_parameters()를 호출하지 않음"
            assert f"get_default_parameters('{method}')" in func_body, \
                   f"generate_{method}_cache()가 올바른 method name으로 get_default_parameters()를 호출하지 않음"

    def test_parameter_types(self):
        """모든 파라미터가 올바른 타입인지 확인"""
        for method in self.METHODS:
            params = get_default_parameters(method)

            for key, value in params.items():
                # 파라미터 값은 숫자(int/float) 또는 문자열이어야 함
                assert isinstance(value, (int, float, str)), \
                       f"{method}.{key}의 타입이 올바르지 않음: {type(value)}"

    def test_common_parameters_exist(self):
        """공통 파라미터가 모든 방법론에 존재하는지 확인"""
        # 대부분의 방법론이 가져야 할 공통 파라미터
        common_params = ['formation_window']

        for method in self.METHODS:
            params = get_default_parameters(method)

            for common_param in common_params:
                if method not in ['copula']:  # copula는 구조가 다를 수 있음
                    assert common_param in params, \
                           f"{method}에 공통 파라미터 {common_param}가 없음"

    def test_parameter_values_reasonable(self):
        """파라미터 값이 합리적인 범위인지 확인"""
        for method in self.METHODS:
            params = get_default_parameters(method)

            # formation_window는 양수여야 함
            if 'formation_window' in params:
                assert params['formation_window'] > 0, \
                       f"{method}.formation_window가 양수가 아님"

            # threshold 값들은 0보다 커야 함
            for key in params:
                if 'threshold' in key.lower() and isinstance(params[key], (int, float)):
                    assert params[key] >= 0, \
                           f"{method}.{key}가 음수임"

            # half_life는 min < max 관계여야 함
            if 'min_half_life' in params and 'max_half_life' in params:
                assert params['min_half_life'] < params['max_half_life'], \
                       f"{method}에서 min_half_life >= max_half_life"


class TestCacheFileConsistency:
    """생성된 캐시 파일의 파라미터 일관성 테스트"""

    def test_cache_files_use_config_parameters(self):
        """생성된 캐시 파일이 config의 파라미터를 사용했는지 확인 (캐시가 있는 경우만)"""
        import pickle

        cache_data_dir = os.path.join(project_root, "cache", "data")
        if not os.path.exists(cache_data_dir):
            pytest.skip("캐시 데이터 디렉토리가 없음")

        methods = ['euclidean', 'ssd', 'cointegration', 'regime', 'ou', 'clustering', 'copula']

        for method in methods:
            cache_file = os.path.join(cache_data_dir, f"{method}_default.pkl")

            if not os.path.exists(cache_file):
                continue  # 캐시 파일이 없으면 스킵

            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)

                if 'parameters' in cache_data:
                    config_params = get_default_parameters(method)
                    cache_params = cache_data['parameters']

                    assert cache_params == config_params, \
                           f"{method} 캐시 파일의 파라미터가 config와 다름\n" \
                           f"캐시: {cache_params}\n" \
                           f"config: {config_params}\n" \
                           f"→ generate_cache.py를 다시 실행하여 캐시를 재생성하세요"

            except Exception as e:
                pytest.fail(f"{method} 캐시 파일 로드 실패: {str(e)}")


if __name__ == "__main__":
    # pytest를 프로그래밍 방식으로 실행
    pytest.main([__file__, "-v", "--tb=short"])
