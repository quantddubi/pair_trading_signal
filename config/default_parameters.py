"""
모든 방법론의 기본 파라미터 중앙 관리
Single Source of Truth for all methodology parameters
"""

# 각 방법론의 기본 파라미터 정의
DEFAULT_PARAMETERS = {
    'euclidean': {
        'formation_window': 756,       # 3년 (generate_cache.py 기준)
        'signal_window': 756,          # 3년
        'enter_threshold': 2.0,
        'exit_threshold': 0.5,
        'stop_loss': 3.0,
        'min_half_life': 5,
        'max_half_life': 60,
        'min_cost_ratio': 5.0,
        'transaction_cost': 0.0001
    },

    'ssd': {
        'formation_window': 252,       # 1년 (generate_cache.py 기준)
        'signal_window': 252,          # 1년
        'enter_threshold': 2.0,
        'exit_threshold': 0.5,
        'stop_loss': 3.0,
        'min_half_life': 5,
        'max_half_life': 60,
        'min_cost_ratio': 5.0,
        'transaction_cost': 0.0001
    },

    'cointegration': {
        'formation_window': 252,       # 1년 (generate_cache.py 기준)
        'signal_window': 60,           # 60일
        'enter_threshold': 2.0,
        'exit_threshold': 0.5,
        'stop_loss': 3.0,
        'min_half_life': 5,
        'max_half_life': 60,
        'min_cost_ratio': 5.0,
        'max_pvalue': 0.05
    },

    'regime': {
        'formation_window': 252,       # 1년 (generate_cache.py 기준)
        'signal_window': 60,           # 60일
        'long_corr_window': 252,       # 1년
        'short_corr_window': 60,       # 60일
        'enter_threshold': 2.0,
        'exit_threshold': 0.5,
        'stop_loss': 3.0,
        'min_half_life': 5,
        'max_half_life': 60,
        'min_cost_ratio': 5.0,
        'min_delta_corr': 0.3
    },

    'ou': {
        'formation_window': 252,       # 1년
        'rolling_window': 60,          # 60일
        'base_threshold': 2.0,
        'exit_threshold': 0.5,
        'stop_loss': 3.0,
        'min_half_life': 5,
        'max_half_life': 60,
        'min_cost_ratio': 5.0,
        'min_mean_reversion_speed': 0.01,
        'max_kappa_cv': 0.6,
        'data_coverage_threshold': 0.9,
        'winsorize_percentile': 0.01
    },

    'clustering': {
        'formation_window': 252,       # 1년
        'signal_window': 60,           # 60일
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
        'formation_window': 3000,      # 12년 (generate_cache.py 기준)
        'min_tail_dependence': 0.1,
        'conditional_prob_threshold': 0.05,
        'min_kendall_tau': 0.3,
        'min_data_coverage': 0.85,
        'copula_consistency_threshold': 0.8
    }
}


def get_default_parameters(method_name: str) -> dict:
    """
    특정 방법론의 기본 파라미터 반환

    Args:
        method_name: 방법론 이름 (euclidean, ssd, cointegration, regime, ou, clustering, copula)

    Returns:
        해당 방법론의 기본 파라미터 딕셔너리
    """
    return DEFAULT_PARAMETERS.get(method_name, {}).copy()


def get_all_parameters() -> dict:
    """
    모든 방법론의 기본 파라미터 반환

    Returns:
        전체 파라미터 딕셔너리
    """
    return DEFAULT_PARAMETERS.copy()
