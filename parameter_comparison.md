# 파라미터 불일치 분석

## 1. Euclidean Distance
| 파라미터 | generate_cache.py | cache_utils.py | 일치 |
|---------|------------------|----------------|------|
| formation_window | 756 | 1512 | ❌ |
| signal_window | 756 | 378 | ❌ |
| max_half_life | 60 | 180 | ❌ |

## 2. SSD Distance
| 파라미터 | generate_cache.py | cache_utils.py | 일치 |
|---------|------------------|----------------|------|
| formation_window | 252 | 1008 | ❌ |
| signal_window | 252 | 252 | ✅ |
| max_half_life | 60 | 180 | ❌ |

## 3. Cointegration
| 파라미터 | generate_cache.py | cache_utils.py | 일치 |
|---------|------------------|----------------|------|
| formation_window | 252 | 756 | ❌ |
| signal_window | 60 | 126 | ❌ |
| max_half_life | 60 | 252 | ❌ |

## 4. Correlation Regime
| 파라미터 | generate_cache.py | cache_utils.py | 일치 |
|---------|------------------|----------------|------|
| formation_window | 252 | 1008 | ❌ |
| signal_window | 60 | 252 | ❌ |
| long_corr_window | 252 | 504 | ❌ |
| short_corr_window | 60 | 126 | ❌ |
| enter_threshold | 2.0 | 1.8 | ❌ |
| max_half_life | 60 | 180 | ❌ |
| min_cost_ratio | 5.0 | 3.0 | ❌ |
| min_delta_corr | 0.3 | 0.15 | ❌ |

## 5. OU Mean Reversion
generate_cache.py에는 있지만 cache_utils.py 파라미터와 비교 필요

## 6. Clustering
**캐시 사용 안 함** - generate_cache.py에는 있지만 페이지에서 캐시 로직 없음

## 7. Copula Rank Correlation
| 파라미터 | generate_cache.py | cache_utils.py | 일치 |
|---------|------------------|----------------|------|
| formation_window | 3000 | 756 | ❌ |
| min_tail_dependence | 0.1 | 0.2 | ❌ |
| conditional_prob_threshold | 0.05 | (없음) | ❌ |
| min_kendall_tau | 0.3 | (없음) | ❌ |

## 결론

**치명적 문제**: 모든 방법론에서 `generate_cache.py`와 `cache_utils.py`의 파라미터가 불일치합니다.

이는 다음을 의미합니다:
1. 페이지에서 `parameters_match_default()` 함수가 항상 `False`를 반환
2. 생성된 캐시 파일이 있어도 **실제로는 사용되지 않음**
3. 모든 페이지가 매번 실시간 계산을 수행

**해결 방법**:
- `generate_cache.py`의 파라미터를 `cache_utils.py`와 일치시키거나
- `cache_utils.py`의 기본 파라미터를 `generate_cache.py`와 일치시켜야 함
