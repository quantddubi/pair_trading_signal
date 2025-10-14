# Cache 폴더 구조

이 폴더는 페어트레이딩 방법론의 사전 계산된 캐시 파일들을 관리합니다.

## 폴더 구조

```
cache/
├── generate_cache.py                    # 전체 캐시 생성 스크립트
├── generate_euclidean_cache.py         # Euclidean Distance 개별 캐시 생성
├── generate_ssd_cache.py               # SSD Distance 개별 캐시 생성
├── generate_cointegration_cache.py     # Cointegration 개별 캐시 생성
├── generate_regime_cache.py            # Correlation Regime 개별 캐시 생성
├── generate_ou_cache.py                # OU Mean Reversion 개별 캐시 생성
├── generate_clustering_cache.py        # Clustering 개별 캐시 생성
├── generate_copula_cache.py            # Copula 개별 캐시 생성
├── data/                               # 생성된 캐시 파일 저장 위치
│   ├── euclidean_default.pkl
│   ├── ssd_default.pkl
│   ├── cointegration_default.pkl
│   ├── regime_default.pkl
│   ├── ou_default.pkl
│   ├── clustering_default.pkl
│   └── copula_default.pkl
└── README.md
```

## 캐시 생성 방법

### 1. 모든 방법론 캐시를 한 번에 생성

```bash
# cache 폴더에서 실행
python generate_cache.py

# 또는 프로젝트 루트에서 실행
python cache/generate_cache.py
```

### 2. 개별 방법론 캐시만 생성

각 방법론별로 개별 스크립트를 실행할 수 있습니다:

```bash
# Euclidean Distance
python cache/generate_euclidean_cache.py

# SSD Distance
python cache/generate_ssd_cache.py

# Cointegration
python cache/generate_cointegration_cache.py

# Correlation Regime
python cache/generate_regime_cache.py

# OU Mean Reversion
python cache/generate_ou_cache.py

# Clustering
python cache/generate_clustering_cache.py

# Copula Rank Correlation
python cache/generate_copula_cache.py
```

## 캐시 파일 위치

- **생성 스크립트**: `cache/generate_cache.py`
- **캐시 데이터**: `cache/data/` 하위 폴더
- 각 방법론별로 `{method_name}_default.pkl` 형식으로 저장됩니다

## 캐시 로딩

다른 모듈에서는 `utils/cache_utils.py`를 통해 캐시를 로드합니다:

```python
from utils.cache_utils import load_cache

# 캐시 로드
cache_data = load_cache('euclidean')
if cache_data:
    enter_signals = cache_data['enter_signals']
    watch_signals = cache_data['watch_signals']
```

## 주의사항

1. `generate_cache.py`는 `cache/` 폴더 내에 위치하며, 캐시 파일은 `cache/data/` 하위 폴더에 저장됩니다.
2. `utils/cache_utils.py`는 자동으로 `cache/data/` 경로에서 캐시 파일을 로드합니다.
3. 파라미터는 `config/default_parameters.py`에서 중앙 관리되므로, 파라미터 변경 시 해당 파일만 수정하면 됩니다.
4. 캐시 재생성이 필요한 경우 `cache/generate_cache.py`를 실행하세요.

## 테스트

파라미터 일관성 테스트를 통해 generate_cache.py와 cache_utils.py가 동일한 파라미터를 사용하는지 확인할 수 있습니다:

```bash
python tests/test_parameter_consistency.py
```
