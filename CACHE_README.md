# 캐시 파일 생성 가이드

페어트레이딩 방법론별 캐시 파일을 개별적으로 생성하는 방법을 설명합니다.

## 📁 캐시 파일 구조

```
cache/
├── euclidean_default.pkl        # 유클리드 거리 방법론
├── ssd_default.pkl             # SSD 방법론  
├── cointegration_default.pkl   # 공적분 방법론
├── regime_default.pkl          # 상관관계 레짐 방법론
├── ou_default.pkl              # OU 평균회귀 방법론
├── clustering_default.pkl      # 클러스터링 방법론
└── copula_default.pkl          # 코퓰라 방법론
```

## 🚀 사용법

### 1. 통합 캐시 생성 도구

모든 방법론을 개별적으로 또는 일괄적으로 생성할 수 있습니다:

```bash
# 특정 방법론만 생성
python generate_cache_individual.py regime
python generate_cache_individual.py cointegration  
python generate_cache_individual.py euclidean

# 모든 방법론 순차 생성 (오래 걸림)
python generate_cache_individual.py all

# 도움말 보기
python generate_cache_individual.py
```

### 2. 개별 방법론 전용 스크립트

특정 방법론만 빠르게 생성하고 싶을 때:

```bash
# 상관관계 레짐 방법론만 생성
python generate_regime_cache.py

# 공적분 방법론만 생성  
python generate_cointegration_cache_new.py
```

## ⏱️ 예상 소요 시간

| 방법론 | 예상 시간 | 설명 |
|--------|-----------|------|
| euclidean | 3-5분 | 유클리드 거리 계산 |
| ssd | 3-5분 | SSD 계산 |
| cointegration | 8-12분 | ADF 테스트 + 회귀분석 |
| regime | 15-20분 | 이중 상관관계 계산 |
| ou | 10-15분 | OU 프로세스 추정 |
| clustering | 5-8분 | 클러스터링 + 거리계산 |
| copula | 12-18분 | 순위상관 + 꼬리의존성 |
| **전체** | **60-90분** | 모든 방법론 순차 실행 |

## 📊 캐시 파일 확인

캐시가 정상적으로 생성되었는지 확인:

```python
from utils.cache_utils import get_cache_info

cache_info = get_cache_info()
for method, info in cache_info.items():
    if info['exists']:
        print(f"✅ {method}: {info['enter_count']}개 진입, {info['watch_count']}개 관찰")
    else:
        print(f"❌ {method}: 파일 없음")
```

## 🔄 캐시 재생성이 필요한 경우

다음의 경우 캐시를 재생성해야 합니다:

1. **데이터 업데이트**: 새로운 시장 데이터가 추가됨
2. **파라미터 변경**: 기본 파라미터가 수정됨  
3. **방법론 수정**: 알고리즘이 개선됨
4. **오류 발생**: 캐시 파일이 손상됨

## ⚡ 성능 최적화 팁

### 1. 필요한 방법론만 생성
자주 사용하는 방법론만 우선 생성:

```bash
# 인기 방법론 우선 생성
python generate_cache_individual.py cointegration
python generate_cache_individual.py regime  
python generate_cache_individual.py euclidean
```

### 2. 백그라운드 실행
시간이 많이 걸리는 경우 백그라운드에서 실행:

```bash
# Linux/Mac에서 백그라운드 실행
nohup python generate_cache_individual.py all > cache_generation.log 2>&1 &

# 진행 상황 확인
tail -f cache_generation.log
```

### 3. 병렬 실행
여러 터미널에서 동시에 다른 방법론 실행:

```bash
# 터미널 1
python generate_cache_individual.py cointegration

# 터미널 2  
python generate_cache_individual.py regime

# 터미널 3
python generate_cache_individual.py clustering
```

## 🐛 문제 해결

### 1. 메모리 부족 오류
```bash
# 메모리 사용량이 높은 방법론들은 개별 실행
python generate_cache_individual.py regime
python generate_cache_individual.py copula
```

### 2. 모듈 import 오류
```bash
# 프로젝트 루트 디렉토리에서 실행
cd /Users/a/PycharmProjects/pair_trading_signal
python generate_cache_individual.py [METHOD]
```

### 3. 데이터 파일 없음
```bash
# 데이터 파일 경로 확인
ls -la data/MU\ Price\(BBG\).csv
```

## 📈 모니터링

캐시 생성 진행 상황은 콘솔 출력으로 확인할 수 있습니다:

```
🔍 상관관계 레짐 분석 캐시 생성 시작...
✅ 데이터 로딩 완료: 89개 자산, 6702일
📊 사용 파라미터:
   formation_window: 504
   signal_window: 126
   long_corr_window: 378
   short_corr_window: 126
   enter_threshold: 1.8
   exit_threshold: 0.5
   stop_loss: 3.0
   min_half_life: 5
   max_half_life: 90
   min_cost_ratio: 3.0
   min_delta_corr: 0.15
🔄 페어 분석 실행 중...
✅ 상관관계 레짐 캐시 생성 완료!
📁 파일: cache/regime_default.pkl
📈 진입 신호: 1개
👀 관찰 대상: 0개
⏱️ 소요 시간: 847.2초
```

## 🎯 권장 실행 순서

처음 캐시를 생성할 때 권장하는 순서:

1. **euclidean** (빠름, 테스트용)
2. **ssd** (빠름, 기본 방법론)  
3. **cointegration** (중간, 중요 방법론)
4. **regime** (느림, 새로 최적화된 방법론)
5. **clustering** (중간)
6. **ou** (중간)
7. **copula** (느림)

```bash
# 순차 실행
python generate_cache_individual.py euclidean
python generate_cache_individual.py ssd
python generate_cache_individual.py cointegration
python generate_cache_individual.py regime
python generate_cache_individual.py clustering  
python generate_cache_individual.py ou
python generate_cache_individual.py copula
```