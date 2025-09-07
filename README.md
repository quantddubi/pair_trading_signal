# 📊 페어트레이딩 분석 도구

이 프로젝트는 유클리드 거리 기반의 페어트레이딩 신호를 분석하고 시각화하는 도구입니다.

## 🚀 Streamlit 웹 앱 실행

```bash
cd streamlit_app
streamlit run streamlit_pair_trading.py
```

브라우저에서 `http://localhost:8501`로 접속하면 대화형 분석 도구를 사용할 수 있습니다.

## ✨ 주요 기능

### 📈 실시간 페어 분석
- **분석 기간 조절**: 252일~1260일 (1년~5년)
- **Z-스코어 계산 기간**: 20일~120일
- **진입 임계값**: 1.5~3.0
- **분석 페어 수**: 5~20개

### 🎯 스마트 추천 시스템
- **최고 추천 페어** 자동 선별
- **진입 신호** vs **관찰 대상** 분리
- **자산 이름 표시**: 티커(한글명) 형태로 직관적 표시
  - 예: `SPX Index(S&P500) - RTY Index(R.2000)`

### 📊 고급 시각화
- **3단 차트 구성**:
  - 상단: 두 자산의 정규화된 가격 비교
  - 중단: 스프레드 (가격 차이)  
  - 하단: Z-스코어 (진입/청산 신호)
- **최근 6개월 강조**: 노란색 배경으로 하이라이트
- **진입 임계값**: ±2.0 주황색 라인
- **현재 Z-스코어**: 실시간 주석 표시

## 📋 결과 해석

### Half-Life
- **의미**: 스프레드가 평균으로 돌아오는데 걸리는 시간
- **범위**: 5~60영업일
- **해석**: 짧을수록 빠른 수렴, 길수록 느린 수렴

### Z-Score
- **진입**: |Z-Score| ≥ 2.0
- **관찰**: 1.5 ≤ |Z-Score| < 2.0
- **방향**: 
  - Z > 0: Long asset1, Short asset2
  - Z < 0: Short asset1, Long asset2

## 🎨 사용 예제

### 현재 최고 추천 페어
```
🥇 XM1 Comdty(Australia 10Y) - CNYUSD Curncy(CNY)
진입 방향: Short XM1 Comdty, Long CNYUSD Curncy  
Z-Score: -2.40
Half-Life: 53.5일
```

## 📁 프로젝트 구조

```
pair_trading_signal/
├── streamlit_app/                 # 🌟 Streamlit 웹 애플리케이션
│   ├── streamlit_pair_trading.py  # 메인 앱
│   └── pages/                     # 다중 페이지
│       ├── 1_유클리드_거리.py      # 유클리드 거리 방법론
│       ├── 2_공적분.py             # 공적분 방법론
│       ├── 3_상관관계_레짐.py       # 상관관계 레짐 방법론  
│       ├── 4_OU_평균회귀.py        # OU 평균회귀 방법론
│       ├── 5_클러스터링.py         # 클러스터링 방법론
│       ├── 6_코퓰라_순위상관.py     # 코퓰라 순위상관 방법론
│       └── 7_통합_스크리너.py       # 통합 스크리너
├── methods/                       # 📊 분석 방법론 엔진들
│   ├── 1_euclidean_distance_pairs.py    # 유클리드 거리 기반
│   ├── 2_cointegration_pairs.py         # 공적분 기반
│   ├── 3_correlation_regime_pairs.py    # 상관관계 레짐 기반
│   ├── 4_ou_mean_reversion_pairs.py     # OU 평균회귀 기반
│   ├── 5_clustering_pairs.py            # 클러스터링 기반
│   ├── 6_copula_rank_correlation_pairs.py # 코퓰라 순위상관 기반
│   └── 7_integrated_screener.py         # 통합 스크리너
├── utils/                         # 🔧 공통 유틸리티
│   └── common_utils.py            # 공통 함수들
├── data/                          # 📈 데이터 파일들
│   └── MU Price(BBG).csv          # 가격 데이터 (89개 자산)
├── quick_screener.py              # ⚡ 빠른 스크리너 (CLI)
├── generate_cointegration_cache.py # 💾 공적분 캐시 생성
├── requirements.txt               # 📦 Python 패키지 의존성
└── README.md                      # 📖 이 파일
```

## 🛠️ 설치 및 요구사항

### 필수 패키지
```bash
pip install -r requirements.txt
```

주요 패키지:
- streamlit >= 1.28.0
- plotly >= 5.15.0  
- pandas >= 2.0.0
- numpy >= 1.24.0
- scipy >= 1.10.0

### 데이터 요구사항
- CSV 파일 형식
- 1행: 티커명
- 2행: 자산명 (한글)
- 3행부터: 일별 가격 데이터

## ⚠️ 주의사항

- **분석 목적**: 이 도구는 교육 및 연구 목적으로 개발되었습니다
- **투자 책임**: 실제 투자 결정은 신중히 판단하시기 바랍니다
- **리스크**: Z-Score가 높아도 손실 가능성은 항상 존재합니다
- **백테스팅**: 과거 데이터 기반 분석이므로 미래 성과를 보장하지 않습니다

## 📞 지원

문제가 발생하거나 개선 사항이 있으면 이슈를 등록해주세요.

---
**Made with ❤️ using Streamlit and Plotly**