"""
페어트레이딩 분석 도구 - OU(Ornstein-Uhlenbeck) 평균회귀 기반 방법론
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os
import importlib.util

# 경로 설정 유틸리티
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from utils_path import setup_path, get_data_file_path
from statsmodels.tsa.arima.model import ARIMA

# 프로젝트 루트 경로 추가
project_root = setup_path()
sys.path.insert(0, project_root)

# 동적 모듈 import
def import_module_from_file(file_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# 모듈 import
common_utils = import_module_from_file(os.path.join(project_root, "utils/common_utils.py"), "common_utils")
cache_utils = import_module_from_file(os.path.join(project_root, "utils/cache_utils.py"), "cache_utils")
ou_module = import_module_from_file(os.path.join(project_root, "methods/5_ou_mean_reversion_pairs.py"), "ou_mean_reversion_pairs")

# 필요한 함수들 import
load_data = common_utils.load_data
normalize_prices = common_utils.normalize_prices
calculate_spread = common_utils.calculate_spread
calculate_zscore = common_utils.calculate_zscore
OUMeanReversionPairTrading = ou_module.OUMeanReversionPairTrading

# 페이지 설정
st.set_page_config(
    page_title="OU 평균회귀 방법론",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 캐시된 데이터 로딩
@st.cache_data
def load_price_data():
    """가격 데이터 로딩"""
    file_path = get_data_file_path()
    return load_data(file_path)

@st.cache_data
def load_asset_names():
    """자산 이름 매핑 로딩 (CSV 파일의 1행: 티커, 2행: 이름)"""
    file_path = get_data_file_path()
    
    with open(file_path, 'r') as f:
        ticker_line = f.readline().strip()
        name_line = f.readline().strip()
    
    tickers = ticker_line.split(',')
    names = name_line.split(',')
    
    # 티커-이름 매핑 딕셔너리 생성
    asset_mapping = {}
    for ticker, name in zip(tickers, names):
        asset_mapping[ticker] = name
    
    return asset_mapping

def format_pair_name(pair, asset_mapping):
    """페어 이름을 이름(티커) 형태로 포맷팅"""
    asset1, asset2 = pair.split('-')
    
    name1 = asset_mapping.get(asset1, asset1)
    name2 = asset_mapping.get(asset2, asset2)
    
    return f"{name1}({asset1}) - {name2}({asset2})"

# 페어 분석 함수
@st.cache_data
def analyze_pairs(formation_window, rolling_window, enter_threshold, exit_threshold, stop_loss, min_half_life, max_half_life, min_cost_ratio, min_mean_reversion_speed, max_kappa_cv, data_coverage_threshold, winsorize_percentile, n_pairs):
    """페어 분석 실행"""
    prices = load_price_data()
    
    trader = OUMeanReversionPairTrading(
        formation_window=formation_window,
        rolling_window=rolling_window,
        base_threshold=enter_threshold,
        exit_threshold=exit_threshold,
        stop_loss=stop_loss,
        min_half_life=min_half_life,
        max_half_life=max_half_life,
        min_cost_ratio=min_cost_ratio,
        min_mean_reversion_speed=min_mean_reversion_speed,
        max_kappa_cv=max_kappa_cv,
        data_coverage_threshold=data_coverage_threshold,
        winsorize_percentile=winsorize_percentile
    )
    
    selected_pairs = trader.select_pairs(prices, n_pairs=n_pairs)
    return selected_pairs, prices

def create_ou_mean_reversion_chart(prices, asset1, asset2, formation_window, signal_window, asset_mapping=None):
    """OU 평균회귀 분석 차트 생성"""
    # 전체 기간 데이터
    end_date = prices.index[-1]
    start_date = end_date - timedelta(days=int(formation_window * 1.4))  # 여유를 두고
    
    chart_data = prices.loc[start_date:end_date, [asset1, asset2]].dropna()
    
    if len(chart_data) == 0:
        st.error(f"데이터가 없습니다: {asset1}, {asset2}")
        return None
    
    # 가격 정규화 (리베이스)
    normalized_data = normalize_prices(chart_data, method='rebase')
    
    # OU 과정 분석
    try:
        # 스프레드 계산
        recent_data = chart_data.tail(formation_window)
        normalized_recent = normalize_prices(recent_data, method='rebase')
        
        # 최적 헤지 비율 계산 (간단한 선형 회귀)
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression().fit(normalized_recent[[asset1]], normalized_recent[asset2])
        hedge_ratio = reg.coef_[0]
        
        spread = normalized_recent[asset2] - hedge_ratio * normalized_recent[asset1]
        
        # OU 과정 파라미터 추정 (간단한 AR(1) 모델로 근사)
        try:
            spread_diff = spread.diff().dropna()
            spread_lag = spread.shift(1).dropna()
            
            # 길이 맞춤
            min_len = min(len(spread_diff), len(spread_lag))
            spread_diff = spread_diff.iloc[-min_len:]
            spread_lag = spread_lag.iloc[-min_len:]
            
            # AR(1) 회귀: dy = alpha + beta * y_t-1 + epsilon
            from sklearn.linear_model import LinearRegression
            ar_reg = LinearRegression().fit(spread_lag.values.reshape(-1, 1), spread_diff.values)
            beta = ar_reg.coef_[0]
            alpha = ar_reg.intercept_
            
            # OU 파라미터 변환
            dt = 1/252  # 일일 데이터
            kappa = -beta / dt  # 평균회귀 속도
            theta = -alpha / beta if beta != 0 else 0  # 장기 평균
            
            # 반감기 계산
            half_life = np.log(2) / kappa if kappa > 0 else np.inf
            
        except Exception as e:
            kappa = 0.01
            theta = spread.mean()
            half_life = 50
            
        # 롤링 평균회귀 속도 계산 (30일 윈도우)
        window = 30
        rolling_kappa = []
        rolling_half_life = []
        
        for i in range(window, len(spread)):
            sub_spread = spread.iloc[i-window:i]
            sub_spread_diff = sub_spread.diff().dropna()
            sub_spread_lag = sub_spread.shift(1).dropna()
            
            if len(sub_spread_diff) > 5 and len(sub_spread_lag) > 5:
                try:
                    min_len = min(len(sub_spread_diff), len(sub_spread_lag))
                    sub_spread_diff = sub_spread_diff.iloc[-min_len:]
                    sub_spread_lag = sub_spread_lag.iloc[-min_len:]
                    
                    ar_reg = LinearRegression().fit(sub_spread_lag.values.reshape(-1, 1), sub_spread_diff.values)
                    sub_beta = ar_reg.coef_[0]
                    sub_kappa = -sub_beta / dt
                    sub_half_life = np.log(2) / sub_kappa if sub_kappa > 0 else 100
                    
                    rolling_kappa.append(sub_kappa)
                    rolling_half_life.append(min(sub_half_life, 100))  # Cap at 100 days
                except:
                    rolling_kappa.append(0.01)
                    rolling_half_life.append(50)
            else:
                rolling_kappa.append(0.01)
                rolling_half_life.append(50)
        
        # Z-스코어 계산
        zscore_window = max(20, min(60, len(spread)//4))
        zscore = calculate_zscore(spread, window=zscore_window)
        
    except Exception as e:
        st.error(f"OU 평균회귀 분석 중 오류 발생: {str(e)}")
        return None
    
    # 서브플롯 생성
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.25, 0.2, 0.2, 0.2, 0.15],
        subplot_titles=[
            f'{asset1} vs {asset2} - 정규화된 가격',
            f'스프레드 (헤지비율: {hedge_ratio:.4f})',
            f'평균회귀 속도 κ ({window}일 롤링)',
            f'반감기 ({window}일 롤링)',
            'Z-Score (진입 신호)'
        ]
    )
    
    # 1. 정규화된 가격 차트
    fig.add_trace(
        go.Scatter(
            x=normalized_data.index,
            y=normalized_data[asset1],
            name=asset1,
            line=dict(color='blue', width=2),
            hovertemplate=f'<b>{asset1}</b><br>Date: %{{x}}<br>Price: %{{y:.4f}}<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=normalized_data.index,
            y=normalized_data[asset2],
            name=asset2,
            line=dict(color='red', width=2),
            hovertemplate=f'<b>{asset2}</b><br>Date: %{{x}}<br>Price: %{{y:.4f}}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2. 스프레드
    fig.add_trace(
        go.Scatter(
            x=spread.index,
            y=spread.values,
            name='OU Spread',
            line=dict(color='green', width=2),
            hovertemplate='<b>OU Spread</b><br>Date: %{x}<br>Value: %{y:.4f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 장기 평균선
    fig.add_hline(y=theta, line_dash="dash", line_color="orange", opacity=0.7, row=2, col=1)
    
    # 3. 평균회귀 속도 κ
    if len(rolling_kappa) > 0:
        kappa_dates = spread.index[window:]
        fig.add_trace(
            go.Scatter(
                x=kappa_dates,
                y=rolling_kappa,
                name='κ (평균회귀속도)',
                line=dict(color='purple', width=2),
                hovertemplate='<b>평균회귀속도</b><br>Date: %{x}<br>κ: %{y:.4f}<extra></extra>'
            ),
            row=3, col=1
        )
        
        # 임계값 라인
        fig.add_hline(y=0.01, line_dash="dash", line_color="gray", opacity=0.5, row=3, col=1)
    
    # 4. 반감기
    if len(rolling_half_life) > 0:
        fig.add_trace(
            go.Scatter(
                x=kappa_dates,
                y=rolling_half_life,
                name='Half-Life',
                line=dict(color='brown', width=2),
                hovertemplate='<b>반감기</b><br>Date: %{x}<br>Half-Life: %{y:.1f}일<extra></extra>'
            ),
            row=4, col=1
        )
        
        # 반감기 기준선들
        fig.add_hline(y=5, line_dash="dash", line_color="green", opacity=0.7, row=4, col=1)
        fig.add_hline(y=60, line_dash="dash", line_color="red", opacity=0.7, row=4, col=1)
    
    # 5. Z-스코어
    if len(zscore.dropna()) > 0:
        zscore_dates = zscore.dropna().index
        zscore_values = zscore.dropna().values
        
        fig.add_trace(
            go.Scatter(
                x=zscore_dates,
                y=zscore_values,
                name='Z-Score',
                line=dict(color='darkred', width=2),
                hovertemplate='<b>Z-Score</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
            ),
            row=5, col=1
        )
        
        # Z-스코어 임계값 라인들
        fig.add_hline(y=2.0, line_dash="dash", line_color="orange", opacity=0.7, row=5, col=1)
        fig.add_hline(y=-2.0, line_dash="dash", line_color="orange", opacity=0.7, row=5, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=5, col=1)
    
    # 최근 6개월 배경색 강조
    six_months_ago = end_date - timedelta(days=180)
    for row in range(1, 6):
        fig.add_vrect(
            x0=six_months_ago, x1=end_date,
            fillcolor="yellow", opacity=0.1,
            layer="below", line_width=0,
            row=row, col=1
        )
    
    # 차트 제목에 자산 이름 포함
    if asset_mapping:
        name1 = asset_mapping.get(asset1, asset1)
        name2 = asset_mapping.get(asset2, asset2)
        chart_title = f"OU 평균회귀 분석: {name1}({asset1}) - {name2}({asset2})"
    else:
        chart_title = f"OU 평균회귀 분석: {asset1} - {asset2}"
    
    # 레이아웃 설정
    fig.update_layout(
        height=1200,
        title=chart_title,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )
    
    # 축 레이블 설정
    fig.update_xaxes(title_text="Date", row=5, col=1)
    fig.update_yaxes(title_text="Normalized Price", row=1, col=1)
    fig.update_yaxes(title_text="Spread", row=2, col=1)
    fig.update_yaxes(title_text="κ", row=3, col=1)
    fig.update_yaxes(title_text="Half-Life (days)", row=4, col=1)
    fig.update_yaxes(title_text="Z-Score", row=5, col=1)
    
    # 현재 값들을 주석으로 추가
    if len(zscore.dropna()) > 0 and len(rolling_kappa) > 0:
        current_zscore = zscore_values[-1] if len(zscore_values) > 0 else 0
        current_kappa = rolling_kappa[-1] if len(rolling_kappa) > 0 else 0
        current_half_life = rolling_half_life[-1] if len(rolling_half_life) > 0 else 0
        
        fig.add_annotation(
            x=zscore_dates[-1] if len(zscore_dates) > 0 else end_date,
            y=current_zscore,
            text=f"현재 Z-Score: {current_zscore:.2f}<br>κ: {current_kappa:.4f}<br>반감기: {current_half_life:.1f}일",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="darkred",
            font=dict(size=12, color="darkred"),
            row=5, col=1
        )
    
    return fig

def check_parameters_default(params):
    """파라미터가 기본값인지 확인"""
    default_params = cache_utils.get_default_parameters('ou')
    return params == default_params

# 메인 앱
def main():
    st.title("OU 평균회귀 속도 기반 페어트레이딩")
    st.markdown("---")
    
    # OU 평균회귀 방법론 설명
    st.info("""
    ### OU(Ornstein-Uhlenbeck) 평균회귀 기반 페어 선정 방법론
    
    **핵심 원리**: 스프레드가 평균으로 돌아가는 속도(평균회귀속도 κ)를 수학적으로 모델링하여, 가장 빠르고 안정적으로 수렴하는 페어를 선정하는 전략
    
    **OU 과정 수학적 모델**: dX(t) = κ(θ - X(t))dt + σdW(t)
    - **κ (kappa)**: 평균회귀속도 → 클수록 빠른 수렴
    - **θ (theta)**: 장기평균 → 스프레드가 수렴할 목표점
    - **σ**: 변동성 → 노이즈 수준
    
    **상세 작동 과정**:
    1. **스프레드 OU 과정 적합**: 각 페어의 스프레드를 OU 모델로 추정
       - AR(1) 근사: ΔSpread(t) = α + β×Spread(t-1) + ε(t)
       - OU 변환: κ = -β/Δt, θ = -α/β
    2. **평균회귀속도 품질평가**: 
       - **κ > 0.01**: 유의미한 평균회귀 존재
       - **κ 안정성**: 시간에 따른 κ 변동성 최소화
       - **반감기**: Half-Life = ln(2)/κ → 5~60일 범위 선호
    3. **동적 모니터링**: 30일 롤링 윈도우로 κ 실시간 추적
       - κ 증가 → 더 강한 평균회귀 → 좋은 신호
       - κ 감소 → 약화된 평균회귀 → 위험 신호
    4. **품질 필터링**:
       - **평균 κ**: 0.01 이상 (충분한 평균회귀 강도)
       - **κ 일관성**: 시간에 따른 안정적 유지
       - **Half-Life**: 5~60일 (너무 빠르거나 느리지 않은 수렴)
    5. **진입 타이밍**: Z-Score + κ 강도 조합으로 최적 진입점 포착
    
    **핵심**: **수학적으로 검증된 평균회귀 강도**를 기반으로 **가장 신뢰할 수 있는 수렴 페어** 선정
    
    **장점**: 수학적 엄밀성, 평균회귀 강도 정량화, 동적 품질 모니터링
    **단점**: 모델 복잡성, 구조변화 감지 지연, 계산 집약적
    """)
    
    # 사이드바 설정
    st.sidebar.header("분석 설정")
    st.sidebar.markdown("### 기간 설정")
    
    formation_window = st.sidebar.slider(
        "Formation Window (일)",
        min_value=100,
        max_value=500,
        value=252,
        step=50,
        help="OU 과정 추정을 위한 과거 데이터 기간"
    )
    
    signal_window = st.sidebar.slider(
        "Signal Window (일)", 
        min_value=20, 
        max_value=120, 
        value=60,
        help="Z-score 계산을 위한 롤링 윈도우"
    )
    
    st.sidebar.markdown("### 신호 설정")
    
    enter_threshold = st.sidebar.slider(
        "진입 임계값 (Z-score)", 
        min_value=1.0, 
        max_value=3.0, 
        value=2.0, 
        step=0.1,
        help="이 값 이상일 때 진입 신호 생성"
    )
    
    exit_threshold = st.sidebar.slider(
        "청산 임계값 (Z-score)", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.5, 
        step=0.1,
        help="이 값 이하일 때 청산 신호 생성"
    )
    
    stop_loss = st.sidebar.slider(
        "손절 임계값 (Z-score)", 
        min_value=2.5, 
        max_value=5.0, 
        value=3.0, 
        step=0.1,
        help="이 값 이상일 때 강제 손절"
    )
    
    st.sidebar.markdown("### 품질 필터")
    
    min_half_life = st.sidebar.slider(
        "최소 반감기 (일)", 
        min_value=1, 
        max_value=20, 
        value=5,
        help="평균회귀 최소 속도 기준"
    )
    
    max_half_life = st.sidebar.slider(
        "최대 반감기 (일)", 
        min_value=30, 
        max_value=120, 
        value=60,
        help="평균회귀 최대 속도 기준"
    )
    
    min_cost_ratio = st.sidebar.slider(
        "최소 비용비율", 
        min_value=1.0, 
        max_value=10.0, 
        value=5.0, 
        step=0.5,
        help="거래비용 대비 수익 최소 비율"
    )
    
    min_mean_reversion_speed = st.sidebar.slider(
        "최소 평균회귀속도 (κ)", 
        min_value=0.001, 
        max_value=0.1, 
        value=0.01, 
        step=0.001,
        format="%.3f",
        help="OU 과정의 최소 평균회귀 속도"
    )
    
    st.sidebar.markdown("### 고급 설정")
    
    rolling_window = st.sidebar.slider(
        "Rolling Window (일)", 
        min_value=30, 
        max_value=120, 
        value=60,
        help="OU 파라미터 추정용 롤링 윈도우"
    )
    
    max_kappa_cv = st.sidebar.slider(
        "최대 κ 변동계수", 
        min_value=0.2, 
        max_value=1.0, 
        value=0.6, 
        step=0.1,
        help="κ 안정성 체크 (낮을수록 안정적)"
    )
    
    data_coverage_threshold = st.sidebar.slider(
        "최소 데이터 커버리지", 
        min_value=0.7, 
        max_value=0.95, 
        value=0.9, 
        step=0.05,
        help="데이터 품질 임계값 (90% = 252일 중 227일)"
    )
    
    winsorize_percentile = st.sidebar.slider(
        "윈저라이즈 퍼센타일", 
        min_value=0.001, 
        max_value=0.05, 
        value=0.01, 
        step=0.001,
        format="%.3f",
        help="이상치 처리 임계값 (1% = 상하위 1% 제거)"
    )
    
    n_pairs = st.sidebar.slider(
        "분석할 페어 수",
        min_value=5,
        max_value=20,
        value=10,
        step=1,
        help="상위 몇 개 페어를 분석할지 설정"
    )
    
    # 분석 실행 버튼
    if st.sidebar.button("분석 실행", type="primary"):
        st.cache_data.clear()  # 캐시 클리어
    
    # 파라미터 딕셔너리
    params = {
        'formation_window': formation_window,
        'rolling_window': rolling_window,
        'enter_threshold': enter_threshold,
        'exit_threshold': exit_threshold,
        'stop_loss': stop_loss,
        'min_half_life': min_half_life,
        'max_half_life': max_half_life,
        'min_cost_ratio': min_cost_ratio,
        'min_mean_reversion_speed': min_mean_reversion_speed,
        'max_kappa_cv': max_kappa_cv,
        'data_coverage_threshold': data_coverage_threshold,
        'winsorize_percentile': winsorize_percentile
    }
    
    # 기본값 여부 확인
    is_default = check_parameters_default(params)
    
    # 메인 콘텐츠
    with st.spinner("OU 평균회귀 페어 분석 중... 잠시만 기다려주세요."):
        try:
            if is_default:
                st.info("🚀 기본 파라미터를 사용 중. 사전 계산된 결과를 즉시 표시")
                
                # 캐시에서 결과 로드
                cache_data = cache_utils.load_cache('ou')
                if cache_data:
                    enter_list = cache_data.get('enter_signals', [])
                    watch_list = cache_data.get('watch_signals', [])
                    # OU에서는 enter_list와 watch_list를 합쳐서 selected_pairs로 사용
                    selected_pairs = enter_list + watch_list
                    prices = load_price_data()
                else:
                    st.error("캐시 데이터를 찾을 수 없음")
                    return
            else:
                st.warning("⚙️ 사용자 정의 파라미터가 설정")
                # 실시간 분석 실행
                selected_pairs, prices = analyze_pairs(
                    formation_window, rolling_window, enter_threshold, exit_threshold,
                    stop_loss, min_half_life, max_half_life, min_cost_ratio, 
                    min_mean_reversion_speed, max_kappa_cv, data_coverage_threshold, 
                    winsorize_percentile, n_pairs
                )
            
            asset_mapping = load_asset_names()  # 자산 이름 매핑 로딩
            
        except Exception as e:
            st.error(f"분석 중 오류 발생: {str(e)}")
            return
    
    # 분석 결과 요약
    st.header("분석 결과 요약")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("선별된 페어", f"{len(selected_pairs)}개", help="모든 품질 필터를 통과한 페어")
    
    with col2:
        entry_signals = sum(1 for pair in selected_pairs if pair.get('signal_type') == 'ENTRY')
        st.metric("진입 신호", f"{entry_signals}개", help="현재 진입 가능한 페어")
    
    with col3:
        avg_kappa = np.mean([pair.get('kappa_avg', 0.01) for pair in selected_pairs]) if selected_pairs else 0
        st.metric("평균 κ", f"{avg_kappa:.4f}", help="평균회귀속도")
        
    with col4:
        avg_half_life = np.mean([pair.get('half_life_avg', 50) for pair in selected_pairs]) if selected_pairs else 0
        st.metric("평균 반감기", f"{avg_half_life:.1f}일", help="선별된 페어들의 평균 반감기")
    
    st.markdown("---")
    
    # 선별된 페어 테이블
    if selected_pairs:
        st.header("선별된 OU 평균회귀 페어")
        
        # 테이블 데이터 준비
        table_data = []
        for i, pair_info in enumerate(selected_pairs, 1):
            # OU 메서드는 asset1, asset2 키를 사용하고, 캐시는 pair 키를 사용
            if 'pair' in pair_info:
                formatted_pair = format_pair_name(pair_info['pair'], asset_mapping)
            else:
                # asset1, asset2로부터 pair 생성
                pair_str = f"{pair_info['asset1']}-{pair_info['asset2']}"
                formatted_pair = format_pair_name(pair_str, asset_mapping)
            
            table_data.append({
                "순위": i,
                "페어": formatted_pair,
                "신호": pair_info.get('signal_type', 'NEUTRAL'),
                "Z-Score": f"{pair_info.get('current_zscore', 0):.2f}",
                "κ (속도)": f"{pair_info.get('kappa_avg', 0.01):.4f}",
                "반감기": f"{pair_info.get('half_life_avg', 50):.1f}일",
                "품질점수": f"{pair_info.get('quality_score', 0.0):.1f}",
                "헤지비율": f"{pair_info.get('hedge_ratio', 1.0):.4f}"
            })
        
        df_pairs = pd.DataFrame(table_data)
        
        # 스타일링된 테이블 표시
        st.dataframe(
            df_pairs,
            use_container_width=True,
            hide_index=True,
            column_config={
                "순위": st.column_config.NumberColumn("순위", width="small"),
                "페어": st.column_config.TextColumn("페어", width="medium"),
                "신호": st.column_config.TextColumn("신호 타입", width="small"),
                "Z-Score": st.column_config.TextColumn("Z-Score", width="small"),
                "κ (속도)": st.column_config.TextColumn("κ (속도)", width="small"),
                "반감기": st.column_config.TextColumn("반감기", width="small"),
                "품질점수": st.column_config.TextColumn("품질점수", width="small"),
                "헤지비율": st.column_config.TextColumn("헤지비율", width="small")
            }
        )
        
        st.markdown("---")
        
        # 페어 선택 및 차트 표시
        st.header("페어 상세 분석")
        
        # 최고 추천 페어 표시
        top_pair = selected_pairs[0]
        if 'pair' in top_pair:
            top_pair_str = top_pair['pair']
        else:
            top_pair_str = f"{top_pair['asset1']}-{top_pair['asset2']}"
        top_formatted_pair = format_pair_name(top_pair_str, asset_mapping)
        st.success(f"최고 품질 페어: {top_formatted_pair}")
        
        # 페어 선택 옵션 (표시는 포맷팅된 이름, 값은 원래 페어)
        # pair 키가 있으면 사용, 없으면 asset1-asset2로 생성
        pair_options = []
        pair_display_names = []
        for pair_info in selected_pairs:
            if 'pair' in pair_info:
                pair_str = pair_info['pair']
            else:
                pair_str = f"{pair_info['asset1']}-{pair_info['asset2']}"
            pair_options.append(pair_str)
            pair_display_names.append(format_pair_name(pair_str, asset_mapping))
        
        # selectbox에서 표시할 옵션들 생성
        pair_mapping = {display: original for display, original in zip(pair_display_names, pair_options)}
        
        selected_display_pair = st.selectbox(
            "분석할 페어 선택:",
            options=pair_display_names,
            index=0,
            help="차트로 분석할 페어를 선택하세요"
        )
        
        # 선택된 페어의 상세 정보 표시
        selected_pair = pair_mapping[selected_display_pair]
        selected_pair_info = None
        
        # 선택된 페어의 정보 찾기
        for signal in enter_list:
            if signal['pair'] == selected_pair:
                selected_pair_info = signal
                break
        
        if selected_pair_info:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("진입 방향", selected_pair_info['direction'])
            with col2:
                st.metric("현재 Z-Score", f"{selected_pair_info['current_zscore']:.2f}")
            with col3:
                st.metric("κ (평균회귀속도)", f"{selected_pair_info.get('kappa_avg', 0.01):.4f}")
            with col4:
                st.metric("반감기", f"{selected_pair_info.get('half_life_avg', 50):.1f}일")
        
        if selected_pair:
            asset1, asset2 = selected_pair.split('-')
            
            # 차트 생성 및 표시
            with st.spinner(f"{selected_display_pair} OU 평균회귀 차트 생성 중..."):
                fig = create_ou_mean_reversion_chart(prices, asset1, asset2, formation_window, signal_window, asset_mapping)
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 차트 설명
                    st.info("""
                    **OU 평균회귀 차트 설명:**
                    - **상단**: 두 자산의 정규화된 가격 추이
                    - **2번째**: OU 스프레드와 장기 평균 (θ) - 주황색 선은 수렴 목표점
                    - **3번째**: 평균회귀속도 κ (30일 롤링) - 높을수록 강한 평균회귀
                    - **4번째**: 반감기 (30일 롤링) - 낮을수록 빠른 수렴
                    - **하단**: Z-스코어 (평균회귀 진입 신호)
                    - **노란색 배경**: 최근 6개월 기간
                    - **초록/빨간색 선**: 반감기 기준 (5일/60일)
                    - **주황색 선**: 진입 임계값 (±2.0)
                    """)
    
    else:
        st.warning("현재 진입 조건을 만족하는 OU 평균회귀 페어가 없음")
        st.info("평균회귀속도 κ 최소값을 낮추거나 Z-Score 임계값을 낮춰야함")
    
    # 관찰 대상 테이블
    if watch_list:
        st.header("관찰 대상 페어")
        
        table_data = []
        for i, signal in enumerate(watch_list, 1):
            formatted_pair = format_pair_name(signal['pair'], asset_mapping)
            table_data.append({
                "순위": i,
                "페어": formatted_pair,
                "Z-Score": f"{signal['current_zscore']:.2f}",
                "κ (속도)": f"{signal.get('kappa_avg', 0.01):.4f}",
                "반감기": f"{signal.get('half_life_avg', 50):.1f}일",
                "품질점수": f"{signal.get('quality_score', 0.0):.1f}",
                "헤지비율": f"{signal.get('hedge_ratio', 1.0):.4f}"
            })
        
        df_watch = pd.DataFrame(table_data)
        st.dataframe(df_watch, use_container_width=True, hide_index=True)
    
    # 캐시 정보 표시 (기본 파라미터 사용시에만)
    if is_default and 'cache_data' in locals():
        st.markdown("---")
        st.caption(f"📅 캐시 생성일: {cache_data.get('generated_at', 'Unknown')}")
        st.caption(f"📊 데이터 기준일: {cache_data.get('data_date', 'Unknown')}")
    
    # 푸터
    st.markdown("---")

# Streamlit 페이지로 실행
if __name__ == "__main__":
    main()
else:
    main()