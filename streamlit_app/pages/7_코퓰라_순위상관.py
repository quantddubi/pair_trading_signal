"""
페어트레이딩 분석 도구 - 코퓰라 순위상관 기반 방법론
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.stats import kendalltau, spearmanr
from datetime import datetime, timedelta
import sys
import os
import importlib.util

# 경로 설정 유틸리티
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from utils_path import setup_path, get_data_file_path

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
copula_module = import_module_from_file(os.path.join(project_root, "methods/7_copula_rank_correlation_pairs.py"), "copula_rank_correlation_pairs")

# 필요한 함수들 import
load_data = common_utils.load_data
normalize_prices = common_utils.normalize_prices
calculate_spread = common_utils.calculate_spread
calculate_zscore = common_utils.calculate_zscore
CopulaRankCorrelationPairTrading = copula_module.CopulaRankCorrelationPairTrading

# 페이지 설정
st.set_page_config(
    page_title="코퓰라 순위상관 방법론",
    page_icon="🎲",
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

def check_parameters_default(params):
    """파라미터가 기본값인지 확인"""
    default_params = cache_utils.get_default_parameters('copula')
    for key, value in default_params.items():
        if params.get(key) != value:
            return False
    return True

# 페어 분석 함수
@st.cache_data
def analyze_pairs(formation_days, signal_days, long_window, short_window, enter_threshold, n_pairs, 
                  min_rank_corr, min_rank_corr_change, tail_quantile):
    """페어 분석 실행"""
    prices = load_price_data()
    
    trader = CopulaRankCorrelationPairTrading(
        formation_window=formation_days,
        signal_window=signal_days,
        long_window=long_window,
        short_window=short_window,
        enter_threshold=enter_threshold,
        exit_threshold=0.5,
        stop_loss=3.0,
        min_half_life=5,
        max_half_life=60,
        min_cost_ratio=5.0,
        min_rank_corr=min_rank_corr,
        min_rank_corr_change=min_rank_corr_change,
        tail_quantile=tail_quantile
    )
    
    enter_list, watch_list = trader.screen_pairs(prices, n_pairs=n_pairs)
    return enter_list, watch_list, prices

def create_copula_scatter(prices, asset1, asset2, formation_days):
    """코퓰라 산점도 생성 (Uniform 변환 후)"""
    # 최근 데이터 추출
    recent_data = prices[[asset1, asset2]].tail(formation_days).dropna()
    
    if len(recent_data) < 100:
        return None
    
    # 수익률 계산
    returns1 = recent_data[asset1].pct_change().dropna()
    returns2 = recent_data[asset2].pct_change().dropna()
    
    # 공통 인덱스
    common_idx = returns1.index.intersection(returns2.index)
    if len(common_idx) < 50:
        return None
        
    returns1_common = returns1[common_idx]
    returns2_common = returns2[common_idx]
    
    # Uniform 변환 (경험적 분포함수)
    def empirical_cdf(x):
        return pd.Series(x).rank() / (len(x) + 1)
    
    u1 = empirical_cdf(returns1_common)
    u2 = empirical_cdf(returns2_common)
    
    # 산점도 생성
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=u1,
        y=u2,
        mode='markers',
        marker=dict(
            size=4,
            opacity=0.6,
            color=returns1_common.values,
            colorscale='RdBu_r',
            showscale=True,
            colorbar=dict(title="Returns")
        ),
        name='Copula',
        text=[f"Date: {idx.strftime('%Y-%m-%d')}<br>U1: {u1_val:.3f}<br>U2: {u2_val:.3f}" 
              for idx, u1_val, u2_val in zip(common_idx, u1, u2)],
        hovertemplate='%{text}<extra></extra>'
    ))
    
    # 독립성 대각선
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        name='Independence Line',
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title=f'Copula Scatter Plot: {asset1} vs {asset2}',
        xaxis_title=f'{asset1} (Uniform)',
        yaxis_title=f'{asset2} (Uniform)',
        width=600,
        height=600,
        template='plotly_white'
    )
    
    return fig

def create_tail_dependence_chart(prices, asset1, asset2, formation_days, tail_quantile=0.1):
    """꼬리 의존성 시각화"""
    recent_data = prices[[asset1, asset2]].tail(formation_days).dropna()
    
    if len(recent_data) < 100:
        return None
    
    returns1 = recent_data[asset1].pct_change().dropna()
    returns2 = recent_data[asset2].pct_change().dropna()
    
    common_idx = returns1.index.intersection(returns2.index)
    if len(common_idx) < 50:
        return None
        
    returns1_common = returns1[common_idx]
    returns2_common = returns2[common_idx]
    
    # 임계값 계산
    threshold1_low = returns1_common.quantile(tail_quantile)
    threshold1_high = returns1_common.quantile(1 - tail_quantile)
    threshold2_low = returns2_common.quantile(tail_quantile)
    threshold2_high = returns2_common.quantile(1 - tail_quantile)
    
    fig = go.Figure()
    
    # 전체 점들
    fig.add_trace(go.Scatter(
        x=returns1_common,
        y=returns2_common,
        mode='markers',
        marker=dict(size=4, color='lightblue', opacity=0.4),
        name='All Returns',
        showlegend=True
    ))
    
    # 하방 꼬리 (동시 극단 하락)
    lower_tail_mask = (returns1_common <= threshold1_low) & (returns2_common <= threshold2_low)
    if lower_tail_mask.sum() > 0:
        fig.add_trace(go.Scatter(
            x=returns1_common[lower_tail_mask],
            y=returns2_common[lower_tail_mask],
            mode='markers',
            marker=dict(size=8, color='red'),
            name=f'Lower Tail ({lower_tail_mask.sum()} points)',
            showlegend=True
        ))
    
    # 상방 꼬리 (동시 극단 상승)
    upper_tail_mask = (returns1_common >= threshold1_high) & (returns2_common >= threshold2_high)
    if upper_tail_mask.sum() > 0:
        fig.add_trace(go.Scatter(
            x=returns1_common[upper_tail_mask],
            y=returns2_common[upper_tail_mask],
            mode='markers',
            marker=dict(size=8, color='green'),
            name=f'Upper Tail ({upper_tail_mask.sum()} points)',
            showlegend=True
        ))
    
    # 임계선들
    fig.add_hline(y=threshold2_low, line_dash="dash", line_color="red", opacity=0.5)
    fig.add_hline(y=threshold2_high, line_dash="dash", line_color="green", opacity=0.5)
    fig.add_vline(x=threshold1_low, line_dash="dash", line_color="red", opacity=0.5)
    fig.add_vline(x=threshold1_high, line_dash="dash", line_color="green", opacity=0.5)
    
    fig.update_layout(
        title=f'Tail Dependence Analysis: {asset1} vs {asset2}',
        xaxis_title=f'{asset1} Returns',
        yaxis_title=f'{asset2} Returns',
        template='plotly_white',
        height=500
    )
    
    return fig

def create_rank_correlation_chart(prices, asset1, asset2, formation_days, long_window, short_window):
    """순위상관 시계열 차트"""
    recent_data = prices[[asset1, asset2]].tail(formation_days * 2).dropna()
    
    if len(recent_data) < max(long_window, short_window) + 50:
        return None
    
    returns1 = recent_data[asset1].pct_change().dropna()
    returns2 = recent_data[asset2].pct_change().dropna()
    
    common_idx = returns1.index.intersection(returns2.index)
    if len(common_idx) < max(long_window, short_window) + 50:
        return None
        
    returns1_common = returns1[common_idx]
    returns2_common = returns2[common_idx]
    
    # 롤링 Kendall's tau 계산
    rolling_kendall = []
    rolling_spearman = []
    dates = []
    
    for i in range(long_window, len(returns1_common)):
        window_r1 = returns1_common.iloc[i-long_window:i]
        window_r2 = returns2_common.iloc[i-long_window:i]
        
        try:
            kendall_corr, _ = kendalltau(window_r1, window_r2)
            spearman_corr, _ = spearmanr(window_r1, window_r2)
            
            rolling_kendall.append(kendall_corr if not np.isnan(kendall_corr) else 0)
            rolling_spearman.append(spearman_corr if not np.isnan(spearman_corr) else 0)
            dates.append(returns1_common.index[i])
        except:
            rolling_kendall.append(0)
            rolling_spearman.append(0)
            dates.append(returns1_common.index[i])
    
    # 단기 롤링 상관계수
    short_kendall = []
    short_spearman = []
    
    for i in range(short_window, len(returns1_common)):
        window_r1 = returns1_common.iloc[i-short_window:i]
        window_r2 = returns2_common.iloc[i-short_window:i]
        
        try:
            kendall_corr, _ = kendalltau(window_r1, window_r2)
            spearman_corr, _ = spearmanr(window_r1, window_r2)
            
            short_kendall.append(kendall_corr if not np.isnan(kendall_corr) else 0)
            short_spearman.append(spearman_corr if not np.isnan(spearman_corr) else 0)
        except:
            short_kendall.append(0)
            short_spearman.append(0)
    
    # 서브플롯 생성
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=['Kendall\'s Tau', 'Spearman\'s Rho']
    )
    
    # Kendall's tau
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=rolling_kendall,
            name=f'Long-term ({long_window}d)',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=dates[-len(short_kendall):],
            y=short_kendall,
            name=f'Short-term ({short_window}d)',
            line=dict(color='red', width=2)
        ),
        row=1, col=1
    )
    
    # Spearman's rho
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=rolling_spearman,
            name=f'Long-term ({long_window}d)',
            line=dict(color='blue', width=2),
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=dates[-len(short_spearman):],
            y=short_spearman,
            name=f'Short-term ({short_window}d)',
            line=dict(color='red', width=2),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # 제로 라인
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)
    
    fig.update_layout(
        title=f'Rolling Rank Correlations: {asset1} vs {asset2}',
        height=600,
        template='plotly_white'
    )
    
    fig.update_yaxes(title_text="Kendall's Tau", row=1, col=1)
    fig.update_yaxes(title_text="Spearman's Rho", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    
    return fig

def create_pair_chart(prices, asset1, asset2, formation_days, signal_days, asset_mapping=None):
    """페어 차트 생성 (코퓰라 방법론에 맞게 조정)"""
    # 전체 기간 데이터
    end_date = prices.index[-1]
    start_date = end_date - timedelta(days=int(formation_days * 1.4))  # 여유를 두고
    
    chart_data = prices.loc[start_date:end_date, [asset1, asset2]].dropna()
    
    if len(chart_data) == 0:
        st.error(f"데이터가 없습니다: {asset1}, {asset2}")
        return None
    
    # 가격 정규화 (리베이스)
    normalized_data = normalize_prices(chart_data, method='rebase')
    
    # 최근 6개월 기준점 계산
    six_months_ago = end_date - timedelta(days=180)
    
    # 스프레드 및 Z-스코어 계산
    recent_data = chart_data.tail(formation_days)
    normalized_recent = normalize_prices(recent_data, method='rebase')
    spread = calculate_spread(normalized_recent[asset1], normalized_recent[asset2], hedge_ratio=1.0)
    # Z-score 계산 - 안전한 윈도우 크기 사용
    zscore_window = min(signal_days, len(spread)//2) if len(spread) > 20 else max(20, len(spread)//4)
    zscore = calculate_zscore(spread, window=zscore_window)
    
    # 디버깅: Z-score 정보 출력 (개발용)
    if len(zscore.dropna()) == 0:
        st.error(f"Z-score 계산 오류: 스프레드 길이={len(spread)}, 윈도우={zscore_window}")
        return None
    
    # 서브플롯 생성
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.4, 0.3, 0.3],
        subplot_titles=[
            f'{asset1} vs {asset2} - 정규화된 가격 (코퓰라 기반)',
            'Spread (Price Difference)',
            'Z-Score (코퓰라·순위상관 신호)'
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
    
    # 2. 스프레드 차트
    spread_dates = spread.index
    fig.add_trace(
        go.Scatter(
            x=spread_dates,
            y=spread.values,
            name='Spread',
            line=dict(color='green', width=2),
            hovertemplate='<b>Spread</b><br>Date: %{x}<br>Value: %{y:.4f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 스프레드 제로 라인
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)
    
    # 3. Z-스코어 차트
    zscore_dates = zscore.dropna().index
    zscore_values = zscore.dropna().values
    
    fig.add_trace(
        go.Scatter(
            x=zscore_dates,
            y=zscore_values,
            name='Z-Score',
            line=dict(color='purple', width=2),
            hovertemplate='<b>Z-Score</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
        ),
        row=3, col=1
    )
    
    # Z-스코어 임계값 라인들
    fig.add_hline(y=2.0, line_dash="dash", line_color="orange", opacity=0.7, row=3, col=1)
    fig.add_hline(y=-2.0, line_dash="dash", line_color="orange", opacity=0.7, row=3, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=3, col=1)
    
    # 최근 6개월 배경색 강조
    for row in [1, 2, 3]:
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
        chart_title = f"코퓰라·순위상관 기반 페어분석: {name1}({asset1}) - {name2}({asset2})"
    else:
        chart_title = f"코퓰라·순위상관 기반 페어분석: {asset1} - {asset2}"
    
    # 레이아웃 설정
    fig.update_layout(
        height=800,
        title=chart_title,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )
    
    # 축 레이블 설정
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Normalized Price", row=1, col=1)
    fig.update_yaxes(title_text="Spread", row=2, col=1)
    fig.update_yaxes(title_text="Z-Score", row=3, col=1)
    
    # Z-스코어 임계값 주석
    current_zscore = zscore_values[-1] if len(zscore_values) > 0 else 0
    fig.add_annotation(
        x=zscore_dates[-1] if len(zscore_dates) > 0 else end_date,
        y=current_zscore,
        text=f"현재 Z-Score: {current_zscore:.2f}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="purple",
        font=dict(size=12, color="purple"),
        row=3, col=1
    )
    
    return fig

# 메인 앱
def main():
    st.title("코퓰라·순위상관 기반 페어트레이딩")
    st.markdown("---")
    
    # 코퓰라 순위상관 방법론 설명
    st.info("""
    ### 코퓰라·순위상관 기반 페어 선정 방법론
    
    **핵심 원리**: 선형 상관관계를 넘어 비선형 의존성과 극단상황에서의 동조현상(꼬리의존성)을 포착하여 더 정교한 페어를 선정하는 고도화된 전략
    
    **상세 작동 과정**:
    1. **순위상관 분석**: Pearson 대신 비모수적 순위상관 사용
       - **Kendall's τ**: 순서쌍의 일치도 측정, 극값에 덜 민감
       - **Spearman's ρ**: 순위 기반 선형관계, 단조증가 관계 포착
       - **장기 vs 단기**: 레짐 변화와 구조적 관계 변화 감지
    2. **꼬리 의존성 분석**: 극단상황에서의 공동움직임 측정
       - **하방 꼬리**: 동시 급락 시 의존성 (위기 시 동조현상)
       - **상방 꼬리**: 동시 급등 시 의존성 (호황 시 동조현상)
       - **비대칭성**: 상/하방 의존성 차이 (시장 충격 비대칭성 반영)
    3. **코퓰라 변환**: 수익률을 [0,1] 균등분포로 변환하여 순수 의존구조만 분석
       - 개별 분포 특성과 독립적인 의존성 구조 추출
       - 정규성 가정 불필요, 실제 데이터 분포 그대로 활용
    4. **품질 필터링**: 
       - **순위상관 변화**: 레짐 전환 감지 (최소 20% 변화)
       - **꼬리 의존성**: 극단상황 공동반응 강도
       - **일치성 비율**: 방향성 동조 빈도
    
    **핵심**: **비선형 의존성 + 꼬리 동조 + 레짐 전환 감지**를 통한 강건한 페어 발굴
    
    **장점**: 정규분포 가정 불필요, 극단위험 고려, 비선형 관계 포착, 시장 레짐 변화 감지
    """)
    
    # 사이드바 설정
    st.sidebar.header("분석 설정")
    st.sidebar.markdown("### 기간 설정")
    
    formation_days = st.sidebar.slider(
        "분석 기간 (일)",
        min_value=252,
        max_value=1260,  # 5년
        value=756,       # 3년
        step=126,        # 6개월 단위
        help="페어 선정을 위한 과거 데이터 기간"
    )
    
    signal_days = st.sidebar.slider(
        "Z-스코어 계산 기간 (일)",
        min_value=20,
        max_value=120,
        value=60,
        step=10,
        help="Z-스코어 신호 계산을 위한 롤링 윈도우"
    )
    
    st.sidebar.markdown("### 순위상관 설정")
    
    long_window = st.sidebar.slider(
        "장기 순위상관 윈도우 (일)",
        min_value=126,
        max_value=504,
        value=252,
        step=63,
        help="장기 추세 순위상관 계산 윈도우"
    )
    
    short_window = st.sidebar.slider(
        "단기 순위상관 윈도우 (일)",
        min_value=20,
        max_value=120,
        value=60,
        step=10,
        help="단기 변화 순위상관 계산 윈도우"
    )
    
    st.sidebar.markdown("### 신호 설정")
    
    enter_threshold = st.sidebar.slider(
        "진입 Z-스코어 임계값",
        min_value=1.5,
        max_value=3.0,
        value=2.0,
        step=0.1,
        help="이 값 이상일 때 진입 신호 생성"
    )
    
    st.sidebar.markdown("### 코퓰라 파라미터")
    
    min_rank_corr = st.sidebar.slider(
        "최소 순위상관",
        min_value=0.1,
        max_value=0.8,
        value=0.3,
        step=0.1,
        help="최소 장기 순위상관 임계값"
    )
    
    min_rank_corr_change = st.sidebar.slider(
        "최소 순위상관 변화",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.05,
        help="레짐 변화 감지를 위한 최소 상관계수 변화"
    )
    
    tail_quantile = st.sidebar.slider(
        "꼬리 분위수",
        min_value=0.05,
        max_value=0.20,
        value=0.10,
        step=0.01,
        help="꼬리 의존성 계산용 극단값 비율"
    )
    
    n_pairs = st.sidebar.slider(
        "분석할 페어 수",
        min_value=5,
        max_value=20,
        value=10,
        step=1,
        help="상위 몇 개 페어를 분석할지 설정"
    )
    
    # 파라미터 딕셔너리
    params = {
        'formation_window': formation_days,
        'signal_window': signal_days,
        'long_window': long_window,
        'short_window': short_window,
        'enter_threshold': enter_threshold,
        'exit_threshold': 0.5,
        'stop_loss': 3.0,
        'min_half_life': 5,
        'max_half_life': 60,
        'min_cost_ratio': 5.0,
        'min_rank_corr': min_rank_corr,
        'min_rank_corr_change': min_rank_corr_change,
        'tail_quantile': tail_quantile
    }
    
    # 기본값 여부 확인
    is_default = check_parameters_default(params)
    
    # 분석 실행 버튼
    if st.sidebar.button("분석 실행", type="primary"):
        st.cache_data.clear()  # 캐시 클리어
    
    # 메인 콘텐츠
    if is_default:
        st.info("🚀 기본 파라미터를 사용 중. 사전 계산된 결과를 즉시 표시")
        
        # 캐시에서 결과 로드
        cache_data = cache_utils.load_cache('copula')
        if cache_data:
            enter_list = cache_data.get('enter_signals', [])
            watch_list = cache_data.get('watch_signals', [])
            prices = load_price_data()
            asset_mapping = load_asset_names()  # 자산 이름 매핑 로딩
        else:
            st.error("캐시 데이터를 찾을 수 없음. 실시간 분석을 실행")
            with st.spinner("코퓰라·순위상관 기반 페어 분석 중... 잠시만 기다려주세요."):
                try:
                    enter_list, watch_list, prices = analyze_pairs(
                        formation_days, signal_days, long_window, short_window, enter_threshold, n_pairs,
                        min_rank_corr, min_rank_corr_change, tail_quantile
                    )
                    asset_mapping = load_asset_names()  # 자산 이름 매핑 로딩
                except Exception as e:
                    st.error(f"분석 중 오류 발생: {str(e)}")
                    return
    else:
        st.warning("⚙️ 사용자 정의 파라미터가 설정")
        
        if st.button("🚀 분석 실행", type="primary"):
            with st.spinner("코퓰라·순위상관 기반 페어 분석 중... 잠시만 기다려주세요."):
                try:
                    enter_list, watch_list, prices = analyze_pairs(
                        formation_days, signal_days, long_window, short_window, enter_threshold, n_pairs,
                        min_rank_corr, min_rank_corr_change, tail_quantile
                    )
                    asset_mapping = load_asset_names()  # 자산 이름 매핑 로딩
                except Exception as e:
                    st.error(f"분석 중 오류 발생: {str(e)}")
                    return
        else:
            return
    
    # 분석 결과 요약
    st.header("분석 결과 요약")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("진입 신호", f"{len(enter_list)}개", help="Z-스코어 임계값 이상의 페어")
    
    with col2:
        st.metric("관찰 대상", f"{len(watch_list)}개", help="진입 직전 단계의 페어")
    
    with col3:
        avg_tail_dep = np.mean([s['tail_total'] for s in enter_list]) if enter_list else 0
        st.metric("평균 꼬리의존성", f"{avg_tail_dep:.3f}", help="진입 신호들의 평균 꼬리의존성")
        
    with col4:
        avg_copula_score = np.mean([s['copula_score'] for s in enter_list]) if enter_list else 0
        st.metric("평균 코퓰라점수", f"{avg_copula_score:.1f}", help="진입 신호들의 평균 코퓰라 품질점수")
    
    st.markdown("---")
    
    # 진입 신호 테이블
    if enter_list:
        st.header("추천 진입 페어")
        
        # 테이블 데이터 준비
        table_data = []
        for i, signal in enumerate(enter_list, 1):
            formatted_pair = format_pair_name(signal['pair'], asset_mapping)
            table_data.append({
                "순위": i,
                "페어": formatted_pair,
                "방향": signal['direction'],
                "Z-Score": f"{signal['current_zscore']:.2f}",
                "코퓰라점수": f"{signal['copula_score']:.1f}",
                "꼬리의존성": f"{signal['tail_total']:.3f}",
                "순위상관(τ)": f"{signal['tau_long']:.3f}",
                "Half-Life": f"{signal['half_life']:.1f}일"
            })
        
        df_enter = pd.DataFrame(table_data)
        
        # 스타일링된 테이블 표시
        st.dataframe(
            df_enter,
            use_container_width=True,
            hide_index=True,
            column_config={
                "순위": st.column_config.NumberColumn("순위", width="small"),
                "페어": st.column_config.TextColumn("페어", width="large"),
                "방향": st.column_config.TextColumn("진입 방향", width="large"),
                "Z-Score": st.column_config.TextColumn("Z-Score", width="small"),
                "코퓰라점수": st.column_config.TextColumn("코퓰라점수", width="small"),
                "꼬리의존성": st.column_config.TextColumn("꼬리의존성", width="small"),
                "순위상관(τ)": st.column_config.TextColumn("순위상관(τ)", width="small"),
                "Half-Life": st.column_config.TextColumn("Half-Life", width="small")
            }
        )
        
        st.markdown("---")
        
        # 페어 선택 및 차트 표시
        st.header("페어 상세 분석")
        
        # 최고 추천 페어 표시
        top_pair = enter_list[0]
        top_formatted_pair = format_pair_name(top_pair['pair'], asset_mapping)
        st.success(f"최고 추천 페어 (코퓰라점수: {top_pair['copula_score']:.1f}): {top_formatted_pair}")
        
        # 페어 선택 옵션 (표시는 포맷팅된 이름, 값은 원래 페어)
        pair_options = [signal['pair'] for signal in enter_list]
        pair_display_names = [format_pair_name(signal['pair'], asset_mapping) for signal in enter_list]
        
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
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("진입 방향", selected_pair_info['direction'])
            with col2:
                st.metric("현재 Z-Score", f"{selected_pair_info['current_zscore']:.2f}")
            with col3:
                st.metric("코퓰라 점수", f"{selected_pair_info['copula_score']:.1f}")
            with col4:
                st.metric("꼬리 의존성", f"{selected_pair_info['tail_total']:.3f}")
            with col5:
                st.metric("순위상관 변화", f"{selected_pair_info.get('current_delta_tau', 0):.3f}")
        
        if selected_pair:
            asset1, asset2 = selected_pair.split('-')
            
            # 메인 페어 차트
            with st.spinner(f"{selected_display_pair} 차트 생성 중..."):
                fig = create_pair_chart(prices, asset1, asset2, formation_days, signal_days, asset_mapping)
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            # 코퓰라 특화 분석 차트들
            st.subheader("코퓰라 상세 분석")
            
            # 3개 열로 나누어 차트 표시
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**코퓰라 산점도 (Uniform 변환)**")
                copula_fig = create_copula_scatter(prices, asset1, asset2, formation_days)
                if copula_fig:
                    st.plotly_chart(copula_fig, use_container_width=True)
            
            with col2:
                st.markdown("**꼬리 의존성 분석**")
                tail_fig = create_tail_dependence_chart(prices, asset1, asset2, formation_days, tail_quantile)
                if tail_fig:
                    st.plotly_chart(tail_fig, use_container_width=True)
            
            st.markdown("**순위상관 시계열**")
            rank_corr_fig = create_rank_correlation_chart(prices, asset1, asset2, formation_days, long_window, short_window)
            if rank_corr_fig:
                st.plotly_chart(rank_corr_fig, use_container_width=True)
            
            # 차트 설명
            st.info("""
            **코퓰라·순위상관 기반 차트 설명:**
            - **메인 차트**: 정규화된 가격, 스프레드, Z-스코어 (코퓰라 필터링 적용)
            - **코퓰라 산점도**: Uniform 변환 후 순수 의존구조, 색상은 원수익률
            - **꼬리 의존성**: 극단상황(상/하위 10%) 동시발생 분석, 빨강=하방꼬리, 초록=상방꼬리
            - **순위상관 시계열**: 장기/단기 Kendall τ, Spearman ρ 변화 추이
            - **특징**: 비선형 의존성과 극단위험 동조현상을 종합적으로 고려한 고도화 분석
            """)
    
    else:
        st.warning("현재 진입 조건을 만족하는 페어가 없음")
        st.info("순위상관 임계값을 낮추거나 꼬리의존성 조건을 완화해야함")
    
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
                "코퓰라점수": f"{signal['copula_score']:.1f}",
                "꼬리의존성": f"{signal['tail_total']:.3f}",
                "순위상관(τ)": f"{signal['tau_long']:.3f}",
                "Half-Life": f"{signal['half_life']:.1f}일"
            })
        
        df_watch = pd.DataFrame(table_data)
        st.dataframe(df_watch, use_container_width=True, hide_index=True)
    
    # 캐시 정보 표시
    if is_default and 'cache_data' in locals() and cache_data:
        st.markdown("---")
        st.caption(f"📅 캐시 생성일: {cache_data.get('generated_at', 'Unknown')}")
        st.caption(f"📊 데이터 기준일: {cache_data.get('data_date', 'Unknown')}")
    
    # 푸터
    st.markdown("---")

# Streamlit 페이지로 실행
main()