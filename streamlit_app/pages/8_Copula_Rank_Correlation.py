"""
Pair Trading Analysis Tool - Copula Rank Correlation Methodology
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
CopulaBasedPairScreening = copula_module.CopulaBasedPairScreening

# 페이지 설정
st.set_page_config(
    page_title="Copula Rank Correlation Methodology",
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
def analyze_pairs(formation_window, min_tail_dependence, conditional_prob_threshold, 
                  min_kendall_tau, min_data_coverage, copula_consistency_threshold, n_pairs):
    """페어 분석 실행"""
    prices = load_price_data()
    
    screener = CopulaBasedPairScreening(
        formation_window=formation_window,
        min_tail_dependence=min_tail_dependence,
        conditional_prob_threshold=conditional_prob_threshold,
        min_kendall_tau=min_kendall_tau,
        min_data_coverage=min_data_coverage,
        copula_consistency_threshold=copula_consistency_threshold
    )
    
    selected_pairs = screener.select_pairs(prices, n_pairs=n_pairs)
    
    # 결과를 enter_list, watch_list 형태로 변환
    enter_list = []
    watch_list = []
    
    for pair_info in selected_pairs:
        signal_type = pair_info.get('signal_type', 'NEUTRAL')
        pair_data = {
            'pair': pair_info['pair'],
            'current_zscore': pair_info.get('current_zscore', 0),
            'direction': 'LONG' if signal_type == 'LONG' else 'SHORT' if signal_type == 'SHORT' else 'NEUTRAL',
            'copula_score': pair_info.get('copula_score', 0),
            'tail_dependence': pair_info.get('tail_dependence_max', 0),
            'kendall_tau': pair_info.get('kendall_tau', 0),
            'conditional_prob': pair_info.get('conditional_prob', 0),
            'copula_family': pair_info.get('copula_family', 'N/A'),
            'consistency': pair_info.get('copula_consistency', 0)
        }
        
        if signal_type in ['LONG', 'SHORT']:
            enter_list.append(pair_data)
        else:
            watch_list.append(pair_data)
    
    return enter_list, watch_list, selected_pairs, prices

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
        height=500,
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

def create_rank_correlation_chart(prices, asset1, asset2, formation_days, long_window=250, short_window=60):
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
    dates = []
    
    for i in range(long_window, len(returns1_common)):
        window_r1 = returns1_common.iloc[i-long_window:i]
        window_r2 = returns2_common.iloc[i-long_window:i]
        
        try:
            kendall_corr, _ = kendalltau(window_r1, window_r2)
            rolling_kendall.append(kendall_corr if not np.isnan(kendall_corr) else 0)
            dates.append(returns1_common.index[i])
        except:
            rolling_kendall.append(0)
            dates.append(returns1_common.index[i])
    
    # 단기 롤링 상관계수
    short_kendall = []
    
    for i in range(short_window, len(returns1_common)):
        window_r1 = returns1_common.iloc[i-short_window:i]
        window_r2 = returns2_common.iloc[i-short_window:i]
        
        try:
            kendall_corr, _ = kendalltau(window_r1, window_r2)
            short_kendall.append(kendall_corr if not np.isnan(kendall_corr) else 0)
        except:
            short_kendall.append(0)
    
    # 차트 생성
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=rolling_kendall,
            name=f'Long-term ({long_window}d)',
            line=dict(color='blue', width=2)
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=dates[-len(short_kendall):],
            y=short_kendall,
            name=f'Short-term ({short_window}d)',
            line=dict(color='red', width=2)
        )
    )
    
    # 제로 라인
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=f'Rolling Kendall\'s Tau: {asset1} vs {asset2}',
        xaxis_title='Date',
        yaxis_title='Kendall\'s Tau',
        height=400,
        template='plotly_white'
    )
    
    return fig

def create_pair_chart(prices, asset1, asset2, formation_days, signal_days, asset_mapping=None):
    """페어 차트 생성 (코퓰라 방법론에 맞게 조정)"""
    # 전체 기간 데이터
    end_date = prices.index[-1]
    start_date = end_date - timedelta(days=int(formation_days * 1.4))
    
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
    zscore_window = min(signal_days, len(spread)//2) if len(spread) > 20 else max(20, len(spread)//4)
    zscore = calculate_zscore(spread, window=zscore_window)
    
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
    st.title("🎲 코퓰라·순위상관 기반 페어트레이딩")
    st.markdown("---")
    
    # 4개 탭 구성
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 분석 결과 요약",
        "📊 상세 작동 과정", 
        "📝 상세 설명",
        "🔍 수식 및 계산"
    ])
    
    # 사이드바 구성
    st.sidebar.header("⚙️ 분석 설정")
    
    formation_window = st.sidebar.slider(
        "형성 기간 (일)",
        min_value=1000,
        max_value=4000,
        value=3000,
        step=250,
        help="12년 형성기간 (≈3000 영업일)"
    )
    
    min_tail_dependence = st.sidebar.slider(
        "최소 꼬리 의존성",
        min_value=0.05,
        max_value=0.3,
        value=0.1,
        step=0.01,
        help="극단 상황 동조성 최소값 (≥0.1)"
    )
    
    conditional_prob_threshold = st.sidebar.slider(
        "조건부 확률 임계값",
        min_value=0.01,
        max_value=0.15,
        value=0.05,
        step=0.01,
        help="미스프라이싱 신호 임계값 (5% 또는 95%)"
    )
    
    min_kendall_tau = st.sidebar.slider(
        "최소 켄달 타우 상관계수",
        min_value=0.1,
        max_value=0.8,
        value=0.3,
        step=0.05,
        help="페어 선정을 위한 최소 상관계수"
    )
    
    # 분석 실행 버튼
    if st.sidebar.button("🚀 분석 실행", type="primary"):
        st.cache_data.clear()
    
    # 추가 파라미터 (숨김)
    min_data_coverage = 0.85
    copula_consistency_threshold = 0.8
    n_pairs = 10
    signal_days = 60
    
    # 파라미터 딕셔너리
    params = {
        'formation_window': formation_window,
        'min_tail_dependence': min_tail_dependence,
        'conditional_prob_threshold': conditional_prob_threshold,
        'min_kendall_tau': min_kendall_tau,
        'min_data_coverage': min_data_coverage,
        'copula_consistency_threshold': copula_consistency_threshold
    }
    
    # 기본값 여부 확인
    is_default = check_parameters_default(params)
    
    # 공통 분석 수행
    with st.spinner("🎲 코퓰라·순위상관 기반 페어 분석 중..."):
        try:
            if is_default:
                # 캐시에서 결과 로드 시도
                cache_data = cache_utils.load_cache('copula')
                if cache_data:
                    enter_list = cache_data.get('enter_signals', [])
                    watch_list = cache_data.get('watch_signals', [])
                    selected_pairs = enter_list + watch_list  # 임시 통합
                    prices = load_price_data()
                    asset_mapping = load_asset_names()
                else:
                    # 캐시 실패시 실시간 분석
                    enter_list, watch_list, selected_pairs, prices = analyze_pairs(
                        formation_window, min_tail_dependence, conditional_prob_threshold,
                        min_kendall_tau, min_data_coverage, copula_consistency_threshold, n_pairs
                    )
                    asset_mapping = load_asset_names()
            else:
                # 사용자 정의 파라미터 사용
                enter_list, watch_list, selected_pairs, prices = analyze_pairs(
                    formation_window, min_tail_dependence, conditional_prob_threshold,
                    min_kendall_tau, min_data_coverage, copula_consistency_threshold, n_pairs
                )
                asset_mapping = load_asset_names()
        except Exception as e:
            st.error(f"분석 중 오류 발생: {str(e)}")
            return
    
    # TAB 1: 📈 분석 결과 요약
    with tab1:
        # 분석 결과 메트릭 (4개 컬럼)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("진입 신호 개수", f"{len(enter_list)}개", help="조건부 확률 임계값을 넘은 페어")
        
        with col2:
            st.metric("관찰 대상 개수", f"{len(watch_list)}개", help="12년 형성기간에서 선별된 페어")
        
        with col3:
            avg_tail_dep = np.mean([pair.get('tail_dependence', 0) for pair in enter_list + watch_list]) if enter_list or watch_list else 0
            st.metric("평균 꼬리의존성", f"{avg_tail_dep:.3f}", help="선별된 페어들의 평균 꼬리의존성")
        
        with col4:
            avg_kendall_tau = np.mean([pair.get('kendall_tau', 0) for pair in enter_list + watch_list]) if enter_list or watch_list else 0
            st.metric("평균 켄달 타우", f"{avg_kendall_tau:.3f}", help="선별된 페어들의 평균 상관계수")
        
        st.markdown("---")
        
        # 추천 진입 페어 테이블
        if enter_list:
            st.subheader("✅ 추천 진입 페어")
            
            table_data = []
            for i, signal in enumerate(enter_list, 1):
                formatted_pair = format_pair_name(signal['pair'], asset_mapping)
                table_data.append({
                    "순위": i,
                    "페어": formatted_pair,
                    "Z-Score": f"{signal['current_zscore']:.2f}",
                    "방향": signal['direction'],
                    "코퓰라": signal.get('copula_family', 'N/A'),
                    "꼬리의존성": f"{signal.get('tail_dependence', 0):.3f}",
                    "켄달타우": f"{signal.get('kendall_tau', 0):.3f}",
                    "조건부확률": f"{signal.get('conditional_prob', 0):.3f}"
                })
            
            df_enter = pd.DataFrame(table_data)
            st.dataframe(
                df_enter,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "순위": st.column_config.NumberColumn("순위", width="small"),
                    "페어": st.column_config.TextColumn("페어", width="large"),
                    "Z-Score": st.column_config.TextColumn("Z-Score", width="small"),
                    "방향": st.column_config.TextColumn("진입방향", width="medium"),
                    "코퓰라": st.column_config.TextColumn("코퓰라", width="small"),
                    "꼬리의존성": st.column_config.TextColumn("꼬리의존성", width="small"),
                    "켄달타우": st.column_config.TextColumn("켄달타우", width="small"),
                    "조건부확률": st.column_config.TextColumn("조건부확률", width="small")
                }
            )
        else:
            st.warning("❌ 현재 진입 조건을 만족하는 페어가 없습니다")
        
        # 관찰 대상 페어 테이블
        if watch_list:
            st.subheader("⭐ 관찰 대상 페어")
            
            table_data = []
            for i, signal in enumerate(watch_list, 1):
                formatted_pair = format_pair_name(signal['pair'], asset_mapping)
                table_data.append({
                    "순위": i,
                    "페어": formatted_pair,
                    "Z-Score": f"{signal['current_zscore']:.2f}",
                    "상태": "관찰중",
                    "코퓰라": signal.get('copula_family', 'N/A'),
                    "꼬리의존성": f"{signal.get('tail_dependence', 0):.3f}"
                })
            
            df_watch = pd.DataFrame(table_data)
            st.dataframe(
                df_watch,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "순위": st.column_config.NumberColumn("순위", width="small"),
                    "페어": st.column_config.TextColumn("페어", width="large"),
                    "Z-Score": st.column_config.TextColumn("Z-Score", width="small"),
                    "상태": st.column_config.TextColumn("상태", width="small"),
                    "코퓰라": st.column_config.TextColumn("코퓰라", width="small"),
                    "꼬리의존성": st.column_config.TextColumn("꼬리의존성", width="small")
                }
            )
        
        st.markdown("---")
        
        # 🔍 페어 상세 분석 (필수 섹션)
        st.subheader("🔍 페어 상세 분석")
        
        if enter_list or watch_list:
            # 통합 드롭다운 (진입+관찰)
            all_pairs = enter_list + watch_list
            pair_options = [signal['pair'] for signal in all_pairs]
            pair_display_names = [format_pair_name(signal['pair'], asset_mapping) for signal in all_pairs]
            pair_mapping = {display: original for display, original in zip(pair_display_names, pair_options)}
            
            selected_display_pair = st.selectbox(
                "분석할 페어 선택:",
                options=pair_display_names,
                index=0,
                help="차트로 분석할 페어를 선택하세요"
            )
            
            # 선택 페어 메트릭 (4개 컬럼)
            selected_pair = pair_mapping[selected_display_pair]
            selected_pair_info = None
            
            for signal in all_pairs:
                if signal['pair'] == selected_pair:
                    selected_pair_info = signal
                    break
            
            if selected_pair_info:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("현재 Z-Score", f"{selected_pair_info['current_zscore']:.2f}")
                with col2:
                    st.metric("진입 방향", selected_pair_info.get('direction', 'NEUTRAL'))
                with col3:
                    st.metric("꼬리 의존성", f"{selected_pair_info.get('tail_dependence', 0):.3f}")
                with col4:
                    st.metric("켄달 타우", f"{selected_pair_info.get('kendall_tau', 0):.3f}")
            
            # 인터랙티브 차트
            if selected_pair:
                asset1, asset2 = selected_pair.split('-')
                
                with st.spinner(f"📊 {selected_display_pair} 차트 생성 중..."):
                    fig = create_pair_chart(prices, asset1, asset2, formation_window, signal_days, asset_mapping)
                    
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                # 코퓰라 특화 시각화
                st.subheader("🎲 코퓰라 특화 분석")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    copula_fig = create_copula_scatter(prices, asset1, asset2, formation_window)
                    if copula_fig:
                        st.plotly_chart(copula_fig, use_container_width=True)
                
                with col2:
                    tail_fig = create_tail_dependence_chart(prices, asset1, asset2, formation_window)
                    if tail_fig:
                        st.plotly_chart(tail_fig, use_container_width=True)
                
                # 순위상관 시계열
                rank_corr_fig = create_rank_correlation_chart(prices, asset1, asset2, formation_window)
                if rank_corr_fig:
                    st.plotly_chart(rank_corr_fig, use_container_width=True)
                
                # 차트 해석 가이드
                with st.expander("📖 차트 해석 가이드"):
                    st.info("""
                    **🎲 코퓰라·순위상관 기반 차트 해석:**
                    - **상단 차트**: 정규화된 가격, 스프레드, Z-스코어 (12년 코퓰라 필터링 적용)
                    - **코퓰라 산점도**: Uniform 변환 후 순수 의존구조, 색상은 원수익률
                    - **꼬리 의존성**: 극단상황(상/하위 10%) 동시발생 분석, 빨강=하방꼬리, 초록=상방꼬리
                    - **순위상관 시계열**: 장기/단기 Kendall τ 변화 추이
                    
                    **💡 신호 해석:**
                    - 노란색 배경: 최근 6개월 기간 (신호 발생 구간)
                    - 주황색 선: 진입 임계값 (±2.0)
                    - 조건부 확률이 5% 미만 또는 95% 초과시 진입 신호
                    
                    **🎯 코퓰라 특징:**
                    - 12년 장기 형성기간으로 안정적 의존구조 파악
                    - 비선형 의존성과 극단위험 동조현상 종합 고려
                    - 6가지 분포 × 5가지 코퓰라 = 30가지 조합에서 최적 선택
                    """)
        else:
            st.info("💡 분석할 페어가 없습니다. 임계값을 조정해보세요.")
    
    # TAB 2: 📊 상세 작동 과정
    with tab2:
        st.header("코퓰라·순위상관 기반 페어트레이딩 작동 과정")
        
        # STEP 1
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("### STEP 1")
            st.info("**12년 장기 형성기간 설정**")
        with col2:
            st.markdown(f"""
            **{formation_window}일 (≈12년) 안정성 검증**
            
            - **장기 안정성**: 여러 경제 사이클에 걸친 구조적 관계 검증
            - **데이터 품질**: {min_data_coverage:.0%} 이상 커버리지 요구 (12년 중 {min_data_coverage*12:.0f}년+ 데이터)
            - **노이즈 제거**: 단기 변동성을 넘어선 본질적 의존성 추출
            
            12년이라는 장기간을 통해 일시적 상관관계가 아닌 구조적 관계만 선별합니다.
            """)
        
        # STEP 2
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("### STEP 2")
            st.warning("**주변분포 및 코퓰라 적합**")
        with col2:
            st.markdown("""
            **6가지 주변분포 + 5가지 코퓰라 패밀리**
            
            주변분포 적합:
            - Normal, Student-t, Logistic, Laplace, Skewed Normal, GEV
            - AIC/BIC/HQIC 기준으로 최적 분포 자동 선택
            - KS 검정으로 적합도 검증
            
            코퓰라 패밀리:
            - Gaussian, Student-t, Gumbel, Clayton, Frank
            - 각각 다른 의존성 패턴 (대칭/비대칭, 꼬리의존성 등)
            """)
        
        # STEP 3
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("### STEP 3")
            st.success("**꼬리 의존성 검증**")
        with col2:
            st.markdown(f"""
            **극단상황 동조성 분석**
            
            - **최소 꼬리 의존성**: {min_tail_dependence:.2f} 이상
            - **상방 꼬리**: 동시 극단 상승 확률
            - **하방 꼬리**: 동시 극단 하락 확률 (위험 관리)
            - **일관성 검증**: 롤링 기간 내 {copula_consistency_threshold:.0%} 이상 동일 코퓰라
            
            단순 선형 상관을 넘어 극단상황에서의 동조성을 정량적으로 측정합니다.
            """)
        
        # STEP 4
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("### STEP 4")
            st.error("**조건부 확률 미스프라이싱 신호**")
        with col2:
            st.markdown(f"""
            **실시간 미스프라이싱 탐지**
            
            - **P(U≤u|V=v) = ∂C(u,v)/∂v**: 조건부 확률 계산
            - **신호 임계값**: {conditional_prob_threshold:.1%} 또는 {100-conditional_prob_threshold*100:.1%}% 이탈시
            - **현재 포지션**: 12년 데이터 기준 상대적 위치 평가
            - **최소 상관계수**: 켄달 타우 {min_kendall_tau:.2f} 이상
            
            코퓰라 기반 조건부 확률로 현재 미스프라이싱 정도를 실시간 측정합니다.
            """)
        
        # 마무리 요소
        st.success("""
        **🎯 코퓰라·순위상관 방법론의 핵심 전략**
        
        12년 장기 형성기간으로 안정적 의존구조를 파악하고, 조건부 확률을 통해 현재 미스프라이싱된 페어를 
        실시간 발굴하는 고도화된 전략입니다. 극단위험까지 고려한 종합적 리스크 관리가 핵심입니다.
        """)
        
        # 방법론별 시각화 (2개 컬럼)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("코퓰라 패밀리 분포")
            if enter_list or watch_list:
                all_pairs = enter_list + watch_list
                copula_counts = {}
                for pair in all_pairs:
                    copula = pair.get('copula_family', 'Unknown')
                    copula_counts[copula] = copula_counts.get(copula, 0) + 1
                
                for copula, count in copula_counts.items():
                    st.metric(f"{copula} 코퓰라", f"{count}개", f"전체의 {count/len(all_pairs)*100:.0f}%")
        
        with col2:
            st.subheader("꼬리의존성 분포")
            if enter_list or watch_list:
                all_pairs = enter_list + watch_list
                tail_deps = [pair.get('tail_dependence', 0) for pair in all_pairs]
                if tail_deps:
                    st.metric("최대 꼬리의존성", f"{max(tail_deps):.3f}")
                    st.metric("최소 꼬리의존성", f"{min(tail_deps):.3f}")
                    st.metric("평균 꼬리의존성", f"{np.mean(tail_deps):.3f}")
                    st.metric("표준편차", f"{np.std(tail_deps):.3f}")
    
    # TAB 3: 📝 상세 설명
    with tab3:
        st.info("""
        ### 코퓰라·순위상관 기반 페어 선정 방법론

        **핵심 원리**: 12년 장기 형성기간으로 안정적 의존성 구조를 파악하고, 조건부 확률을 통해 현재 미스프라이싱된 페어를 실시간 발굴하는 고도화된 전략

        **상세 작동 과정**:
        1. **12년 형성기간 (≈3000일)**: 
           - **장기 안정성**: 여러 경제 사이클에 걸친 구조적 관계 검증
           - **데이터 품질**: 85% 이상 커버리지 요구 (12년 중 10년+ 데이터)
           - **노이즈 제거**: 단기 변동성을 넘어선 본질적 의존성 추출
        2. **6가지 주변분포 적합**:
           - **Normal, Student-t, Logistic, Laplace, Skewed Normal, GEV**
           - **자동 선택**: AIC/BIC/HQIC 기준으로 최적 분포 선택
           - **품질 평가**: KS 검정으로 적합도 검증
        3. **5가지 코퓰라 패밀리**:
           - **Gaussian, Student-t, Gumbel, Clayton, Frank**
           - **꼬리 의존성**: 상/하방 극단상황 동조성 ≥ 0.1
           - **일관성 검증**: 롤링 기간 내 80% 이상 동일 코퓰라
        4. **조건부 확률 미스프라이싱**: 
           - **P(U≤u|V=v) = ∂C(u,v)/∂v**: 한 자산 조건부 다른 자산 확률
           - **실시간 신호**: 5% 또는 95% 이탈 시 진입 신호
           - **현재 포지션**: 12년 데이터 기준 상대적 위치 평가

        **핵심**: **12년 일관성 + 조건부 확률 + 꼬리 의존성**을 통한 고품질 실시간 페어 발굴

        **장점**: 장기 안정성 검증, 실시간 미스프라이싱 감지, 극단위험 고려, 코퓰라 일관성 보장
        
        **특별한 특징**:
        - 비선형 의존성 구조 완전 포착 (순위상관 기반)
        - 극단위험 상황의 동조성 정량 측정
        - 30가지 분포-코퓰라 조합에서 최적 선택
        - 조건부 확률 기반 실시간 미스프라이싱 신호
        
        **적용 시나리오**:
        - 장기 구조적 관계가 중요한 자산 클래스
        - 극단위험 관리가 핵심인 포트폴리오
        - 비선형 의존성이 강한 금융상품 간 관계
        - 고도화된 정량적 리스크 관리 필요시
        """)
    
    # TAB 4: 🔍 수식 및 계산
    with tab4:
        # 2개 컬럼 레이아웃
        col1, col2 = st.columns(2)
        
        # 왼쪽: 핵심 수식
        with col1:
            st.subheader("핵심 수식")
            
            st.markdown("**1. 코퓰라 함수**")
            st.latex(r'''
            C(u_1, u_2) = P(U_1 \leq u_1, U_2 \leq u_2)
            ''')
            
            st.markdown("**2. 조건부 확률**")
            st.latex(r'''
            P(U_1 \leq u_1 | U_2 = u_2) = \frac{\partial C(u_1, u_2)}{\partial u_2}
            ''')
            
            st.markdown("**3. 꼬리 의존성**")
            st.latex(r'''
            \lambda_L = \lim_{u \to 0^+} P(U_2 \leq u | U_1 \leq u)
            ''')
        
        # 오른쪽: 보조 수식
        with col2:
            st.subheader("보조 수식")
            
            st.markdown("**1. 켄달 타우**")
            st.latex(r'''
            \tau = P((X_1-Y_1)(X_2-Y_2) > 0) - P((X_1-Y_1)(X_2-Y_2) < 0)
            ''')
            
            st.markdown("**2. 경험적 코퓰라**")
            st.latex(r'''
            C_n(u_1, u_2) = \frac{1}{n} \sum_{i=1}^{n} \mathbf{1}_{R_i^{(1)}/n \leq u_1, R_i^{(2)}/n \leq u_2}
            ''')
            
            st.markdown("**3. 스피어만 상관계수**")
            st.latex(r'''
            \rho_S = 12 \int_0^1 \int_0^1 C(u_1, u_2) du_1 du_2 - 3
            ''')
        
        st.markdown("---")
        
        # 실제 계산 예시
        col1, col2 = st.columns(2)
        
        # 왼쪽: Python 코드 예시  
        with col1:
            st.subheader("Python 구현 예시")
            
            st.code("""
# 코퓰라 적합 예시
import numpy as np
from scipy import stats
from scipy.stats import kendalltau

def fit_copula(u1, u2, copula_type='gaussian'):
    \"\"\"코퓰라 적합\"\"\"
    if copula_type == 'gaussian':
        # 정규 코퓰라
        norm_u1 = stats.norm.ppf(u1)
        norm_u2 = stats.norm.ppf(u2)
        correlation = np.corrcoef(norm_u1, norm_u2)[0,1]
        return correlation
    
    elif copula_type == 'gumbel':
        # 검벨 코퓰라 (Clayton과 유사한 방식)
        tau, _ = kendalltau(u1, u2)
        theta = 1 / (1 - tau)
        return theta

def conditional_probability(u1, u2, theta, copula_type):
    \"\"\"조건부 확률 계산\"\"\"
    if copula_type == 'gaussian':
        # 정규 코퓰라 조건부 확률
        norm_u1 = stats.norm.ppf(u1)
        norm_u2 = stats.norm.ppf(u2)
        
        conditional_mean = theta * norm_u2
        conditional_std = np.sqrt(1 - theta**2)
        
        return stats.norm.cdf((norm_u1 - conditional_mean) / conditional_std)

# 꼬리 의존성 추정
def tail_dependence(u1, u2, quantile=0.1):
    \"\"\"경험적 꼬리 의존성\"\"\"
    # 하방 꼬리
    lower_mask = (u1 <= quantile) & (u2 <= quantile)
    lambda_lower = lower_mask.sum() / (u1 <= quantile).sum()
    
    # 상방 꼬리
    upper_mask = (u1 >= 1-quantile) & (u2 >= 1-quantile)
    lambda_upper = upper_mask.sum() / (u1 >= 1-quantile).sum()
    
    return lambda_lower, lambda_upper
            """)
        
        # 오른쪽: 해석 및 활용법
        with col2:
            st.subheader("해석 및 활용법")
            
            st.markdown("""
            **코퓰라 해석:**
            - **조건부 확률**: 한 자산이 특정 위치일 때 다른 자산의 기대 위치
            - **꼬리 의존성**: 극단상황에서의 동시 발생 확률
            - **켄달 타우**: 순위 기반 상관계수 (분포에 무관)
            
            **실전 적용:**
            - **신호 생성**: 조건부 확률이 5% 미만 또는 95% 초과시 진입
            - **위험 관리**: 꼬리 의존성으로 극단위험 평가
            - **페어 선별**: 켄달 타우 0.3 이상의 안정적 관계
            
            **코퓰라 선택 기준:**
            - **Gaussian**: 대칭적, 꼬리독립성
            - **Student-t**: 대칭적, 꼬리의존성
            - **Gumbel**: 상방 꼬리의존성 (동반 상승)
            - **Clayton**: 하방 꼬리의존성 (동반 하락)
            - **Frank**: 중앙 의존성, 꼬리독립성
            
            **성과 모니터링:**
            - 조건부 확률 신호의 적중률 추적
            - 꼬리 의존성 안정성 모니터링
            - 코퓰라 적합도 정기 검증 (KS 검정)
            """)

if __name__ == "__main__":
    main()