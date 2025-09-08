"""
페어트레이딩 분석 도구 - SSD 거리 기반 방법론
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

# 프로젝트 루트 경로 추가
project_root = setup_path()

# 동적 모듈 import
def import_module_from_file(file_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# 모듈 import
common_utils = import_module_from_file(os.path.join(project_root, "utils/common_utils.py"), "common_utils")
ssd_module = import_module_from_file(os.path.join(project_root, "methods/8_ssd_distance_pairs.py"), "ssd_distance_pairs")

# 필요한 함수들 import
load_data = common_utils.load_data
normalize_prices = common_utils.normalize_prices
calculate_spread = common_utils.calculate_spread
calculate_zscore = common_utils.calculate_zscore
SSDDistancePairTrading = ssd_module.SSDDistancePairTrading

# 페이지 설정
st.set_page_config(
    page_title="SSD 거리 방법론",
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
    """페어 이름을 티커(이름) 형태로 포맷팅"""
    asset1, asset2 = pair.split('-')
    
    name1 = asset_mapping.get(asset1, asset1)
    name2 = asset_mapping.get(asset2, asset2)
    
    return f"{asset1}({name1}) - {asset2}({name2})"

# 페어 분석 함수
@st.cache_data
def analyze_pairs(formation_days, signal_days, enter_threshold, n_pairs):
    """페어 분석 실행"""
    prices = load_price_data()
    
    trader = SSDDistancePairTrading(
        formation_window=formation_days,
        signal_window=formation_days,  # 페어 선정 기간과 동일
        enter_threshold=enter_threshold,
        exit_threshold=0.5,
        stop_loss=3.0,
        min_half_life=5,
        max_half_life=60,
        min_cost_ratio=5.0,
        transaction_cost=0.0001
    )
    
    enter_list, watch_list = trader.screen_pairs(prices, n_pairs=n_pairs)
    return enter_list, watch_list, prices

def create_pair_chart(prices, asset1, asset2, formation_days, signal_days, asset_mapping=None):
    """페어 차트 생성 (SSD 방법론에 맞게 수정)"""
    # 전체 기간 데이터
    end_date = prices.index[-1]
    start_date = end_date - timedelta(days=int(formation_days * 1.4))  # 여유를 두고
    
    chart_data = prices.loc[start_date:end_date, [asset1, asset2]].dropna()
    
    if len(chart_data) == 0:
        st.error(f"데이터가 없습니다: {asset1}, {asset2}")
        return None
    
    # 누적수익률 계산 (SSD 방법론)
    cumulative_returns = (1 + chart_data.pct_change().fillna(0)).cumprod()
    
    # 정규화 (첫날=1)
    normalized_data = cumulative_returns / cumulative_returns.iloc[0]
    
    # 최근 6개월 기준점 계산
    six_months_ago = end_date - timedelta(days=180)
    
    # 스프레드 및 편차 계산 (SSD 방법론)
    recent_data = chart_data.tail(formation_days)
    recent_cumret = (1 + recent_data.pct_change().fillna(0)).cumprod()
    recent_normalized = recent_cumret / recent_cumret.iloc[0]
    
    spread = recent_normalized[asset1] - recent_normalized[asset2]
    
    # 표준편차 기준 편차값 계산 (2σ 트리거)
    spread_mean = spread.mean()
    spread_std = spread.std()
    deviation_sigma = (spread - spread_mean) / spread_std if spread_std > 0 else spread * 0
    
    # 서브플롯 생성
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.4, 0.3, 0.3],
        subplot_titles=[
            f'{asset1} vs {asset2} - 누적수익률 (SSD 방법론)',
            'Spread (Cumulative Return Difference)',
            'Deviation (σ units)'
        ]
    )
    
    # 1. 누적수익률 차트
    fig.add_trace(
        go.Scatter(
            x=normalized_data.index,
            y=normalized_data[asset1],
            name=f'{asset1} (누적수익률)',
            line=dict(color='blue', width=2),
            hovertemplate=f'<b>{asset1}</b><br>Date: %{{x}}<br>Cumulative Return: %{{y:.4f}}<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=normalized_data.index,
            y=normalized_data[asset2],
            name=f'{asset2} (누적수익률)',
            line=dict(color='red', width=2),
            hovertemplate=f'<b>{asset2}</b><br>Date: %{{x}}<br>Cumulative Return: %{{y:.4f}}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2. 스프레드 차트 (누적수익률 차이)
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
    
    # 스프레드 평균 라인
    fig.add_hline(y=spread_mean, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)
    
    # 3. 편차 차트 (σ 단위)
    deviation_dates = deviation_sigma.index
    deviation_values = deviation_sigma.values
    
    fig.add_trace(
        go.Scatter(
            x=deviation_dates,
            y=deviation_values,
            name='Deviation (σ)',
            line=dict(color='purple', width=2),
            hovertemplate='<b>Deviation</b><br>Date: %{x}<br>Value: %{y:.2f}σ<extra></extra>'
        ),
        row=3, col=1
    )
    
    # 편차 임계값 라인들 (2σ 트리거)
    fig.add_hline(y=2.0, line_dash="dash", line_color="orange", opacity=0.7, row=3, col=1)
    fig.add_hline(y=-2.0, line_dash="dash", line_color="orange", opacity=0.7, row=3, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=3, col=1)
    
    # 1.5σ 관찰 라인들
    fig.add_hline(y=1.5, line_dash="dot", line_color="yellow", opacity=0.5, row=3, col=1)
    fig.add_hline(y=-1.5, line_dash="dot", line_color="yellow", opacity=0.5, row=3, col=1)
    
    # 최근 6개월 배경색 강조
    fig.add_vrect(
        x0=six_months_ago, x1=end_date,
        fillcolor="yellow", opacity=0.1,
        layer="below", line_width=0,
        row=1, col=1
    )
    fig.add_vrect(
        x0=six_months_ago, x1=end_date,
        fillcolor="yellow", opacity=0.1,
        layer="below", line_width=0,
        row=2, col=1
    )
    fig.add_vrect(
        x0=six_months_ago, x1=end_date,
        fillcolor="yellow", opacity=0.1,
        layer="below", line_width=0,
        row=3, col=1
    )
    
    # 차트 제목에 자산 이름 포함
    if asset_mapping:
        name1 = asset_mapping.get(asset1, asset1)
        name2 = asset_mapping.get(asset2, asset2)
        chart_title = f"SSD 페어트레이딩 분석: {asset1}({name1}) - {asset2}({name2})"
    else:
        chart_title = f"SSD 페어트레이딩 분석: {asset1} - {asset2}"
    
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
    fig.update_yaxes(title_text="Cumulative Return Index", row=1, col=1)
    fig.update_yaxes(title_text="Spread", row=2, col=1)
    fig.update_yaxes(title_text="Deviation (σ)", row=3, col=1)
    
    # 현재 편차값 주석
    current_deviation = deviation_values[-1] if len(deviation_values) > 0 else 0
    fig.add_annotation(
        x=deviation_dates[-1] if len(deviation_dates) > 0 else end_date,
        y=current_deviation,
        text=f"현재 편차: {current_deviation:.2f}σ",
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
    st.title("SSD 거리 기반 페어트레이딩")
    st.markdown("---")
    
    # SSD 방법론 설명
    st.info("""
    ### SSD (Sum of Squared Deviations) 거리 기반 페어 선정 방법론
    
    **핵심 원리**: Gatev et al. (2006) "Pairs Trading: Performance of a Relative-Value Arbitrage Rule" 논문의 방법론
    
    **상세 작동 과정**:
    1. **형성 기간 (Formation Period)**: 12개월(252일) 데이터로 페어 선정
    2. **누적수익률 계산**: 각 자산의 cumulative total return index 계산 (배당재투자 가정)
    3. **SSD 계산**: 정규화된 두 가격시계열 간 제곱편차 합(Sum of Squared Deviations) 계산
       - SSD = Σ(Normalized_Price1ᵢ - Normalized_Price2ᵢ)²
    4. **최적 페어 매칭**: 각 종목에 대해 SSD가 최소가 되는 상대 종목을 찾아 페어 구성
    5. **트리거 조건**: 형성기간 스프레드 표준편차(σ) 기준으로 2σ 이상 벗어나면 진입
    
    **학술적 근거**: 
    - "실무 트레이더들이 말하는 '둘이 함께 움직인다(move together)'를 수치화한 것이 SSD"
    - 실제 실무 관행을 가장 잘 근사하는 방법으로 논문에서 검증됨
    - Wall Street에서 실제 사용되는 페어트레이딩 전략의 학술적 구현
    
    **유클리드 거리와의 차이점**:
    - **유클리드**: 단순 가격 경로의 기하학적 거리
    - **SSD**: 누적수익률 기반 제곱편차 합 → 더 정교하고 실무적
    
    **장점**: 학술적 검증, 실무 검증, 배당 효과 반영, 정교한 수익률 기반 매칭
    """)
    
    # 사이드바 설정
    st.sidebar.header("분석 설정")
    st.sidebar.markdown("### 기간 설정")
    
    formation_days = st.sidebar.slider(
        "형성 기간 (일)",
        min_value=252,
        max_value=756,   # 3년 최대
        value=252,       # 12개월 (논문 기준)
        step=63,         # 3개월 단위
        help="페어 선정을 위한 과거 데이터 기간 (논문: 12개월)"
    )
    
    # 신호 계산 기간은 형성 기간과 동일
    signal_days = formation_days
    st.sidebar.info(f"**신호 계산 기간**: {signal_days}일 (형성 기간과 동일)")
    
    st.sidebar.markdown("### 트리거 설정")
    
    enter_threshold = st.sidebar.slider(
        "진입 임계값 (σ)",
        min_value=1.5,
        max_value=3.0,
        value=2.0,       # 논문 기준
        step=0.1,
        help="논문 기준: 2σ 이상 벗어나면 진입"
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
    
    # 메인 콘텐츠
    with st.spinner("SSD 페어 분석 중... 잠시만 기다려주세요."):
        try:
            enter_list, watch_list, prices = analyze_pairs(formation_days, signal_days, enter_threshold, n_pairs)
            asset_mapping = load_asset_names()  # 자산 이름 매핑 로딩
        except Exception as e:
            st.error(f"분석 중 오류 발생: {str(e)}")
            return
    
    # 분석 결과 요약
    st.header("분석 결과 요약")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("진입 신호", f"{len(enter_list)}개", help="2σ 이상 벗어난 페어")
    
    with col2:
        st.metric("관찰 대상", f"{len(watch_list)}개", help="1.5σ~2σ 범위의 페어")
    
    with col3:
        st.metric("형성 기간", f"{formation_days}일", help="페어 선정에 사용된 데이터 기간")
        
    with col4:
        avg_deviation = np.mean([abs(s['current_deviation']) for s in enter_list]) if enter_list else 0
        st.metric("평균 편차", f"{avg_deviation:.2f}σ", help="진입 신호들의 평균 편차")
    
    st.markdown("---")
    
    # 진입 신호 테이블
    if enter_list:
        st.header("📈 추천 진입 페어 (2σ 이상)")
        
        # 테이블 데이터 준비
        table_data = []
        for i, signal in enumerate(enter_list, 1):
            formatted_pair = format_pair_name(signal['pair'], asset_mapping)
            table_data.append({
                "순위": i,
                "페어": formatted_pair,
                "방향": signal['direction'],
                "편차": f"{signal['current_deviation']:.2f}σ",
                "Half-Life": f"{signal['half_life']:.1f}일",
                "SSD": f"{signal['ssd_distance']:.3f}"
            })
        
        df_enter = pd.DataFrame(table_data)
        
        # 스타일링된 테이블 표시
        st.dataframe(
            df_enter,
            use_container_width=True,
            hide_index=True,
            column_config={
                "순위": st.column_config.NumberColumn("순위", width="small"),
                "페어": st.column_config.TextColumn("페어", width="medium"),
                "방향": st.column_config.TextColumn("진입 방향", width="large"),
                "편차": st.column_config.TextColumn("편차", width="small"),
                "Half-Life": st.column_config.TextColumn("Half-Life", width="small"),
                "SSD": st.column_config.TextColumn("SSD거리", width="small")
            }
        )
        
        st.markdown("---")
        
        # 페어 선택 및 차트 표시
        st.header("페어 상세 분석")
        
        # 최고 추천 페어 표시
        top_pair = enter_list[0]
        top_formatted_pair = format_pair_name(top_pair['pair'], asset_mapping)
        st.success(f"🏆 최고 추천 페어: {top_formatted_pair} (편차: {top_pair['current_deviation']:.2f}σ)")
        
        # 페어 선택 옵션
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
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("진입 방향", selected_pair_info['direction'])
            with col2:
                st.metric("현재 편차", f"{selected_pair_info['current_deviation']:.2f}σ")
            with col3:
                st.metric("Half-Life", f"{selected_pair_info['half_life']:.1f}일")
            with col4:
                st.metric("SSD 거리", f"{selected_pair_info['ssd_distance']:.3f}")
        
        if selected_pair:
            asset1, asset2 = selected_pair.split('-')
            
            # 차트 생성 및 표시
            with st.spinner(f"{selected_display_pair} 차트 생성 중..."):
                fig = create_pair_chart(prices, asset1, asset2, formation_days, signal_days, asset_mapping)
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 차트 설명
                    st.info("""
                    **SSD 방법론 차트 설명:**
                    - **상단**: 두 자산의 누적수익률 (배당재투자 포함)
                    - **중간**: 스프레드 (누적수익률 차이)
                    - **하단**: 편차 (σ 단위) - 2σ 이상 시 진입 신호
                    - **노란색 배경**: 최근 6개월 기간
                    - **주황색 실선**: 진입 임계값 (±2σ)
                    - **노란색 점선**: 관찰 임계값 (±1.5σ)
                    
                    **Gatev et al. (2006) 논문 방법론**: 형성기간 표준편차 기준 2σ 트리거
                    """)
    
    else:
        st.warning("현재 2σ 진입 조건을 만족하는 페어가 없습니다.")
        st.info("임계값을 낮추거나 형성 기간을 조정해보세요.")
    
    # 관찰 대상 테이블
    if watch_list:
        st.header("👀 관찰 대상 페어 (1.5σ~2σ)")
        
        table_data = []
        for i, signal in enumerate(watch_list, 1):
            formatted_pair = format_pair_name(signal['pair'], asset_mapping)
            table_data.append({
                "순위": i,
                "페어": formatted_pair,
                "편차": f"{signal['current_deviation']:.2f}σ",
                "Half-Life": f"{signal['half_life']:.1f}일",
                "SSD": f"{signal['ssd_distance']:.3f}"
            })
        
        df_watch = pd.DataFrame(table_data)
        st.dataframe(df_watch, use_container_width=True, hide_index=True)
    
    # 방법론 참조
    st.markdown("---")
    st.markdown("""
    **📚 학술적 근거:**
    
    Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). 
    "Pairs Trading: Performance of a Relative-Value Arbitrage Rule". 
    *The Review of Financial Studies*, 19(3), 797-827.
    
    이 방법론은 Wall Street의 실제 페어트레이딩 전략을 학술적으로 구현한 것으로, 
    실무에서 검증된 수익성을 보여줍니다.
    """)

# Streamlit 페이지로 실행
main()