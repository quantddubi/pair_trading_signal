"""
페어트레이딩 분석 도구 - 공적분(Cointegration) 기반 방법론
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
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression

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
cointegration_module = import_module_from_file(os.path.join(project_root, "methods/2_cointegration_pairs.py"), "cointegration_pairs")

# 필요한 함수들 import
load_data = common_utils.load_data
normalize_prices = common_utils.normalize_prices
calculate_spread = common_utils.calculate_spread
calculate_zscore = common_utils.calculate_zscore
CointegrationPairTrading = cointegration_module.CointegrationPairTrading

# 페이지 설정
st.set_page_config(
    page_title="공적분 방법론",
    page_icon="📈", 
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
def analyze_pairs(formation_window, signal_window, enter_threshold, exit_threshold, stop_loss, min_half_life, max_half_life, min_cost_ratio, max_pvalue, n_pairs):
    """페어 분석 실행"""
    prices = load_price_data()
    
    trader = CointegrationPairTrading(
        formation_window=formation_window,
        signal_window=signal_window,
        enter_threshold=enter_threshold,
        exit_threshold=exit_threshold,
        stop_loss=stop_loss,
        min_half_life=min_half_life,
        max_half_life=max_half_life,
        min_cost_ratio=min_cost_ratio,
        max_pvalue=max_pvalue
    )
    
    enter_list, watch_list = trader.screen_pairs(prices, n_pairs=n_pairs)
    return enter_list, watch_list, prices

def create_cointegration_chart(prices, asset1, asset2, formation_window, signal_window, asset_mapping=None):
    """공적분 분석 차트 생성"""
    # 전체 기간 데이터
    end_date = prices.index[-1]
    start_date = end_date - timedelta(days=int(formation_window * 1.4))  # 여유를 두고
    
    chart_data = prices.loc[start_date:end_date, [asset1, asset2]].dropna()
    
    if len(chart_data) == 0:
        st.error(f"데이터가 없습니다: {asset1}, {asset2}")
        return None
    
    # 공적분 관계 분석을 위한 데이터 준비
    recent_data = chart_data.tail(formation_window)
    
    # 로그 변환
    log_data = np.log(recent_data)
    
    # 공적분 분석 수행
    try:
        # ADF 테스트 결과
        adf_asset1 = adfuller(log_data[asset1])
        adf_asset2 = adfuller(log_data[asset2])
        
        # 선형 회귀로 헤지 비율 계산
        reg = LinearRegression().fit(log_data[[asset1]], log_data[asset2])
        hedge_ratio = reg.coef_[0]
        intercept = reg.intercept_
        
        # 잔차 (스프레드) 계산
        spread = log_data[asset2] - hedge_ratio * log_data[asset1] - intercept
        
        # 스프레드의 ADF 테스트
        adf_spread = adfuller(spread)
        
        # Z-스코어 계산
        zscore_window = max(20, min(60, len(spread)//4))
        zscore = calculate_zscore(spread, window=zscore_window)
        
    except Exception as e:
        st.error(f"공적분 분석 중 오류 발생: {str(e)}")
        return None
    
    # 서브플롯 생성
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.25, 0.25, 0.25, 0.25],
        subplot_titles=[
            f'{asset1} vs {asset2} - 로그 가격',
            f'Linear Relationship (Hedge Ratio: {hedge_ratio:.4f})',
            'Cointegration Spread (Residuals)',
            'Z-Score'
        ]
    )
    
    # 1. 로그 가격 차트
    fig.add_trace(
        go.Scatter(
            x=log_data.index,
            y=log_data[asset1],
            name=f'Log {asset1}',
            line=dict(color='blue', width=2),
            hovertemplate=f'<b>Log {asset1}</b><br>Date: %{{x}}<br>Price: %{{y:.4f}}<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=log_data.index,
            y=log_data[asset2],
            name=f'Log {asset2}',
            line=dict(color='red', width=2),
            hovertemplate=f'<b>Log {asset2}</b><br>Date: %{{x}}<br>Price: %{{y:.4f}}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2. 선형 관계 산점도
    fig.add_trace(
        go.Scatter(
            x=log_data[asset1],
            y=log_data[asset2],
            mode='markers',
            name='Price Relationship',
            marker=dict(color='green', size=4, opacity=0.6),
            hovertemplate=f'<b>{asset1}</b>: %{{x:.4f}}<br><b>{asset2}</b>: %{{y:.4f}}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 회귀선 추가
    x_range = np.linspace(log_data[asset1].min(), log_data[asset1].max(), 100)
    y_fitted = hedge_ratio * x_range + intercept
    
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_fitted,
            mode='lines',
            name='Regression Line',
            line=dict(color='red', width=2, dash='dash'),
            hovertemplate=f'<b>Fitted Line</b><br>{asset1}: %{{x:.4f}}<br>{asset2}: %{{y:.4f}}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 3. 스프레드 차트
    fig.add_trace(
        go.Scatter(
            x=spread.index,
            y=spread.values,
            name='Cointegration Spread',
            line=dict(color='green', width=2),
            hovertemplate='<b>Spread</b><br>Date: %{x}<br>Value: %{y:.4f}<extra></extra>'
        ),
        row=3, col=1
    )
    
    # 스프레드 평균선
    fig.add_hline(y=spread.mean(), line_dash="dash", line_color="gray", opacity=0.5, row=3, col=1)
    
    # 4. Z-스코어 차트
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
        row=4, col=1
    )
    
    # Z-스코어 임계값 라인들
    fig.add_hline(y=2.0, line_dash="dash", line_color="orange", opacity=0.7, row=4, col=1)
    fig.add_hline(y=-2.0, line_dash="dash", line_color="orange", opacity=0.7, row=4, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=4, col=1)
    
    # 최근 6개월 배경색 강조
    six_months_ago = end_date - timedelta(days=180)
    for row in [1, 3, 4]:  # 2번째 행은 산점도라서 제외
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
        chart_title = f"공적분 분석: {asset1}({name1}) - {asset2}({name2})"
    else:
        chart_title = f"공적분 분석: {asset1} - {asset2}"
    
    # 레이아웃 설정
    fig.update_layout(
        height=1000,
        title=chart_title,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )
    
    # 축 레이블 설정
    fig.update_xaxes(title_text="Date", row=4, col=1)
    fig.update_xaxes(title_text=f"Log {asset1}", row=2, col=1)
    fig.update_yaxes(title_text="Log Price", row=1, col=1)
    fig.update_yaxes(title_text=f"Log {asset2}", row=2, col=1)
    fig.update_yaxes(title_text="Spread", row=3, col=1)
    fig.update_yaxes(title_text="Z-Score", row=4, col=1)
    
    # 통계 정보를 주석으로 추가
    current_zscore = zscore_values[-1] if len(zscore_values) > 0 else 0
    fig.add_annotation(
        x=zscore_dates[-1] if len(zscore_dates) > 0 else end_date,
        y=current_zscore,
        text=f"현재 Z-Score: {current_zscore:.2f}<br>ADF p-value: {adf_spread[1]:.4f}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="purple",
        font=dict(size=12, color="purple"),
        row=4, col=1
    )
    
    return fig

def check_parameters_default(params):
    """파라미터가 기본값인지 확인"""
    default_params = cache_utils.get_default_parameters('cointegration')
    return params == default_params

# 메인 앱
def main():
    st.title("공적분 기반 페어트레이딩")
    st.markdown("---")
    
    # 공적분 방법론 설명
    st.info("""
    ### 공적분(Cointegration) 기반 페어 선정 방법론
    
    **핵심 원리**: 두 자산이 장기적으로 안정적인 균형관계를 유지하는 공적분 관계를 찾아 단기 괴리를 이용한 평균회귀 전략
    
    **상세 작동 과정**:
    1. **단위근 검정**: 개별 자산 가격이 비정상적(non-stationary)인지 ADF 테스트로 확인
       - H0: 단위근 존재 (비정상적) vs H1: 정상적
       - p-value > 0.05 → 비정상적 시계열 (가격 데이터의 일반적 특성)
    2. **공적분 관계 검정**: Engle-Granger 2단계 접근법 적용
       - **1단계**: 선형회귀로 헤지비율 추정: Asset2 = α + β×Asset1 + ε
       - **2단계**: 잔차(스프레드)의 정상성 검정: ADF test on residuals
    3. **공적분 벡터 검증**: 잔차가 정상적(p-value < 0.05)이면 공적분 관계 성립
    4. **품질 필터링**:
       - **P-Value**: 0.05 미만 (통계적 유의성 확보)
       - **Half-Life**: 5~60일 범위 (적절한 평균회귀 속도)
       - **헤지비율 안정성**: 시간에 따른 베타 계수의 일관성 검증
    5. **신호 생성**: 공적분 스프레드의 Z-Score 기반 진입/청산 신호
    
    **핵심**: **장기 균형관계**가 통계적으로 검증된 페어들이 **단기적으로 벌어질 때** 수렴을 노리는 전략
    
    **장점**: 통계적 근거 확실, 장기 안정성 우수, False Signal 최소화
    **단점**: 구조적 변화에 취약, 계산 복잡성, 느린 신호 생성
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
        help="공적분 관계 분석을 위한 과거 데이터 기간"
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
    
    max_pvalue = st.sidebar.slider(
        "최대 p-value", 
        min_value=0.01, 
        max_value=0.10, 
        value=0.05, 
        step=0.01,
        help="공적분 검정 통계적 유의수준"
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
        'signal_window': signal_window,
        'enter_threshold': enter_threshold,
        'exit_threshold': exit_threshold,
        'stop_loss': stop_loss,
        'min_half_life': min_half_life,
        'max_half_life': max_half_life,
        'min_cost_ratio': min_cost_ratio,
        'max_pvalue': max_pvalue
    }
    
    # 기본값 여부 확인
    is_default = check_parameters_default(params)
    
    # 메인 콘텐츠
    with st.spinner("공적분 페어 분석 중... 잠시만 기다려주세요."):
        try:
            if is_default:
                st.info("🚀 기본 파라미터를 사용 중. 사전 계산된 결과를 즉시 표시.")
                
                # 캐시에서 결과 로드
                cache_data = cache_utils.load_cache('cointegration')
                if cache_data:
                    enter_list = cache_data.get('enter_signals', [])
                    watch_list = cache_data.get('watch_signals', [])
                    prices = load_price_data()
                else:
                    st.error("캐시 데이터를 찾을 수 없습니다.")
                    return
            else:
                st.warning("⚙️ 사용자 정의 파라미터가 설정되었습니다.")
                # 실시간 분석 실행
                enter_list, watch_list, prices = analyze_pairs(
                    formation_window, signal_window, enter_threshold, exit_threshold,
                    stop_loss, min_half_life, max_half_life, min_cost_ratio, max_pvalue, n_pairs
                )
            
            asset_mapping = load_asset_names()  # 자산 이름 매핑 로딩
            
        except Exception as e:
            st.error(f"분석 중 오류 발생: {str(e)}")
            return
    
    # 분석 결과 요약
    st.header("분석 결과 요약")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("진입 신호", f"{len(enter_list)}개", help="Z-스코어 임계값 이상의 페어")
    
    with col2:
        st.metric("관찰 대상", f"{len(watch_list)}개", help="진입 직전 단계의 페어")
    
    with col3:
        st.metric("Formation Window", f"{formation_window}일", help="공적분 관계 분석에 사용된 기간")
        
    with col4:
        avg_zscore = np.mean([abs(s['current_zscore']) for s in enter_list]) if enter_list else 0
        st.metric("평균 Z-스코어", f"{avg_zscore:.2f}", help="진입 신호들의 평균 Z-스코어")
    
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
                "P-Value": f"{signal.get('p_value', 0.0):.4f}",
                "Half-Life": f"{signal.get('half_life', 0.0):.1f}일",
                "헤지비율": f"{signal.get('hedge_ratio', 1.0):.4f}"
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
                "Z-Score": st.column_config.TextColumn("Z-Score", width="small"),
                "P-Value": st.column_config.TextColumn("P-Value", width="small"),
                "Half-Life": st.column_config.TextColumn("Half-Life", width="small"),
                "헤지비율": st.column_config.TextColumn("헤지비율", width="small")
            }
        )
        
        st.markdown("---")
        
        # 페어 선택 및 차트 표시
        st.header("페어 상세 분석")
        
        # 최고 추천 페어 표시
        top_pair = enter_list[0]
        top_formatted_pair = format_pair_name(top_pair['pair'], asset_mapping)
        st.success(f"최고 추천 페어: {top_formatted_pair}")
        
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
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("진입 방향", selected_pair_info['direction'])
            with col2:
                st.metric("현재 Z-Score", f"{selected_pair_info['current_zscore']:.2f}")
            with col3:
                st.metric("P-Value", f"{selected_pair_info.get('p_value', 0.0):.4f}")
            with col4:
                st.metric("헤지비율", f"{selected_pair_info.get('hedge_ratio', 1.0):.4f}")
        
        if selected_pair:
            asset1, asset2 = selected_pair.split('-')
            
            # 차트 생성 및 표시
            with st.spinner(f"{selected_display_pair} 공적분 차트 생성 중..."):
                fig = create_cointegration_chart(prices, asset1, asset2, formation_window, signal_window, asset_mapping)
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 차트 설명
                    st.info("""
                    **공적분 차트 설명:**
                    - **상단**: 두 자산의 로그 가격 추이
                    - **2번째**: 선형 관계 산점도와 회귀선 (헤지비율 확인)
                    - **3번째**: 공적분 스프레드 (잔차) - 정상적이어야 함
                    - **하단**: Z-스코어 (평균회귀 진입 신호)
                    - **노란색 배경**: 최근 6개월 기간
                    - **주황색 선**: 진입 임계값 (±2.0)
                    - **ADF p-value < 0.05**: 공적분 관계 성립
                    """)
    
    else:
        st.warning("현재 진입 조건을 만족하는 공적분 페어가 없습니다.")
        st.info("P-Value 임계값을 높이거나 Z-Score 임계값을 낮춰보세요.")
    
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
                "P-Value": f"{signal.get('p_value', 0.0):.4f}",
                "Half-Life": f"{signal.get('half_life', 0.0):.1f}일",
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