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
cointegration_module = import_module_from_file(os.path.join(project_root, "methods/3_cointegration_pairs.py"), "cointegration_pairs")

# 필요한 함수들 import
load_data = common_utils.load_data
normalize_prices = common_utils.normalize_prices
calculate_spread = common_utils.calculate_spread
calculate_zscore = common_utils.calculate_zscore
CointegrationPairTrading = cointegration_module.CointegrationPairTrading

# 페이지 설정
st.set_page_config(
    page_title="Cointegration Methodology",
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
    """페어 이름을 이름(티커) 형태로 포맷팅"""
    asset1, asset2 = pair.split('-')
    
    name1 = asset_mapping.get(asset1, asset1)
    name2 = asset_mapping.get(asset2, asset2)
    
    return f"{name1}({asset1}) - {name2}({asset2})"

# 페어 분석 함수
@st.cache_data
def analyze_pairs(formation_window, signal_window, enter_threshold, exit_threshold, stop_loss, min_half_life, max_half_life, min_cost_ratio, max_pvalue, n_pairs):
    """페어 분석 실행 (캐시 우선 사용)"""
    
    # 기본 파라미터와 일치하는지 확인
    user_params = {
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
    
    # 기본 파라미터와 일치하면 캐시 사용
    if cache_utils.parameters_match_default('cointegration', user_params):
        cache_data = cache_utils.load_cache('cointegration')
        if cache_data:
            # 캐시된 데이터에서 요청된 페어 수만큼 반환
            cached_enter = cache_data['enter_signals'][:n_pairs] if len(cache_data['enter_signals']) >= n_pairs else cache_data['enter_signals']
            cached_watch = cache_data['watch_signals'][:n_pairs] if len(cache_data['watch_signals']) >= n_pairs else cache_data['watch_signals']
            
            if len(cached_enter) < n_pairs:
                st.warning(f"📋 캐시에 {len(cached_enter)}개 진입신호만 있어서 실시간 계산으로 전환합니다")
            else:
                st.info("📋 캐시된 결과를 사용합니다 (통합 스크리너와 동일)")
                prices = load_price_data()
                return cached_enter, cached_watch, prices
    
    # 캐시를 사용할 수 없으면 실시간 계산
    st.info("🔄 사용자 설정으로 실시간 계산합니다")
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

def create_pair_chart(prices, asset1, asset2, formation_window, signal_window, asset_mapping=None):
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
        chart_title = f"공적분 분석: {name1}({asset1}) - {name2}({asset2})"
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

# def check_parameters_default 삭제 (중복 제거)

# 메인 앱
def main():
    st.title("Cointegration Pair Trading")
    st.markdown("---")
    
    # 4개 탭 구성 (아이콘 + 명칭 통일)
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 분석 결과 요약",    # 실제 분석 + 인터랙션
        "📊 상세 작동 과정",     # 방법론 단계별 시각화
        "📝 상세 설명",         # 텍스트 중심 설명
        "🔍 수식 및 계산"       # 수학적 기초
    ])
    
    with tab1:
        # 사이드바 설정
        st.sidebar.header("Analysis Settings")
        st.sidebar.markdown("### 기간 설정")
        
        formation_window = st.sidebar.slider(
            "형성 기간 (일)",
            min_value=100,
            max_value=500,
            value=252,
            step=50,
            help="공적분 관계 분석을 위한 과거 데이터 기간"
        )
        
        signal_window = st.sidebar.slider(
            "신호 윈도우 (일)", 
            min_value=20, 
            max_value=120, 
            value=60,
            help="Z-score 계산을 위한 롤링 윈도우"
        )
        
        st.sidebar.markdown("### 신호 설정")
        
        enter_threshold = st.sidebar.slider(
            "진입 Z-스코어 임계값", 
            min_value=1.0, 
            max_value=3.0, 
            value=2.0, 
            step=0.1,
            help="이 값 이상일 때 진입 신호 생성"
        )
        
        exit_threshold = st.sidebar.slider(
            "청산 Z-스코어 임계값", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.5, 
            step=0.1,
            help="이 값 이하일 때 청산 신호 생성"
        )
        
        st.sidebar.markdown("### 품질 필터")
        
        max_pvalue = st.sidebar.slider(
            "최대 P-Value", 
            min_value=0.01, 
            max_value=0.10, 
            value=0.05, 
            step=0.01,
            help="공적분 검정 통계적 유의수준 (낮을수록 엄격)"
        )
        
        max_half_life = st.sidebar.slider(
            "최대 반감기 (일)", 
            min_value=30, 
            max_value=120, 
            value=60,
            help="평균회귀 최대 속도 기준"
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
        if st.sidebar.button("Run Analysis", type="primary"):
            st.cache_data.clear()  # 캐시 클리어
        
        # 파라미터 딕셔너리
        params = {
            'formation_window': formation_window,
            'signal_window': signal_window,
            'enter_threshold': enter_threshold,
            'exit_threshold': exit_threshold,
            'stop_loss': 3.0,  # 고정값
            'min_half_life': 5,  # 고정값
            'max_half_life': max_half_life,
            'min_cost_ratio': 5.0,  # 고정값
            'max_pvalue': max_pvalue
        }
        
        # 기본값 여부 확인
        def check_parameters_default(params):
            """파라미터가 기본값인지 확인"""
            default_params = cache_utils.get_default_parameters('cointegration')
            for key, value in default_params.items():
                if params.get(key) != value:
                    return False
            return True
        
        is_default = check_parameters_default(params)
        
        # 메인 콘텐츠
        with st.spinner("공적분 기반 페어 분석 중... 잠시만 기다려주세요."):
            try:
                if is_default:
                    st.success("🚀 기본 파라미터를 사용 중. 사전 계산된 결과를 즉시 표시")
                    # 캐시에서 로딩
                    cache_data = cache_utils.load_cache('cointegration')
                    if cache_data:
                        enter_list = cache_data.get('enter_signals', [])
                        watch_list = cache_data.get('watch_signals', [])
                        prices = load_price_data()
                    else:
                        st.error("캐시 데이터를 찾을 수 없음")
                        return
                else:
                    st.warning("⚙️ 사용자 정의 파라미터가 설정")
                    # 실시간 분석 실행
                    enter_list, watch_list, prices = analyze_pairs(
                        formation_window, signal_window, enter_threshold, exit_threshold,
                        3.0, 5, max_half_life, 5.0, max_pvalue, n_pairs
                    )
                
                asset_mapping = load_asset_names()  # 자산 이름 매핑 로딩
                
            except Exception as e:
                st.error(f"분석 중 오류 발생: {str(e)}")
                return
        
        # 분석 결과 요약
        st.header("📈 분석 결과 요약")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Entry Signals", f"{len(enter_list)}개", help="Z-스코어 임계값 이상의 공적분 페어")
        
        with col2:
            st.metric("Watch List", f"{len(watch_list)}개", help="진입 직전 단계의 공적분 페어")
        
        with col3:
            avg_pvalue = np.mean([s.get('p_value', 0.05) for s in enter_list]) if enter_list else 0.05
            st.metric("평균 P-Value", f"{avg_pvalue:.4f}", help="진입 신호들의 평균 공적분 p-값")
            
        with col4:
            avg_half_life = np.mean([s.get('half_life', 50) for s in enter_list]) if enter_list else 0
            st.metric("평균 반감기", f"{avg_half_life:.1f}일", help="진입 신호들의 평균 반감기")
    
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
                    "반감기": f"{signal.get('half_life', 0.0):.1f}일",
                    "헤지비율": f"{signal.get('hedge_ratio', 1.0):.4f}",
                    "품질점수": f"{signal.get('quality_score', 0.0):.1f}"
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
                    "반감기": st.column_config.TextColumn("반감기", width="small"),
                    "헤지비율": st.column_config.TextColumn("헤지비율", width="small"),
                    "품질점수": st.column_config.TextColumn("품질점수", width="small")
                }
            )
        else:
            st.warning("현재 진입 신호가 있는 페어가 없습니다.")
        
        # 관찰 대상 테이블
        if watch_list:
            st.markdown("---")
            st.header("관찰 대상 페어")
            
            # 테이블 데이터 준비
            watch_table_data = []
            for i, signal in enumerate(watch_list, 1):
                formatted_pair = format_pair_name(signal['pair'], asset_mapping)
                watch_table_data.append({
                    "순위": i,
                    "페어": formatted_pair,
                    "Z-Score": f"{signal['current_zscore']:.2f}",
                    "P-Value": f"{signal.get('p_value', 0.0):.4f}",
                    "반감기": f"{signal.get('half_life', 0.0):.1f}일",
                    "상태": "진입 대기"
                })
            
            df_watch = pd.DataFrame(watch_table_data)
            
            st.dataframe(
                df_watch,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "순위": st.column_config.NumberColumn("순위", width="small"),
                    "페어": st.column_config.TextColumn("페어", width="medium"),
                    "Z-Score": st.column_config.TextColumn("Z-Score", width="small"),
                    "P-Value": st.column_config.TextColumn("P-Value", width="small"),
                    "반감기": st.column_config.TextColumn("반감기", width="small"),
                    "상태": st.column_config.TextColumn("상태", width="small")
                }
            )
        
        # 페어 상세 분석 섹션
        st.markdown("---")
        st.header("🔍 페어 상세 분석")
        
        # 진입 신호와 관찰 대상을 합쳐서 선택 리스트 생성
        combined_pairs = []
        
        # 진입 신호 페어 추가
        for signal in enter_list:
            formatted_pair = format_pair_name(signal['pair'], asset_mapping)
            combined_pairs.append({
                'display': f"[진입 신호] {formatted_pair}",
                'pair': signal['pair'],
                'type': 'Entry Signals',
                'signal_data': signal
            })
        
        # 관찰 대상 페어 추가
        for signal in watch_list:
            formatted_pair = format_pair_name(signal['pair'], asset_mapping)
            combined_pairs.append({
                'display': f"[관찰 대상] {formatted_pair}",
                'pair': signal['pair'],
                'type': 'Watch List',
                'signal_data': signal
            })
        
        if combined_pairs:
            # 페어 선택
            selected_pair_display = st.selectbox(
                "분석할 페어 선택:",
                options=[pair['display'] for pair in combined_pairs],
                help="진입 신호 페어와 관찰 대상 페어 중에서 선택하여 상세 분석"
            )
            
            # 선택된 페어 정보 찾기
            selected_pair_info = None
            for pair_info in combined_pairs:
                if pair_info['display'] == selected_pair_display:
                    selected_pair_info = pair_info
                    break
            
            if selected_pair_info:
                # 페어 정보 표시
                pair_name = selected_pair_info['pair']
                asset1, asset2 = pair_name.split('-')
                signal_data = selected_pair_info['signal_data']
                
                # 상세 정보 표시
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("페어 타입", selected_pair_info['type'])
                
                with col2:
                    current_zscore = signal_data.get('current_zscore', 0)
                    st.metric("현재 Z-Score", f"{current_zscore:.2f}")
                
                with col3:
                    pvalue = signal_data.get('p_value', 0.0)
                    st.metric("P-Value", f"{pvalue:.4f}")
                
                with col4:
                    half_life = signal_data.get('half_life', 50)
                    st.metric("반감기", f"{half_life:.1f}일")
                
                # 진입 신호인 경우 추가 정보 표시
                if selected_pair_info['type'] == 'Entry Signals':
                    st.markdown("#### 📊 진입 신호 상세 정보")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        direction = signal_data.get('direction', 'N/A')
                        st.info(f"**진입 방향**: {direction}")
                    
                    with col2:
                        hedge_ratio = signal_data.get('hedge_ratio', 1.0)
                        st.info(f"**헤지 비율**: {hedge_ratio:.4f}")
                    
                    with col3:
                        quality_score = signal_data.get('quality_score', 0.0)
                        st.info(f"**품질 점수**: {quality_score:.1f}")
                
                # 차트 생성 및 표시
                st.markdown("#### 📈 페어 차트 분석")
                
                try:
                    chart = create_pair_chart(
                        prices, asset1, asset2, 
                        formation_window, signal_window, 
                        asset_mapping
                    )
                    
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                        
                        # 차트 해석 도움말
                        with st.expander("📖 차트 해석 가이드"):
                            st.markdown("""
                            **📊 공적분 분석 차트 구성**:
                            - **상단**: 로그 가격 추이 (두 자산의 장기 움직임)
                            - **2번째**: 선형 관계 산점도 + 회귀선 (헤지비율 확인)
                            - **3번째**: 공적분 스프레드 (잔차) - 정상적이어야 함
                            - **하단**: Z-Score (평균회귀 진입 신호)
                            
                            **🎯 거래 신호 해석**:
                            - **Z-Score > +2.0**: Asset1 매도, Asset2 매수 신호
                            - **Z-Score < -2.0**: Asset1 매수, Asset2 매도 신호
                            - **Z-Score → 0**: 포지션 청산 신호
                            
                            **📅 기간 구분**:
                            - **노란색 배경**: 최근 6개월 (거래 집중 분석 구간)
                            - **주황색 실선**: 진입 임계값 (±2.0)
                            
                            **🔬 통계적 검증**:
                            - **P-Value < 0.05**: 공적분 관계 성립 (통계적 유의)
                            - **헤지비율**: 두 자산 간의 최적 비율
                            - **반감기**: 스프레드가 평균으로 수렴하는 속도
                            """)
                    else:
                        st.error("차트 생성 중 오류가 발생했습니다.")
                        
                except Exception as e:
                    st.error(f"차트 생성 오류: {str(e)}")
        else:
            st.warning("분석할 페어가 없습니다. 파라미터를 조정하여 다시 실행해보세요.")

    with tab2:
        st.markdown("### 📊 상세 작동 과정")
        
        # Step 1: 단위근 검정
        with st.container():
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### STEP 1")
                st.info("**🔍 단위근 검정**")
            
            with col2:
                st.markdown("")
                st.markdown("")
                st.markdown("""
                #### 개별 자산 가격이 비정상적(non-stationary)인지 ADF 테스트로 확인
                - ✅ H0: 단위근 존재 (비정상적) vs H1: 정상적
                - ✅ p-value > 0.05 → 비정상적 시계열 (가격 데이터의 일반적 특성)
                - ✅ 대부분의 금융 자산 가격은 비정상적 시계열
                - ✅ 개별 자산은 비정상적이지만 선형결합은 정상적일 수 있음
                """)
        
        st.markdown("---")
        
        # Step 2: 공적분 관계 검정
        with st.container():
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### STEP 2")
                st.warning("**📈 공적분 관계 검정**")
            
            with col2:
                st.markdown("")
                st.markdown("")
                st.markdown("""
                #### Engle-Granger 2단계 접근법 적용
                """)
                
                # 2단계 상세 설명
                subcol1, subcol2 = st.columns(2)
                
                with subcol1:
                    st.markdown("""
                    **1단계: 선형회귀**
                    - Asset2 = α + β×Asset1 + ε
                    - OLS로 헤지비율(β) 추정
                    - 절편(α)과 잔차(ε) 계산
                    """)
                
                with subcol2:
                    st.markdown("""
                    **2단계: 잔차 검정**
                    - 잔차(스프레드)의 정상성 검정
                    - ADF test on residuals
                    - p-value < 0.05면 공적분 성립
                    """)
        
        st.markdown("---")
        
        # Step 3: 공적분 벡터 검증
        with st.container():
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### STEP 3")
                st.success("**🎯 공적분 벡터 검증**")
            
            with col2:
                st.markdown("")
                st.markdown("")
                st.markdown("""
                #### 잔차가 정상적이면 공적분 관계 성립
                """)
                st.latex(r"Spread_t = Asset2_t - \beta \times Asset1_t - \alpha")
                
                # 공적분 검증 테이블
                st.markdown("""
                | P-Value | 해석 | 공적분 관계 |
                |---------|------|-------------|
                | < 0.01 | 매우 강한 증거 | ⭐⭐⭐ 최우수 |
                | 0.01 ~ 0.05 | 통계적 유의 | ⭐⭐ 우수 |
                | 0.05 ~ 0.10 | 약한 증거 | ⭐ 고려 가능 |
                | > 0.10 | 증거 부족 | ❌ 부적합 |
                """)
                st.caption("💡 P-Value가 낮을수록 더 강한 공적분 관계")
        
        st.markdown("---")
        
        # Step 4: 품질 필터링
        with st.container():
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### STEP 4")
                st.error("**🔍 품질 필터링**")
            
            with col2:
                st.markdown("")
                st.markdown("")
                
                # 두 개의 서브 컬럼으로 필터 표시
                subcol1, subcol2 = st.columns(2)
                
                with subcol1:
                    st.markdown("""
                    #### 통계적 검증
                    - **P-Value**: 0.05 미만 (유의수준)
                    - **헤지비율 안정성**: 시간에 따른 베타 일관성
                    - **잔차 정규성**: 스프레드의 정상성 확인
                    - **자기상관**: 잔차의 독립성 검정
                    """)
                
                with subcol2:
                    st.markdown("""
                    #### 실용적 필터
                    - **Half-Life**: 5~60일 범위 (평균회귀 속도)
                    - **거래비용**: 수익성 대비 비용 분석
                    - **유동성**: 거래 가능성 확인
                    - **구조변화**: 장기 관계 안정성
                    """)
        
        st.markdown("---")
        
        # Step 5: 신호 생성
        with st.container():
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### STEP 5")
                st.info("**⚡ 신호 생성**")
            
            with col2:
                st.markdown("")
                st.markdown("")
                st.markdown("""
                #### 공적분 스프레드의 Z-Score 기반 진입/청산 신호
                """)
                st.latex(r"Z\text{-Score}_t = \frac{Spread_t - \mu_{spread}}{\sigma_{spread}}")
                
                # 신호 생성 테이블
                st.markdown("""
                | Z-Score 수준 | 의미 | 액션 |
                |-------------|------|------|
                | < -2.0 | 스프레드 과소 | 🚀 **Long Spread** |
                | -2.0 ~ -0.5 | 수렴 중 | 📊 포지션 유지 |
                | -0.5 ~ +0.5 | 균형 구간 | 💰 청산 고려 |
                | +0.5 ~ +2.0 | 수렴 중 | 📊 포지션 유지 |
                | > +2.0 | 스프레드 과대 | 🚀 **Short Spread** |
                """)
        
        st.markdown("---")
        
        # 핵심 요약
        st.success("""
        ### 🎯 Cointegration Methodology의 핵심 전략
        **장기 균형관계가 통계적으로 검증된 페어들이 단기적으로 벌어질 때 수렴을 노리는 전략**
        
        **✅ 통계적 근거 확실한 장점**
        - ADF 테스트로 공적분 관계 엄격 검증
        - Engle-Granger 방법론의 학술적 기초
        - False Signal 최소화 (p-value < 0.05)
        - 장기 안정성 우수 (구조적 균형관계)
        """)
        
        st.markdown("---")
        
        # Cointegration Methodology 비교
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 공적분 vs 기타 방법론")
            st.markdown("""
            | 구분 | 공적분 | SSD/유클리드 |
            |------|--------|-------------|
            | 기준 | 통계적 검증 | 거리/유사성 |
            | 강점 | 이론적 기초 | 직관적 이해 |
            | 신뢰성 | 매우 높음 | 보통 |
            | 계산복잡성 | 높음 | 낮음 |
            """)
        
        with col2:
            st.markdown("#### 🎯 실무 활용 가이드")
            st.markdown("""
            **페어 선정 기준**:
            - P-Value < 0.05: 필수 조건
            - Half-Life 5-30일: 이상적 범위  
            - 헤지비율 안정성: 시간별 일관성
            
            **리스크 관리**:
            - 구조 변화 모니터링 필수
            - 정기적 공적분 관계 재검증
            - 포지션 크기 보수적 설정
            """)
    
    with tab3:
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
        
        **장점**: 통계적 근거 확실, 장기 안정성 우수, False Signal 최소화, 학술적 기초 탄탄
        **단점**: 구조적 변화에 취약, 계산 복잡성, 느린 신호 생성, 높은 진입 장벽
        """)
    
    with tab4:
        st.markdown("### 수학적 기초 및 계산 과정")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 1️⃣ ADF 단위근 검정")
            st.latex(r"\Delta X_t = \alpha + \beta t + \gamma X_{t-1} + \sum_{i=1}^{p} \delta_i \Delta X_{t-i} + \varepsilon_t")
            st.caption("H0: γ=0 (단위근 존재), H1: γ<0 (정상적)")
            
            st.markdown("#### 2️⃣ 공적분 회귀")
            st.latex(r"Y_t = \alpha + \beta X_t + \varepsilon_t")
            st.caption("OLS로 헤지비율(β) 추정")
        
        with col2:
            st.markdown("#### 3️⃣ 잔차 정상성 검정")
            st.latex(r"\Delta \varepsilon_t = \rho \varepsilon_{t-1} + \sum_{i=1}^{k} \phi_i \Delta \varepsilon_{t-i} + \nu_t")
            st.caption("Engle-Granger: H0: ρ=0 (비정상), H1: ρ<0 (정상)")
            
            st.markdown("#### 4️⃣ Z-Score 신호")
            st.latex(r"Z_t = \frac{\varepsilon_t - \bar{\varepsilon}}{\sigma_\varepsilon}")
            st.caption("표준화된 스프레드로 진입/청산 신호 생성")
        
        st.markdown("---")
        
        # 실제 계산 예시
        st.markdown("#### 💡 실제 계산 예시")
        
        example_col1, example_col2 = st.columns(2)
        
        with example_col1:
            st.code("""
# 1. ADF 테스트
from statsmodels.tsa.stattools import adfuller

# 개별 자산 단위근 검정
adf_asset1 = adfuller(log_prices_asset1)
adf_asset2 = adfuller(log_prices_asset2)
print(f"Asset1 p-value: {adf_asset1[1]:.4f}")
print(f"Asset2 p-value: {adf_asset2[1]:.4f}")

# 2. 공적분 회귀
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(log_asset1.reshape(-1,1), log_asset2)
hedge_ratio = reg.coef_[0]
intercept = reg.intercept_

# 3. 잔차 계산 및 검정
residuals = log_asset2 - hedge_ratio * log_asset1 - intercept
adf_residuals = adfuller(residuals)
print(f"Cointegration p-value: {adf_residuals[1]:.4f}")
            """, language='python')
        
        with example_col2:
            st.markdown("""
            **해석**:
            - 개별 자산 p-value > 0.05: 비정상적 (일반적)
            - 잔차 p-value < 0.05: 공적분 관계 성립
            - 헤지비율: 두 자산 간 최적 균형 비율
            
            **트레이딩 신호**:
            1. Z-Score = (잔차 - 평균) / 표준편차
            2. Z-Score > +2.0: Short Spread (Asset1↓, Asset2↑)
            3. Z-Score < -2.0: Long Spread (Asset1↑, Asset2↓)
            4. Z-Score → 0: 포지션 청산
            
            **실무 적용**:
            - 252일 형성기간으로 공적분 검정
            - 60일 롤링으로 Z-Score 계산
            - P-value < 0.05 필터링으로 품질 확보
            - Half-Life로 평균회귀 속도 검증
            """)

# Streamlit 페이지로 실행
if __name__ == "__main__":
    main()
else:
    main()