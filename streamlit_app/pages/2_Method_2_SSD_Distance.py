"""
Pair Trading Analysis Tool - SSD Distance Methodology
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
ssd_module = import_module_from_file(os.path.join(project_root, "methods/2_ssd_distance_pairs.py"), "ssd_distance_pairs")
cache_utils = import_module_from_file(os.path.join(project_root, "utils/cache_utils.py"), "cache_utils")

# 필요한 함수들 import
load_data = common_utils.load_data
normalize_prices = common_utils.normalize_prices
calculate_spread = common_utils.calculate_spread
calculate_zscore = common_utils.calculate_zscore
SSDDistancePairTrading = ssd_module.SSDDistancePairTrading

# 페이지 설정
st.set_page_config(
    page_title="SSD Distance Methodology",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 데이터 로딩 (캐시 없음 - 항상 최신 데이터 로드)
def load_price_data():
    """가격 데이터 로딩"""
    file_path = get_data_file_path()
    return load_data(file_path)

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
def analyze_pairs(formation_days, signal_days, enter_threshold, n_pairs):
    """페어 분석 실행 (캐시 우선 사용)"""
    
    # 기본 파라미터와 일치하는지 확인
    user_params = {
        'formation_window': formation_days,
        'signal_window': formation_days,
        'enter_threshold': enter_threshold,
        'exit_threshold': 0.5,
        'stop_loss': 3.0,
        'min_half_life': 5,
        'max_half_life': 60,
        'min_cost_ratio': 5.0,
        'transaction_cost': 0.0001
    }
    
    # 기본 파라미터와 일치하면 캐시 사용 (상위 n_pairs개만 반환)
    if cache_utils.parameters_match_default('ssd', user_params):
        cache_data = cache_utils.load_cache('ssd')
        if cache_data:
            # 캐시된 데이터에서 요청된 페어 수만큼 반환
            cached_enter = cache_data['enter_signals'][:n_pairs] if len(cache_data['enter_signals']) >= n_pairs else cache_data['enter_signals']
            cached_watch = cache_data['watch_signals'][:n_pairs] if len(cache_data['watch_signals']) >= n_pairs else cache_data['watch_signals']
            
            st.info("📋 캐시된 결과를 사용합니다 (통합 스크리너와 동일)")
            prices = load_price_data()
            return cached_enter, cached_watch, prices
    
    # 캐시를 사용할 수 없으면 실시간 계산
    st.info("🔄 사용자 설정으로 실시간 계산합니다")
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
        chart_title = f"SSD 페어트레이딩 분석: {name1}({asset1}) - {name2}({asset2})"
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
    st.title("SSD Distance Pair Trading")
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
        if st.sidebar.button("Run Analysis", type="primary"):
            pass  # 캐시 사용 안 함
        
        # 파라미터 딕셔너리
        params = {
            'formation_window': formation_days,
            'signal_window': signal_days,
            'enter_threshold': enter_threshold,
            'exit_threshold': 0.5,
            'stop_loss': 3.0,
            'min_half_life': 5,
            'max_half_life': 60,
            'min_cost_ratio': 5.0,
        }
        
        # 기본값 여부 확인
        def check_parameters_default(params):
            """파라미터가 기본값인지 확인"""
            default_params = cache_utils.get_default_parameters('ssd')
            for key, value in default_params.items():
                if params.get(key) != value:
                    return False
            return True
        
        is_default = check_parameters_default(params)
        
        # 메인 콘텐츠
        with st.spinner("SSD 거리 기반 페어 분석 중... 잠시만 기다려주세요."):
            try:
                if is_default:
                    st.success("🚀 기본 파라미터를 사용 중. 사전 계산된 결과를 즉시 표시")
                    # 캐시에서 로딩
                    cache_data = cache_utils.load_cache('ssd')
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
                    enter_list, watch_list, prices = analyze_pairs(formation_days, signal_days, enter_threshold, n_pairs)
                
                asset_mapping = load_asset_names()  # 자산 이름 매핑 로딩
                
            except Exception as e:
                st.error(f"분석 중 오류 발생: {str(e)}")
                return
        
        # 분석 결과 요약
        st.header("📈 분석 결과 요약")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Entry Signals", f"{len(enter_list)}개", help="2σ 이상 벗어난 페어")
        
        with col2:
            st.metric("Watch List", f"{len(watch_list)}개", help="1.5σ~2σ 범위의 페어")
        
        with col3:
            avg_ssd = np.mean([s.get('ssd_distance', 0) for s in enter_list]) if enter_list else 0
            st.metric("평균 SSD 거리", f"{avg_ssd:.3f}", help="진입 신호들의 평균 SSD 거리")
            
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
                    "편차": f"{signal.get('current_deviation', 0):.2f}σ",
                    "SSD거리": f"{signal.get('ssd_distance', 0):.3f}",
                    "반감기": f"{signal.get('half_life', 50):.1f}일",
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
                    "편차": st.column_config.TextColumn("편차(σ)", width="small"),
                    "SSD거리": st.column_config.TextColumn("SSD거리", width="small"),
                    "반감기": st.column_config.TextColumn("반감기", width="small"),
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
                    "편차": f"{signal.get('current_deviation', 0):.2f}σ",
                    "SSD거리": f"{signal.get('ssd_distance', 0):.3f}",
                    "반감기": f"{signal.get('half_life', 50):.1f}일",
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
                    "편차": st.column_config.TextColumn("편차(σ)", width="small"),
                    "SSD거리": st.column_config.TextColumn("SSD거리", width="small"),
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
                    current_deviation = signal_data.get('current_deviation', 0)
                    st.metric("현재 편차", f"{current_deviation:.2f}σ")
                
                with col3:
                    ssd_distance = signal_data.get('ssd_distance', 0)
                    st.metric("SSD 거리", f"{ssd_distance:.3f}")
                
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
                        quality_score = signal_data.get('quality_score', 0.0)
                        st.info(f"**품질 점수**: {quality_score:.1f}")
                    
                    with col3:
                        st.info(f"**논문 기준**: 2σ 이상 벗어나면 진입")
                
                # 차트 생성 및 표시
                st.markdown("#### 📈 페어 차트 분석")
                
                try:
                    chart = create_pair_chart(
                        prices, asset1, asset2, 
                        formation_days, signal_days, 
                        asset_mapping
                    )
                    
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                        
                        # 차트 해석 도움말
                        with st.expander("📖 차트 해석 가이드"):
                            st.markdown("""
                            **📊 SSD 방법론 차트 구성**:
                            - **상단**: 누적수익률 비교 (배당재투자 포함, SSD 방법론 기준)
                            - **중단**: 스프레드 (누적수익률 차이)
                            - **하단**: 편차 (σ 단위) - 2σ 이상 시 진입 신호
                            
                            **🎯 거래 신호 해석**:
                            - **편차 > +2.0σ**: Asset1 매도, Asset2 매수 신호
                            - **편차 < -2.0σ**: Asset1 매수, Asset2 매도 신호
                            - **편차 → 0**: 포지션 청산 신호
                            
                            **📅 기간 구분**:
                            - **노란색 배경**: 최근 6개월 (거래 집중 분석 구간)
                            - **주황색 실선**: 진입 임계값 (±2σ)
                            - **노란색 점선**: 관찰 임계값 (±1.5σ)
                            
                            **📚 학술적 근거**:
                            - Gatev et al. (2006) 논문 방법론
                            - 형성기간 표준편차 기준 2σ 트리거
                            - Wall Street 실무 관행을 학술적으로 구현
                            """)
                    else:
                        st.error("차트 생성 중 오류가 발생했습니다.")
                        
                except Exception as e:
                    st.error(f"차트 생성 오류: {str(e)}")
        else:
            st.warning("분석할 페어가 없습니다. 파라미터를 조정하여 다시 실행해보세요.")

    with tab2:
        st.markdown("### 📊 상세 작동 과정")
        
        # Step 1: 형성 기간 데이터 준비
        with st.container():
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### STEP 1")
                st.info("**📅 형성 기간 설정**")
            
            with col2:
                st.markdown("")
                st.markdown("")
                st.markdown("""
                #### 12개월(252일) 형성 기간으로 페어 선정
                - ✅ Gatev et al. (2006) 논문의 표준 방법
                - ✅ 충분한 데이터로 안정적 관계 파악
                - ✅ 계절성 및 사이클 효과 포함
                - ✅ 시장 상황 변화 적절히 반영
                """)
        
        st.markdown("---")
        
        # Step 2: 누적수익률 계산
        with st.container():
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### STEP 2")
                st.warning("**📈 누적수익률 계산**")
            
            with col2:
                st.markdown("")
                st.markdown("")
                st.markdown("""
                #### 각 자산의 cumulative total return index 계산 (배당재투자 가정)
                """)
                st.latex(r"CumReturn_t = \prod_{i=1}^{t} (1 + r_i)")
                st.caption("rt: 일일 수익률, 첫날을 1.0으로 정규화하여 상대적 성과 비교")
        
        st.markdown("---")
        
        # Step 3: SSD 계산
        with st.container():
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### STEP 3")
                st.success("**📏 SSD 거리 계산**")
            
            with col2:
                st.markdown("")
                st.markdown("")
                st.markdown("""
                #### 정규화된 두 가격시계열 간 제곱편차의 합
                """)
                st.latex(r"SSD_{AB} = \sum_{t=1}^{T} (P_A^{norm}(t) - P_B^{norm}(t))^2")
                
                # SSD 해석 테이블
                st.markdown("""
                | SSD 값 | 해석 | 페어 적합성 |
                |---------|------|-------------|
                | 0 ~ 0.1 | 거의 동일한 움직임 | ⭐⭐⭐ 최적 |
                | 0.1 ~ 0.3 | 매우 유사한 움직임 | ⭐⭐ 우수 |
                | 0.3 ~ 0.5 | 유사한 움직임 | ⭐ 양호 |
                | > 0.5 | 상이한 움직임 | ❌ 부적합 |
                """)
                st.caption("💡 SSD가 낮을수록 두 자산이 더 유사하게 움직임")
        
        st.markdown("---")
        
        # Step 4: 최적 페어 매칭
        with st.container():
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### STEP 4")
                st.error("**🎯 최적 페어 매칭**")
            
            with col2:
                st.markdown("")
                st.markdown("")
                
                # 두 개의 서브 컬럼으로 매칭 과정 설명
                subcol1, subcol2 = st.columns(2)
                
                with subcol1:
                    st.markdown("""
                    #### 매칭 알고리즘
                    1. **각 자산별로** 모든 다른 자산과의 SSD 계산
                    2. **최소 SSD** 찾아 최적 파트너 결정
                    3. **상호 매칭** 확인 (A→B, B→A 모두 최적)
                    4. **중복 제거** 및 최종 페어 리스트 구성
                    """)
                
                with subcol2:
                    st.markdown("""
                    #### 품질 검증 과정
                    - **Half-Life**: 5~60일 범위 확인
                    - **코인테그레이션**: 장기 균형관계 검증  
                    - **거래비용**: 수익성 대비 비용 분석
                    - **안정성**: 형성기간 내 일관성 확인
                    """)
        
        st.markdown("---")
        
        # Step 5: 트리거 시스템
        with st.container():
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### STEP 5")
                st.info("**⚡ 트리거 시스템**")
            
            with col2:
                st.markdown("")
                st.markdown("")
                st.markdown("""
                #### 형성기간 표준편차 기준 2σ 트리거 시스템
                """)
                st.latex(r"Trigger = |Spread_t| > 2 \times \sigma_{formation}")
                
                # 트리거 레벨 설명
                st.markdown("""
                | 편차 수준 | 의미 | 액션 |
                |----------|------|------|
                | < 1.5σ | 정상 범위 | 📊 모니터링 |
                | 1.5σ ~ 2σ | 관찰 구간 | 👀 **관찰 대상** |
                | > 2σ | 진입 신호 | 🚀 **진입 신호** |
                | > 3σ | 극단적 괴리 | ⚠️ 리스크 주의 |
                """)
        
        st.markdown("---")
        
        # 핵심 요약
        st.success("""
        ### 🎯 SSD 방법론의 핵심 전략
        **"실무 트레이더들이 말하는 '둘이 함께 움직인다'를 수치화한 것이 SSD"**
        
        **✅ 학술적 검증된 장점**
        - Gatev et al. (2006) 논문으로 학술적 근거 확보
        - Wall Street 실제 트레이딩 룸에서 사용되는 방법론
        - 누적수익률 기반으로 더 정교한 유사성 측정
        - 12개월 형성기간으로 안정적 관계 파악
        """)
        
        st.markdown("---")
        
        # SSD 방법론 비교
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 SSD vs 유클리드 거리")
            st.markdown("""
            | 구분 | SSD | 유클리드 |
            |------|-----|----------|
            | 기준 | 누적수익률 | 정규화 가격 |
            | 계산 | 제곱편차 합 | 기하학적 거리 |
            | 특징 | 수익률 중심 | 가격 경로 중심 |
            | 장점 | 실무적, 학술적 | 직관적, 빠름 |
            """)
        
        with col2:
            st.markdown("#### 🎯 실무 활용 가이드")
            st.markdown("""
            **페어 선정 기준**:
            - SSD < 0.3: 우선 고려 대상
            - Half-Life 5-30일: 이상적 범위  
            - 2σ 트리거: 논문 검증된 기준
            
            **리스크 관리**:
            - 3σ 이상: 극단적 상황 주의
            - Stop-loss: 일반적으로 3σ 설정
            - 포지션 크기: 변동성 고려 조정
            """)
    
    with tab3:
        st.info("""
        ### SSD (Sum of Squared Deviations) 거리 기반 페어 선정 방법론
        
        **핵심 원리**: Gatev et al. (2006) "Pairs Trading: Performance of a Relative-Value Arbitrage Rule" 논문에서 제시된 월스트리트 실무 방법론을 학술적으로 구현
        
        **상세 작동 과정**:
        1. **형성 기간 설정**: 12개월(252일) 데이터로 페어 선정 - 논문의 표준 방법론
        2. **누적수익률 계산**: 각 자산의 cumulative total return index 계산 (배당재투자 가정)
        3. **SSD 계산**: 정규화된 두 가격시계열 간 제곱편차 합 계산
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
        
        **핵심**: SSD 값이 낮을수록 두 자산이 더 유사하게 움직이며, 2σ 이상 벗어나면 평균회귀 기회
        
        **장점**: 학술적 검증, 실무 검증, 수익률 기반 매칭, 월스트리트 실전 경험 반영
        """)
    
    with tab4:
        st.markdown("### 수학적 기초 및 계산 과정")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 1️⃣ 누적수익률 계산")
            st.latex(r"CumReturn_t = \prod_{i=1}^{t} (1 + r_i)")
            st.caption("첫날을 1.0으로 정규화하여 상대적 성과 비교")
            
            st.markdown("#### 2️⃣ SSD 거리")
            st.latex(r"SSD_{AB} = \sum_{t=1}^{T} (P_A^{norm}(t) - P_B^{norm}(t))^2")
            st.caption("T=252일 동안의 정규화된 가격 차이의 제곱합")
        
        with col2:
            st.markdown("#### 3️⃣ 스프레드 계산")
            st.latex(r"Spread_t = P_A^{norm}(t) - P_B^{norm}(t)")
            st.caption("정규화된 누적수익률의 차이")
            
            st.markdown("#### 4️⃣ 표준화된 편차")
            st.latex(r"Z_t = \frac{Spread_t - \mu_{spread}}{\sigma_{spread}}")
            st.caption("형성기간 기준 표준화 (μ: 평균, σ: 표준편차)")
        
        st.markdown("---")
        
        # 실제 계산 예시
        st.markdown("#### 💡 실제 계산 예시")
        
        example_col1, example_col2 = st.columns(2)
        
        with example_col1:
            st.code("""
# 1. 누적수익률 계산
returns_A = [0.01, 0.02, -0.005, 0.015]
returns_B = [0.008, 0.018, -0.002, 0.012]

# 누적수익률 (1일차 = 1.0)
cum_A = [1.0]
cum_B = [1.0]

for r_a, r_b in zip(returns_A, returns_B):
    cum_A.append(cum_A[-1] * (1 + r_a))
    cum_B.append(cum_B[-1] * (1 + r_b))

# 2. SSD 계산
differences = [a - b for a, b in zip(cum_A, cum_B)]
ssd = sum(d**2 for d in differences)
            """, language='python')
        
        with example_col2:
            st.markdown("""
            **해석**:
            - SSD 값이 낮을수록 두 자산 유사
            - 형성기간 동안의 전체적 관계 파악
            - 단순 상관관계보다 더 정교한 측정
            
            **트리거 계산**:
            1. 형성기간 스프레드의 평균/표준편차 계산
            2. 현재 스프레드를 표준화
            3. ±2σ 벗어나면 진입 신호 발생
            
            **실무 적용**:
            - 12개월 형성기간으로 SSD 계산
            - 매일 새로운 편차 모니터링
            - 2σ 트리거로 진입/청산 결정
            """)

# Streamlit 페이지로 실행
main()