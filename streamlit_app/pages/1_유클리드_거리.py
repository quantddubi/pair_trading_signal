"""
페어트레이딩 분석 도구 - 유클리드 거리 기반 방법론
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
euclidean_module = import_module_from_file(os.path.join(project_root, "methods/1_euclidean_distance_pairs.py"), "euclidean_distance_pairs")
cache_utils = import_module_from_file(os.path.join(project_root, "utils/cache_utils.py"), "cache_utils")

# 필요한 함수들 import
load_data = common_utils.load_data
normalize_prices = common_utils.normalize_prices
calculate_spread = common_utils.calculate_spread
calculate_zscore = common_utils.calculate_zscore
EuclideanDistancePairTrading = euclidean_module.EuclideanDistancePairTrading

# 페이지 설정
st.set_page_config(
    page_title="유클리드 거리 방법론",
    page_icon="📐",
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
    
    # 기본 파라미터와 일치하면 캐시 사용
    if cache_utils.parameters_match_default('euclidean', user_params):
        cache_data = cache_utils.load_cache('euclidean')
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
    
    trader = EuclideanDistancePairTrading(
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
    """페어 차트 생성"""
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
    
    # 스프레드 및 Z-스코어 계산 (페어 선정 기간과 일치)
    recent_data = chart_data.tail(formation_days)
    normalized_recent = normalize_prices(recent_data, method='rebase')
    spread = calculate_spread(normalized_recent[asset1], normalized_recent[asset2], hedge_ratio=1.0)
    # Z-score 계산 - 안전한 윈도우 크기 사용
    zscore_window = max(20, min(60, len(spread)//4))  # 최소 20일, 최대 60일
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
            f'{asset1} vs {asset2} - 정규화된 가격',
            'Spread (Price Difference)',
            'Z-Score'
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
        chart_title = f"페어트레이딩 분석: {name1}({asset1}) - {name2}({asset2})"
    else:
        chart_title = f"페어트레이딩 분석: {asset1} - {asset2}"
    
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
    st.title("유클리드 거리 기반 페어트레이딩")
    st.markdown("---")
    
    # 방법론 개요를 탭으로 구성
    tab1, tab2, tab3, tab4 = st.tabs(["📈 분석 결과 요약", "📊 상세 작동 과정", "📝 상세 설명", "🔍 수식 및 계산"])
    
    with tab1:
        # 사이드바 설정을 먼저 가져와서 분석 결과를 표시
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
        
        # Z-스코어 계산 기간은 분석 기간과 동일하게 설정
        signal_days = formation_days
        st.sidebar.info(f"**Z-스코어 계산 기간**: {signal_days}일 (분석 기간과 동일)")
        
        st.sidebar.markdown("### 신호 설정")
        
        enter_threshold = st.sidebar.slider(
            "진입 Z-스코어 임계값",
            min_value=1.5,
            max_value=3.0,
            value=2.0,
            step=0.1,
            help="이 값 이상일 때 진입 신호 생성"
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
            default_params = cache_utils.get_default_parameters('euclidean')
            for key, value in default_params.items():
                if params.get(key) != value:
                    return False
            return True
        
        is_default = check_parameters_default(params)
        
        # 메인 콘텐츠
        with st.spinner("유클리드 거리 기반 페어 분석 중... 잠시만 기다려주세요."):
            try:
                if is_default:
                    st.success("🚀 기본 파라미터를 사용 중. 사전 계산된 결과를 즉시 표시")
                    # 캐시에서 로딩
                    cache_data = cache_utils.load_cache('euclidean')
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
            st.metric("진입 신호", f"{len(enter_list)}개", help="Z-스코어 임계값 이상의 페어")
        
        with col2:
            st.metric("관찰 대상", f"{len(watch_list)}개", help="진입 직전 단계의 페어")
        
        with col3:
            avg_distance = np.mean([s.get('distance', 0) for s in enter_list]) if enter_list else 0
            st.metric("평균 거리", f"{avg_distance:.2f}", help="진입 신호들의 평균 유클리드 거리")
            
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
                    "거리": f"{signal.get('distance', 0):.2f}",
                    "반감기": f"{signal.get('half_life', 50):.1f}일",
                    "품질점수": f"{signal.get('quality_score', 0.0):.1f}",
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
                    "거리": st.column_config.TextColumn("거리", width="small"),
                    "반감기": st.column_config.TextColumn("반감기", width="small"),
                    "품질점수": st.column_config.TextColumn("품질점수", width="small"),
                    "헤지비율": st.column_config.TextColumn("헤지비율", width="small")
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
                    "거리": f"{signal.get('distance', 0):.2f}",
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
                    "Z-Score": st.column_config.TextColumn("Z-Score", width="small"),
                    "거리": st.column_config.TextColumn("거리", width="small"),
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
                'type': '진입 신호',
                'signal_data': signal
            })
        
        # 관찰 대상 페어 추가
        for signal in watch_list:
            formatted_pair = format_pair_name(signal['pair'], asset_mapping)
            combined_pairs.append({
                'display': f"[관찰 대상] {formatted_pair}",
                'pair': signal['pair'],
                'type': '관찰 대상',
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
                    distance = signal_data.get('distance', 0)
                    st.metric("유클리드 거리", f"{distance:.2f}")
                
                with col4:
                    half_life = signal_data.get('half_life', 50)
                    st.metric("반감기", f"{half_life:.1f}일")
                
                # 진입 신호인 경우 추가 정보 표시
                if selected_pair_info['type'] == '진입 신호':
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
                        formation_days, signal_days, 
                        asset_mapping
                    )
                    
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                        
                        # 차트 해석 도움말
                        with st.expander("📖 차트 해석 가이드"):
                            st.markdown("""
                            **📊 차트 구성**:
                            - **상단**: 정규화된 가격 비교 (두 자산의 상대적 움직임)
                            - **중단**: 스프레드 (Asset1 - Asset2의 차이)
                            - **하단**: Z-Score (표준화된 스프레드 신호)
                            
                            **🎯 거래 신호 해석**:
                            - **Z-Score > +2.0**: Asset1 매도, Asset2 매수 신호
                            - **Z-Score < -2.0**: Asset1 매수, Asset2 매도 신호
                            - **Z-Score → 0**: 포지션 청산 신호
                            
                            **📅 기간 구분**:
                            - **노란색 배경**: 최근 6개월 (거래 집중 분석 구간)
                            - **전체 구간**: 과거 패턴 참고용
                            
                            **⚠️ 주의사항**:
                            - Half-Life가 짧을수록 빠른 수렴 예상
                            - 거래비용을 고려한 실제 진입/청산 결정 필요
                            """)
                    else:
                        st.error("차트 생성 중 오류가 발생했습니다.")
                        
                except Exception as e:
                    st.error(f"차트 생성 오류: {str(e)}")
        else:
            st.warning("분석할 페어가 없습니다. 파라미터를 조정하여 다시 실행해보세요.")

    with tab2:
        st.markdown("### 📊 상세 작동 과정")
        
        # Step 1: 가격 정규화
        with st.container():
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### STEP 1")
                st.info("**🔄 가격 정규화**")
            
            with col2:
                st.markdown("")
                st.markdown("")
                st.markdown("""
                #### 최근 3년(756일) 데이터를 첫 거래일 = 1.0으로 리베이싱
                - ✅ 절대 가격차이 제거
                - ✅ 상대적 움직임만 비교
                - ✅ 모든 자산 동일 스케일
                """)
        
        st.markdown("---")
        
        # Step 2: 유클리드 거리 계산
        with st.container():
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### STEP 2")
                st.warning("**📏 유클리드 거리 계산**")
            
            with col2:
                st.markdown("")
                st.markdown("")
                st.markdown("""
                #### 모든 자산 쌍에 대해 정규화된 가격 경로 간 유클리드 거리 측정
                """)
                st.latex(r"d = \sqrt{\sum_{i=1}^{n} (Asset1_i - Asset2_i)^2}")
                st.caption("n = 756일 (3년), 거리가 낮을수록 유사한 움직임")
        
        st.markdown("---")
        
        # Step 3: 거리 기준 스크리닝
        with st.container():
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### STEP 3")
                st.success("**🎯 거리 기준 스크리닝**")
            
            with col2:
                st.markdown("")
                st.markdown("")
                st.markdown("""
                #### 거리값이 가장 낮은 순서로 정렬
                """)
                
                # 거리 기준 테이블
                st.markdown("""
                | 순위 | 페어 예시 | 거리 | 결과 |
                |------|-----------|------|------|
                | 1 | A-B | 2.3 | ✅ **선정** |
                | 2 | C-D | 3.7 | ✅ **선정** |
                | 3 | E-F | 5.2 | ✅ **선정** |
                | ... | ... | ... | ... |
                | 50 | Y-Z | 25.8 | ❌ 제외 |
                """)
                st.caption("💡 가장 비슷한 움직임을 보인 페어들을 우선 선택")
        
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
                    #### Half-Life 검증
                    - **정의**: 스프레드가 평균으로 절반 수렴하는 시간
                    - **계산**: AR(1) 모델 → HL = -ln(2)/ln(φ)
                    - **기준**: 5~60일 범위
                    """)
                
                with subcol2:
                    st.markdown("""
                    #### Half-Life 해석
                    - 5~15일: 단기 수익형 ⚡
                    - 15~30일: 우수한 페어 ⭐
                    - 30~60일: 중장기형 🕐
                    - >60일: 제외 ❌
                    """)
                
                st.markdown("""
                #### Z-Score 계산
                - **윈도우**: 60일 롤링
                - **목적**: 가격 괴리 신호 생성
                - **기준**: 적절한 통계적 유의성 확보
                """)
        
        st.markdown("---")
        
        # 핵심 요약
        st.success("""
        ### 🎯 핵심 전략
        **거리가 가장 작은 = 가격 경로가 가장 비슷한** 자산쌍들이 일시적으로 벌어질 때 수렴을 노리는 전략
        
        **✅ 장점**
        - 계산 속도 빠름
        - 직관적 이해 가능  
        - 강력한 평균회귀 신호 포착
        """)
        
        st.markdown("---")
        
        # 유클리드 거리 시각화
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📏 유클리드 거리 계산 예시")
            st.latex(r"""
            d(A, B) = \sqrt{\sum_{i=1}^{n} (P_A^i - P_B^i)^2}
            """)
            st.caption("""
            - PA, PB: 정규화된 가격 (첫날=1.0)
            - n: 관찰 기간 (예: 756일)
            - d: 유클리드 거리 (낮을수록 유사)
            """)
        
        with col2:
            st.markdown("#### 📈 거리에 따른 페어 품질")
            st.markdown("""
            | 거리 범위 | 해석 | 적합성 |
            |---------|------|--------|
            | 0 ~ 5 | 매우 유사 | ⭐⭐⭐ 최우선 |
            | 5 ~ 10 | 유사 | ⭐⭐ 양호 |
            | 10 ~ 20 | 보통 | ⭐ 고려 가능 |
            | > 20 | 상이 | ❌ 부적합 |
            """)
    
    with tab3:
        st.info("""
        ### 유클리드 거리 기반 페어 선정 방법론
        
        **핵심 원리**: 가격 움직임의 경로가 가장 유사한 자산들을 페어로 선정하여 평균회귀를 노리는 전략
        
        **상세 작동 과정**:
        1. **가격 정규화**: 최근 3년(756일) 데이터를 첫 거래일 = 1.0으로 리베이싱 → 절대 가격차이 제거, 상대적 움직임만 비교
        2. **유클리드 거리 계산**: 모든 자산 쌍에 대해 정규화된 가격 경로 간 유클리드 거리 측정  
           - 거리 공식: √Σ(Asset1ᵢ - Asset2ᵢ)² 
        3. **거리 기준 스크리닝**: **거리값이 가장 낮은 순서로 정렬** → 가장 비슷한 움직임을 보인 페어들을 우선 선택
        4. **품질 필터링**: 
           - **Half-Life**: 5~60일 범위 (평균회귀 속도 검증)
             * 정의: 스프레드가 현재값에서 평균값으로 절반만큼 수렴하는데 걸리는 시간
             * 계산: AR(1) 모델로 HL = -ln(2)/ln(φ), φ는 자기회귀 계수
             * 해석: 5 ~ 15일=단기 수익형, 15 ~ 30일=우수한 페어, 30 ~ 60일=중장기형
           - **Z-Score 계산**: 60일 롤링 윈도우로 가격 괴리 신호 생성 (적절한 통계적 유의성 확보)
        
        **핵심**: 거리가 **가장 작은 = 가격 경로가 가장 비슷한** 자산쌍들이 일시적으로 벌어질 때 수렴을 노리는 전략
        
        **장점**: 계산 속도 빠름, 직관적 이해 가능, 강력한 평균회귀 신호 포착
        """)
    
    with tab4:
        st.markdown("### 수학적 기초 및 계산 과정")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 1️⃣ 가격 정규화 (Rebasing)")
            st.latex(r"P_{norm}^t = \frac{P^t}{P^0}")
            st.caption("모든 자산의 시작점을 1.0으로 통일")
            
            st.markdown("#### 2️⃣ 유클리드 거리")
            st.latex(r"d_{AB} = \sqrt{\sum_{t=1}^{T} (P_A^t - P_B^t)^2}")
            st.caption("T기간 동안의 가격 경로 차이")
        
        with col2:
            st.markdown("#### 3️⃣ 스프레드 계산")
            st.latex(r"S_t = P_A^t - \beta \cdot P_B^t")
            st.caption("β는 OLS 회귀로 추정한 헤지비율")
            
            st.markdown("#### 4️⃣ Z-Score 신호")
            st.latex(r"Z_t = \frac{S_t - \mu_S}{\sigma_S}")
            st.caption("μ: 평균, σ: 표준편차 (롤링 윈도우)")
        
        st.markdown("---")
        
        # 실제 계산 예시
        st.markdown("#### 💡 실제 계산 예시")
        
        example_col1, example_col2 = st.columns(2)
        
        with example_col1:
            st.code("""
# 1. 정규화
asset_A = [100, 105, 110, 108]
asset_B = [50, 52, 54, 53]

norm_A = [1.00, 1.05, 1.10, 1.08]
norm_B = [1.00, 1.04, 1.08, 1.06]

# 2. 거리 계산
differences = [0, 0.01, 0.02, 0.02]
squared = [0, 0.0001, 0.0004, 0.0004]
distance = sqrt(0.0009) = 0.03
            """, language='python')
        
        with example_col2:
            st.markdown("""
            **해석**:
            - 거리 0.03은 매우 낮음
            - 두 자산의 움직임이 거의 동일
            - 페어트레이딩에 적합한 후보
            
            **다음 단계**:
            1. Half-Life 계산 (5-60일 확인)
            2. 거래비용 대비 수익성 검증
            3. Z-Score 모니터링 시작
            """)

# Streamlit 페이지로 실행
main()