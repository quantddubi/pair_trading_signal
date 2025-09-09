"""
페어트레이딩 분석 도구 - 상관관계 레짐 전환 기반 방법론
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
regime_module = import_module_from_file(os.path.join(project_root, "methods/4_correlation_regime_pairs.py"), "correlation_regime_pairs")

# 필요한 함수들 import
load_data = common_utils.load_data
normalize_prices = common_utils.normalize_prices
calculate_spread = common_utils.calculate_spread
calculate_zscore = common_utils.calculate_zscore
CorrelationRegimePairTrading = regime_module.CorrelationRegimePairTrading

# 페이지 설정
st.set_page_config(
    page_title="상관관계 레짐 방법론",
    page_icon="🔄",
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
def analyze_pairs(formation_window, signal_window, long_corr_window, short_corr_window, enter_threshold, exit_threshold, stop_loss, min_half_life, max_half_life, min_cost_ratio, min_delta_corr, n_pairs):
    """페어 분석 실행 (캐시 우선 사용)"""
    
    # 기본 파라미터와 일치하는지 확인
    user_params = {
        'formation_window': formation_window,
        'signal_window': signal_window,
        'long_corr_window': long_corr_window,
        'short_corr_window': short_corr_window,
        'enter_threshold': enter_threshold,
        'exit_threshold': exit_threshold,
        'stop_loss': stop_loss,
        'min_half_life': min_half_life,
        'max_half_life': max_half_life,
        'min_cost_ratio': min_cost_ratio,
        'min_delta_corr': min_delta_corr
    }
    
    # 기본 파라미터와 일치하면 캐시 사용
    if cache_utils.parameters_match_default('regime', user_params):
        cache_data = cache_utils.load_cache('regime')
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
    
    trader = CorrelationRegimePairTrading(
        formation_window=formation_window,
        signal_window=signal_window,
        long_corr_window=long_corr_window,
        short_corr_window=short_corr_window,
        enter_threshold=enter_threshold,
        exit_threshold=exit_threshold,
        stop_loss=stop_loss,
        min_half_life=min_half_life,
        max_half_life=max_half_life,
        min_cost_ratio=min_cost_ratio,
        min_delta_corr=min_delta_corr
    )
    
    enter_list, watch_list = trader.screen_pairs(prices, n_pairs=n_pairs)
    return enter_list, watch_list, prices

def create_pair_chart(prices, asset1, asset2, formation_window, long_corr_window, short_corr_window, asset_mapping=None):
    """상관관계 레짐 분석 차트 생성"""
    # 전체 기간 데이터
    end_date = prices.index[-1]
    start_date = end_date - timedelta(days=int(formation_window * 1.4))  # 여유를 두고
    
    chart_data = prices.loc[start_date:end_date, [asset1, asset2]].dropna()
    
    if len(chart_data) == 0:
        st.error(f"데이터가 없습니다: {asset1}, {asset2}")
        return None
    
    # 가격 정규화 (리베이스)
    normalized_data = normalize_prices(chart_data, method='rebase')
    
    # 상관관계 계산
    try:
        # 장기 상관관계 (롤링)
        long_corr = normalized_data[asset1].rolling(window=long_corr_window).corr(normalized_data[asset2])
        
        # 단기 상관관계 (롤링)
        short_corr = normalized_data[asset1].rolling(window=short_corr_window).corr(normalized_data[asset2])
        
        # 상관관계 차이 (레짐 변화 신호)
        corr_diff = short_corr - long_corr
        
        # 스프레드 및 Z-스코어 계산
        recent_data = chart_data.tail(formation_window)
        normalized_recent = normalize_prices(recent_data, method='rebase')
        spread = calculate_spread(normalized_recent[asset1], normalized_recent[asset2], hedge_ratio=1.0)
        zscore_window = max(20, min(60, len(spread)//4))
        zscore = calculate_zscore(spread, window=zscore_window)
        
    except Exception as e:
        st.error(f"상관관계 레짐 분석 중 오류 발생: {str(e)}")
        return None
    
    # 서브플롯 생성
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.25, 0.2, 0.2, 0.2, 0.15],
        subplot_titles=[
            f'{asset1} vs {asset2} - 정규화된 가격',
            f'장기 상관관계 ({long_corr_window}일 롤링)',
            f'단기 상관관계 ({short_corr_window}일 롤링)',
            '상관관계 차이 (레짐 변화 신호)',
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
    
    # 2. 장기 상관관계
    fig.add_trace(
        go.Scatter(
            x=long_corr.index,
            y=long_corr.values,
            name=f'장기 상관관계 ({long_corr_window}일)',
            line=dict(color='darkblue', width=2),
            hovertemplate='<b>장기 상관관계</b><br>Date: %{x}<br>Correlation: %{y:.3f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 3. 단기 상관관계
    fig.add_trace(
        go.Scatter(
            x=short_corr.index,
            y=short_corr.values,
            name=f'단기 상관관계 ({short_corr_window}일)',
            line=dict(color='orange', width=2),
            hovertemplate='<b>단기 상관관계</b><br>Date: %{x}<br>Correlation: %{y:.3f}<extra></extra>'
        ),
        row=3, col=1
    )
    
    # 4. 상관관계 차이 (레짐 변화)
    fig.add_trace(
        go.Scatter(
            x=corr_diff.index,
            y=corr_diff.values,
            name='상관관계 차이',
            line=dict(color='purple', width=2),
            hovertemplate='<b>상관관계 차이</b><br>Date: %{x}<br>Δ Correlation: %{y:.3f}<extra></extra>'
        ),
        row=4, col=1
    )
    
    # 상관관계 차이 기준선들
    fig.add_hline(y=0.15, line_dash="dash", line_color="green", opacity=0.7, row=4, col=1)
    fig.add_hline(y=-0.15, line_dash="dash", line_color="green", opacity=0.7, row=4, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=4, col=1)
    
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
        fig.add_hline(y=1.8, line_dash="dash", line_color="orange", opacity=0.7, row=5, col=1)
        fig.add_hline(y=-1.8, line_dash="dash", line_color="orange", opacity=0.7, row=5, col=1)
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
        chart_title = f"상관관계 레짐 분석: {name1}({asset1}) - {name2}({asset2})"
    else:
        chart_title = f"상관관계 레짐 분석: {asset1} - {asset2}"
    
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
    fig.update_yaxes(title_text="Correlation", row=2, col=1)
    fig.update_yaxes(title_text="Correlation", row=3, col=1)
    fig.update_yaxes(title_text="Δ Correlation", row=4, col=1)
    fig.update_yaxes(title_text="Z-Score", row=5, col=1)
    
    # 현재 값들을 주석으로 추가
    if len(zscore.dropna()) > 0:
        current_zscore = zscore_values[-1] if len(zscore_values) > 0 else 0
        current_corr_diff = corr_diff.dropna().iloc[-1] if len(corr_diff.dropna()) > 0 else 0
        
        fig.add_annotation(
            x=zscore_dates[-1] if len(zscore_dates) > 0 else end_date,
            y=current_zscore,
            text=f"현재 Z-Score: {current_zscore:.2f}<br>Δ 상관관계: {current_corr_diff:.3f}",
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
    default_params = cache_utils.get_default_parameters('regime')
    for key, value in default_params.items():
        if params.get(key) != value:
            return False
    return True

# 메인 앱
def main():
    st.title("상관관계 레짐 전환 기반 페어트레이딩")
    st.markdown("---")
    
    # 사이드바 설정
    st.sidebar.header("분석 설정")
    st.sidebar.markdown("### 기간 설정")
    
    formation_window = st.sidebar.slider(
        "Formation Window (일)",
        min_value=100,
        max_value=800,
        value=504,
        step=50,
        help="페어 선정을 위한 과거 데이터 기간 (기본값: 2년)"
    )
    
    signal_window = st.sidebar.slider(
        "Signal Window (일)", 
        min_value=20, 
        max_value=300, 
        value=126,
        help="Z-score 계산을 위한 롤링 윈도우 (기본값: 6개월)"
    )
    
    long_corr_window = st.sidebar.slider(
        "장기 상관관계 윈도우 (일)",
        min_value=100,
        max_value=600,
        value=378,
        step=50,
        help="장기 구조적 상관관계 계산 기간 (기본값: 18개월)"
    )
    
    short_corr_window = st.sidebar.slider(
        "단기 상관관계 윈도우 (일)",
        min_value=20,
        max_value=300,
        value=126,
        help="단기 상관관계 계산 기간 (기본값: 6개월)"
    )
    
    st.sidebar.markdown("### 신호 설정")
    
    enter_threshold = st.sidebar.slider(
        "진입 임계값 (Z-score)", 
        min_value=1.0, 
        max_value=3.0, 
        value=1.8, 
        step=0.1,
        help="이 값 이상일 때 진입 신호 생성 (기본값: 1.8)"
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
        max_value=150, 
        value=90,
        help="평균회귀 최대 속도 기준 (기본값: 90일)"
    )
    
    min_cost_ratio = st.sidebar.slider(
        "최소 비용비율", 
        min_value=1.0, 
        max_value=10.0, 
        value=3.0, 
        step=0.5,
        help="거래비용 대비 수익 최소 비율 (기본값: 3.0)"
    )
    
    min_delta_corr = st.sidebar.slider(
        "최소 상관관계 변화", 
        min_value=0.05, 
        max_value=0.8, 
        value=0.15, 
        step=0.05,
        help="레짐 변화로 인정할 최소 상관관계 차이 (기본값: 0.15)"
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
        'long_corr_window': long_corr_window,
        'short_corr_window': short_corr_window,
        'enter_threshold': enter_threshold,
        'exit_threshold': exit_threshold,
        'stop_loss': stop_loss,
        'min_half_life': min_half_life,
        'max_half_life': max_half_life,
        'min_cost_ratio': min_cost_ratio,
        'min_delta_corr': min_delta_corr
    }
    
    # 기본값 여부 확인
    is_default = check_parameters_default(params)
    
    # 메인 콘텐츠
    with st.spinner("상관관계 레짐 페어 분석 중... 잠시만 기다려주세요."):
        try:
            if is_default:
                st.info("🚀 기본 파라미터를 사용 중. 사전 계산된 결과를 즉시 표시")
                
                # 캐시에서 결과 로드
                cache_data = cache_utils.load_cache('regime')
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
                    formation_window, signal_window, long_corr_window, short_corr_window,
                    enter_threshold, exit_threshold, stop_loss, min_half_life, max_half_life, 
                    min_cost_ratio, min_delta_corr, n_pairs
                )
            
            asset_mapping = load_asset_names()  # 자산 이름 매핑 로딩
            
        except Exception as e:
            st.error(f"분석 중 오류 발생: {str(e)}")
            return
    
    # 4개 탭 구성 (아이콘 + 명칭 통일)
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 분석 결과 요약",
        "📊 상세 작동 과정", 
        "📝 상세 설명",
        "🔍 수식 및 계산"
    ])
    
    # =====================================================
    # TAB 1: 분석 결과 요약
    # =====================================================
    with tab1:
        # 분석 결과 요약 메트릭
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("진입 신호", f"{len(enter_list)}개", help="Z-스코어 임계값 이상의 페어")
        
        with col2:
            st.metric("관찰 대상", f"{len(watch_list)}개", help="진입 직전 단계의 페어")
        
        with col3:
            st.metric("상관관계 윈도우", f"{long_corr_window}/{short_corr_window}일", help="장기/단기 상관관계 분석 기간")
            
        with col4:
            avg_zscore = np.mean([abs(s['current_zscore']) for s in enter_list]) if enter_list else 0
            st.metric("평균 Z-스코어", f"{avg_zscore:.2f}", help="진입 신호들의 평균 Z-스코어")
        
        st.markdown("---")
        
        # 진입 신호 테이블
        if enter_list:
            st.subheader("추천 진입 페어")
            
            # 테이블 데이터 준비
            table_data = []
            for i, signal in enumerate(enter_list, 1):
                formatted_pair = format_pair_name(signal['pair'], asset_mapping)
                table_data.append({
                    "순위": i,
                    "페어": formatted_pair,
                    "방향": signal['direction'],
                    "Z-Score": f"{signal['current_zscore']:.2f}",
                    "Δ상관관계": f"{signal.get('delta_correlation', 0.0):.3f}",
                    "장기상관": f"{signal.get('long_corr', 0.0):.3f}",
                    "단기상관": f"{signal.get('short_corr', 0.0):.3f}",
                    "Half-Life": f"{signal.get('half_life', 0.0):.1f}일"
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
                    "Δ상관관계": st.column_config.TextColumn("Δ상관관계", width="small"),
                    "장기상관": st.column_config.TextColumn("장기상관", width="small"),
                    "단기상관": st.column_config.TextColumn("단기상관", width="small"),
                    "Half-Life": st.column_config.TextColumn("Half-Life", width="small")
                }
            )
            
            st.markdown("---")
            
            # 통합 페어 상세 분석 섹션
            st.subheader("페어 상세 분석")
            
            # 최고 추천 페어 표시
            top_pair = enter_list[0]
            top_formatted_pair = format_pair_name(top_pair['pair'], asset_mapping)
            st.success(f"최고 추천 페어: {top_formatted_pair}")
            
            # 모든 진입&관찰 페어를 통합하여 선택 가능하도록 구성
            all_pairs = enter_list + watch_list
            all_pair_options = [signal['pair'] for signal in all_pairs]
            all_pair_display_names = [format_pair_name(signal['pair'], asset_mapping) for signal in all_pairs]
            
            # selectbox에서 표시할 옵션들 생성
            pair_mapping = {display: original for display, original in zip(all_pair_display_names, all_pair_options)}
            
            selected_display_pair = st.selectbox(
                "분석할 페어 선택:",
                options=all_pair_display_names,
                index=0,
                help="차트로 분석할 페어를 선택하세요 (진입 + 관찰 페어 모두 포함)"
            )
            
            # 선택된 페어의 상세 정보 표시
            selected_pair = pair_mapping[selected_display_pair]
            selected_pair_info = None
            
            # 선택된 페어의 정보 찾기 (진입 리스트와 관찰 리스트 모두에서)
            for signal in all_pairs:
                if signal['pair'] == selected_pair:
                    selected_pair_info = signal
                    break
            
            if selected_pair_info:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    direction = selected_pair_info.get('direction', '관찰중')
                    st.metric("진입 방향", direction)
                with col2:
                    st.metric("현재 Z-Score", f"{selected_pair_info['current_zscore']:.2f}")
                with col3:
                    st.metric("Δ상관관계", f"{selected_pair_info.get('delta_correlation', 0.0):.3f}")
                with col4:
                    regime_strength = "강함" if abs(selected_pair_info.get('delta_correlation', 0.0)) > 0.5 else "보통"
                    st.metric("레짐 강도", regime_strength)
            
            if selected_pair:
                asset1, asset2 = selected_pair.split('-')
                
                # 차트 생성 및 표시
                with st.spinner(f"{selected_display_pair} 상관관계 레짐 차트 생성 중..."):
                    fig = create_pair_chart(prices, asset1, asset2, formation_window, long_corr_window, short_corr_window, asset_mapping)
                    
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 차트 설명
                        st.info("""
                        **상관관계 레짐 차트 설명:**
                        - **상단**: 두 자산의 정규화된 가격 추이
                        - **2번째**: 장기 상관관계 (구조적 기본 관계)
                        - **3번째**: 단기 상관관계 (최근 변화 추세)
                        - **4번째**: 상관관계 차이 (레짐 변화 신호) - ±0.15 이상시 유의미
                        - **하단**: Z-스코어 (평균회귀 진입 신호)
                        - **노란색 배경**: 최근 6개월 기간
                        - **초록색 선**: 레짐 변화 임계값 (±0.15)
                        - **주황색 선**: 진입 임계값 (±1.8)
                        """)
        
        else:
            st.warning("현재 진입 조건을 만족하는 상관관계 레짐 페어가 없습니다.")
            st.info("상관관계 변화 임계값을 낮추거나 Z-Score 임계값을 낮춰보세요.")
        
        # 관찰 대상 테이블
        if watch_list:
            st.subheader("관찰 대상 페어")
            
            table_data = []
            for i, signal in enumerate(watch_list, 1):
                formatted_pair = format_pair_name(signal['pair'], asset_mapping)
                table_data.append({
                    "순위": i,
                    "페어": formatted_pair,
                    "Z-Score": f"{signal['current_zscore']:.2f}",
                    "Δ상관관계": f"{signal.get('delta_correlation', 0.0):.3f}",
                    "장기상관": f"{signal.get('long_corr', 0.0):.3f}",
                    "단기상관": f"{signal.get('short_corr', 0.0):.3f}",
                    "Half-Life": f"{signal.get('half_life', 0.0):.1f}일"
                })
            
            df_watch = pd.DataFrame(table_data)
            st.dataframe(df_watch, use_container_width=True, hide_index=True)
        
        # 캐시 정보 표시 (기본 파라미터 사용시에만)
        if is_default and 'cache_data' in locals():
            st.markdown("---")
            st.caption(f"📅 캐시 생성일: {cache_data.get('generated_at', 'Unknown')}")
            st.caption(f"📊 데이터 기준일: {cache_data.get('data_date', 'Unknown')}")
    
    # =====================================================
    # TAB 2: 상세 작동 과정
    # =====================================================
    with tab2:
        st.markdown("### 상관관계 레짐 방법론 상세 작동 과정")
        
        # STEP별 작동 과정 (상관관계 레짐 특화)
        st.markdown("#### STEP 1: 이중 상관관계 추적")
        st.info("""
        - **장기 상관관계 (378일)**: 구조적 기본 관계 추적 (18개월)
        - **단기 상관관계 (126일)**: 최근 변화 추세 포착 (6개월)
        - 두 상관관계를 동시에 모니터링하여 레짐 변화 감지
        """)
        
        st.markdown("#### STEP 2: 레짐 변화 감지")
        st.info("""
        - **Δ상관관계 = 단기상관 - 장기상관** 계산
        - **양수**: 최근 상관관계 증가 (동조화) → 기회 감소
        - **음수**: 최근 상관관계 감소 (독립화) → 페어트레이딩 기회!
        - **임계값**: |Δ상관관계| > 0.15 시 유의미한 레짐 변화 (완화된 기준)
        """)
        
        st.markdown("#### STEP 3: 가격 괴리 확인")
        st.info("""
        - Z-스코어를 통한 가격 괴리 측정
        - 상관관계가 약해진 시점에서 가격 괴리 발생 시 진입 기회
        - 구조적 연관성은 유지하되 단기적 독립 움직임 활용
        """)
        
        st.markdown("#### STEP 4: 품질 필터링")
        st.info("""
        - **Δ상관관계 최소값**: 0.15 이상 (충분한 레짐 변화 확인)
        - **Half-Life**: 적절한 평균회귀 속도 (5-90일)
        - **비용비율**: 거래비용 대비 수익성 검증 (3.0 이상)
        """)
        
        st.markdown("#### STEP 5: 타이밍 최적화")
        st.info("""
        - 시장 혼란기, 섹터별 차별화 시점 포착
        - 개별 뉴스 반응 차이로 인한 일시적 상관관계 약화 활용
        - 레짐 변화와 가격 괴리의 동시 발생 시점에 진입
        """)
    
    # =====================================================
    # TAB 3: 상세 설명
    # =====================================================
    with tab3:
        st.markdown("### 상관관계 레짐 전환 기반 페어 선정 방법론")
        
        st.markdown("#### 📍 핵심 원리")
        st.info("""
        두 자산 간 상관관계가 시간에 따라 변하는 **레짐(체제) 전환을 포착**하여, 
        상관관계가 일시적으로 약해질 때 평균회귀를 노리는 전략
        """)
        
        st.markdown("#### 🎯 핵심 아이디어")
        st.success("""
        **상관관계가 약해진 타이밍**에 **가격이 괴리된 페어**를 발견하여 **관계 정상화**를 노리는 전략
        """)
        
        st.markdown("#### ⚡ 레짐 변화 감지 메커니즘")
        st.markdown("""
        **1. 이중 상관관계 추적**
        - **장기 상관관계 (378일)**: 구조적 기본 관계 (시장 전반의 기본 연동성, 18개월)
        - **단기 상관관계 (126일)**: 최근 변화 추세 (단기 시장 충격/뉴스 반응, 6개월)
        
        **2. 레짐 변화 신호**
        - **Δ상관관계 = 단기상관 - 장기상관**
        - **양수**: 최근 상관관계 증가 (더 동조화) → 분산 기회 감소
        - **음수**: 최근 상관관계 감소 (독립적 움직임) → 페어트레이딩 기회!
        - **임계값**: |Δ상관관계| > 0.15 시 유의미한 레짐 변화로 판단 (완화된 기준)
        """)
        
        st.markdown("#### 🎪 활용 시나리오")
        st.markdown("""
        **타이밍 최적화가 핵심**
        - **시장 혼란기**: 전반적 불확실성으로 인한 상관관계 약화
        - **섹터별 차별화**: 업종별로 다른 뉴스/이벤트 반응
        - **개별 뉴스 반응**: 한 자산만의 고유 이벤트 발생
        - **구조적 연관성 유지**: 장기적으로는 여전히 관련있는 자산들
        """)
        
        st.markdown("#### ✅ 장점 vs ❌ 단점")
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("""
            **✅ 장점**
            - 시장 레짐 변화 대응 우수
            - 위기 상황에서 기회 포착
            - 동적 페어 선정 가능
            - 전통적 상관관계 방법 대비 정교함
            """)
        
        with col2:
            st.error("""
            **❌ 단점**
            - 복잡한 신호 해석 필요
            - 상관관계 불안정성 존재
            - 레짐 지속성 불확실
            - 계산 복잡도 높음
            """)
        
        st.markdown("#### 🔧 품질 필터링")
        st.info("""
        **Δ상관관계 최소값**: 0.15 이상 (충분한 레짐 변화, 완화된 기준)
        **Half-Life**: 5~90일 (적절한 평균회귀 속도)
        **진입 임계값**: Z-Score 1.8 이상 (완화된 기준)
        **비용비율**: 거래비용 대비 수익성 검증 (3.0 이상)
        """)
    
    # =====================================================
    # TAB 4: 수식 및 계산
    # =====================================================
    with tab4:
        st.markdown("### 상관관계 레짐 방법론 수식 및 계산")
        
        st.markdown("#### 1. 이중 상관관계 계산")
        st.latex(r'''
        \text{장기 상관관계: } \rho_{long}(t) = \text{corr}(X_{t-378:t}, Y_{t-378:t})
        ''')
        st.latex(r'''
        \text{단기 상관관계: } \rho_{short}(t) = \text{corr}(X_{t-126:t}, Y_{t-126:t})
        ''')
        
        st.markdown("#### 2. 레짐 변화 신호 (Δ상관관계)")
        st.latex(r'''
        \Delta\rho(t) = \rho_{short}(t) - \rho_{long}(t)
        ''')
        
        st.markdown("**해석:**")
        st.markdown("""
        - $\Delta\rho > 0$: 최근 상관관계 증가 (동조화) → 기회 감소
        - $\Delta\rho < 0$: 최근 상관관계 감소 (독립화) → 페어트레이딩 기회
        - $|\Delta\rho| > 0.15$: 유의미한 레짐 변화 (완화된 기준)
        """)
        
        st.markdown("#### 3. Z-스코어 계산 (기본 평균회귀)")
        st.latex(r'''
        Z(t) = \frac{S(t) - \mu_{S}}{\sigma_{S}}
        ''')
        
        st.markdown("여기서:")
        st.latex(r'''
        S(t) = \log(X_t) - \beta \cdot \log(Y_t)
        ''')
        st.latex(r'''
        \beta = \frac{\text{cov}(X, Y)}{\text{var}(Y)}
        ''')
        
        st.markdown("#### 4. Half-Life 계산")
        st.latex(r'''
        \text{Half-Life} = \frac{\ln(2)}{-\ln(\phi)}
        ''')
        
        st.markdown("여기서 $\phi$는 AR(1) 모델의 자기회귀 계수:")
        st.latex(r'''
        S_t = \phi \cdot S_{t-1} + \epsilon_t
        ''')
        
        st.markdown("#### 5. 진입 조건")
        st.code("""
        진입 조건:
        1. |Δ상관관계| > min_delta_corr (기본값: 0.15)
        2. |Z-Score| > enter_threshold (기본값: 1.8)
        3. min_half_life < Half-Life < max_half_life (기본값: 5-90일)
        4. 비용비율 > min_cost_ratio (기본값: 3.0)
        """)
        
        st.markdown("#### 6. 계산 예시")
        
        if enter_list:
            # 첫 번째 페어를 예시로 사용
            example_pair = enter_list[0]
            formatted_pair = format_pair_name(example_pair['pair'], asset_mapping)
            
            st.markdown(f"**예시 페어: {formatted_pair}**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**입력 데이터:**")
                st.code(f"""
장기 상관관계: {example_pair.get('long_corr', 0.0):.3f}
단기 상관관계: {example_pair.get('short_corr', 0.0):.3f}
현재 Z-Score: {example_pair['current_zscore']:.3f}
Half-Life: {example_pair.get('half_life', 0.0):.1f}일
                """)
            
            with col2:
                st.markdown("**계산 결과:**")
                delta_corr = example_pair.get('delta_correlation', 0.0)
                st.code(f"""
Δ상관관계: {delta_corr:.3f}
레짐 변화: {'유의미' if abs(delta_corr) > 0.15 else '미미'}
진입 신호: {'진입' if abs(example_pair['current_zscore']) > 1.8 else '관찰'}
방향: {example_pair.get('direction', '관찰중')}
                """)
        
        st.markdown("#### 7. 상관관계 윈도우 최적화")
        st.info("""
        **장기 윈도우 (378일)**:
        - 18개월 거래일 기준 (1.5년)
        - 구조적 기본 관계 포착
        - 레짐 변화를 안정적으로 감지하기 위한 충분한 기간
        
        **단기 윈도우 (126일)**:
        - 6개월 거래일 기준  
        - 최근 변화 추세 민감하게 포착
        - 과도한 노이즈 방지와 반응성의 균형점
        """)
    
    # 푸터
    st.markdown("---")

# Streamlit 페이지로 실행
if __name__ == "__main__":
    main()
else:
    main()