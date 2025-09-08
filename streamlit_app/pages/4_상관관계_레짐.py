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
    """페어 이름을 티커(이름) 형태로 포맷팅"""
    asset1, asset2 = pair.split('-')
    
    name1 = asset_mapping.get(asset1, asset1)
    name2 = asset_mapping.get(asset2, asset2)
    
    return f"{asset1}({name1}) - {asset2}({name2})"

# 페어 분석 함수
@st.cache_data
def analyze_pairs(formation_window, signal_window, long_corr_window, short_corr_window, enter_threshold, exit_threshold, stop_loss, min_half_life, max_half_life, min_cost_ratio, min_delta_corr, n_pairs):
    """페어 분석 실행"""
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

def create_correlation_regime_chart(prices, asset1, asset2, formation_window, long_corr_window, short_corr_window, asset_mapping=None):
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
    fig.add_hline(y=0.3, line_dash="dash", line_color="green", opacity=0.7, row=4, col=1)
    fig.add_hline(y=-0.3, line_dash="dash", line_color="green", opacity=0.7, row=4, col=1)
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
        chart_title = f"상관관계 레짐 분석: {asset1}({name1}) - {asset2}({name2})"
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
    return params == default_params

# 메인 앱
def main():
    st.title("상관관계 레짐 전환 기반 페어트레이딩")
    st.markdown("---")
    
    # 상관관계 레짐 방법론 설명
    st.info("""
    ### 상관관계 레짐 전환 기반 페어 선정 방법론
    
    **핵심 원리**: 두 자산 간 상관관계가 시간에 따라 변하는 레짐(체제) 전환을 포착하여, 상관관계가 일시적으로 약해질 때 평균회귀를 노리는 전략
    
    **상세 작동 과정**:
    1. **이중 상관관계 추적**: 장기(252일)와 단기(60일) 롤링 상관관계를 동시에 모니터링
       - **장기 상관관계**: 구조적 기본 관계 (시장 전반의 기본 연동성)
       - **단기 상관관계**: 최근 변화 추세 (단기 시장 충격/뉴스 반응)
    2. **레짐 변화 감지**: Δ상관관계 = 단기상관 - 장기상관 계산
       - **양수**: 최근 상관관계가 증가 (더 동조화) → 분산 기회 감소
       - **음수**: 최근 상관관계가 감소 (독립적 움직임) → 페어트레이딩 기회!
       - **임계값**: |Δ상관관계| > 0.3 시 유의미한 레짐 변화로 판단
    3. **타이밍 최적화**: 상관관계가 낮아진 시점에서 가격 괴리 발생 시 진입
       - 구조적으로는 연관있지만(장기 상관관계 존재) 단기적으로 따로 움직일 때
       - 시장 혼란기, 섹터별 차별화, 개별 뉴스 반응 차이 등을 활용
    4. **품질 필터링**: 
       - **Δ상관관계 최소값**: 0.3 이상 (충분한 레짐 변화)
       - **Half-Life**: 5~60일 (적절한 평균회귀 속도)
       - **기본 평균회귀**: Z-Score 기반 진입/청산 신호 동일 적용
    
    **핵심**: **상관관계가 약해진 타이밍**에 **가격이 괴리된 페어**를 발견하여 **관계 정상화**를 노리는 전략
    
    **장점**: 시장 레짐 변화 대응 우수, 위기 상황 기회 포착, 동적 페어 선정
    **단점**: 복잡한 신호 해석, 상관관계 불안정성, 레짐 지속성 불확실
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
        help="페어 선정을 위한 과거 데이터 기간"
    )
    
    signal_window = st.sidebar.slider(
        "Signal Window (일)", 
        min_value=20, 
        max_value=120, 
        value=60,
        help="Z-score 계산을 위한 롤링 윈도우"
    )
    
    long_corr_window = st.sidebar.slider(
        "장기 상관관계 윈도우 (일)",
        min_value=100,
        max_value=500,
        value=252,
        step=50,
        help="장기 구조적 상관관계 계산 기간"
    )
    
    short_corr_window = st.sidebar.slider(
        "단기 상관관계 윈도우 (일)",
        min_value=20,
        max_value=120,
        value=60,
        help="단기 상관관계 계산 기간"
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
    
    min_delta_corr = st.sidebar.slider(
        "최소 상관관계 변화", 
        min_value=0.1, 
        max_value=0.8, 
        value=0.3, 
        step=0.1,
        help="레짐 변화로 인정할 최소 상관관계 차이"
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
    
    # 분석 결과 요약
    st.header("분석 결과 요약")
    
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
                st.metric("Δ상관관계", f"{selected_pair_info.get('delta_correlation', 0.0):.3f}")
            with col4:
                st.metric("레짐 강도", "강함" if abs(selected_pair_info.get('delta_correlation', 0.0)) > 0.5 else "보통")
        
        if selected_pair:
            asset1, asset2 = selected_pair.split('-')
            
            # 차트 생성 및 표시
            with st.spinner(f"{selected_display_pair} 상관관계 레짐 차트 생성 중..."):
                fig = create_correlation_regime_chart(prices, asset1, asset2, formation_window, long_corr_window, short_corr_window, asset_mapping)
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 차트 설명
                    st.info("""
                    **상관관계 레짐 차트 설명:**
                    - **상단**: 두 자산의 정규화된 가격 추이
                    - **2번째**: 장기 상관관계 (구조적 기본 관계)
                    - **3번째**: 단기 상관관계 (최근 변화 추세)
                    - **4번째**: 상관관계 차이 (레짐 변화 신호) - ±0.3 이상시 유의미
                    - **하단**: Z-스코어 (평균회귀 진입 신호)
                    - **노란색 배경**: 최근 6개월 기간
                    - **초록색 선**: 레짐 변화 임계값 (±0.3)
                    - **주황색 선**: 진입 임계값 (±2.0)
                    """)
    
    else:
        st.warning("현재 진입 조건을 만족하는 상관관계 레짐 페어가 없습니다.")
        st.info("상관관계 변화 임계값을 낮추거나 Z-Score 임계값을 낮춰보세요.")
    
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
    
    # 푸터
    st.markdown("---")

# Streamlit 페이지로 실행
if __name__ == "__main__":
    main()
else:
    main()