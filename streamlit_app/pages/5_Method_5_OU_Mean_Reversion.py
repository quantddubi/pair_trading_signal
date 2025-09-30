"""
페어트레이딩 분석 도구 - OU 평균회귀 속도 기반 방법론
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

# 모듈 경로 설정 및 로드
try:
    methods_dir = os.path.join(project_root, 'methods')
    sys.path.insert(0, methods_dir)
    ou_module_path = os.path.join(methods_dir, '5_ou_mean_reversion_pairs.py')
    ou_module = import_module_from_file(ou_module_path, 'ou_mean_reversion_pairs')
    OUMeanReversionPairTrading = ou_module.OUMeanReversionPairTrading
    
    # 추가 모듈 임포트
    from utils import cache_utils
    
except ImportError as e:
    st.error(f"필요한 모듈을 찾을 수 없습니다: {e}")
    st.stop()

# Streamlit 페이지 설정
st.set_page_config(
    page_title="OU 평균회귀 속도 기반 페어트레이딩",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 스타일 설정
st.markdown("""
<style>
    /* 메인 컨테이너 스타일 */
    .main {
        padding-top: 1rem;
    }
    
    /* 메트릭 스타일링 */
    [data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #e0e2e6;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
    }
    
    /* 데이터프레임 스타일 */
    .dataframe {
        font-size: 14px;
    }
    
    /* 헤더 스타일 */
    h1, h2, h3 {
        color: #1f2937;
    }
</style>
""", unsafe_allow_html=True)

# 가격 데이터 로딩 (캐시 없음 - 항상 최신 데이터 로드)
def load_price_data():
    """가격 데이터 로딩"""
    data_path = get_data_file_path()
    # common_utils의 load_data 함수 사용 (BBG 형식 헤더 처리)
    from utils.common_utils import load_data
    prices = load_data(data_path)
    return prices

# 자산명 매핑 로딩
def load_asset_names():
    """자산 이름 매핑 로딩"""
    return cache_utils.get_asset_mapping()

# 포맷팅 함수
def format_pair_name(pair_str, asset_mapping):
    """페어 이름 포맷팅 (산업 정보 포함)"""
    assets = pair_str.split('-')
    if len(assets) == 2:
        asset1_info = asset_mapping.get(assets[0], {})
        asset2_info = asset_mapping.get(assets[1], {})
        
        asset1_display = f"{asset1_info.get('name', assets[0])} ({asset1_info.get('industry', 'N/A')})"
        asset2_display = f"{asset2_info.get('name', assets[1])} ({asset2_info.get('industry', 'N/A')})"
        
        return f"{asset1_display} - {asset2_display}"
    return pair_str

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

def create_pair_chart(prices, asset1, asset2, formation_window, signal_window, asset_mapping=None):
    """OU 평균회귀 분석 차트 생성"""
    # 전체 기간 데이터
    end_date = prices.index[-1]
    start_date = end_date - timedelta(days=int(formation_window * 1.4))  # 여유를 두고
    
    chart_data = prices.loc[start_date:end_date, [asset1, asset2]].dropna()
    
    if len(chart_data) == 0:
        st.error(f"데이터가 없습니다: {asset1}, {asset2}")
        return None
    
    # 가격 정규화 (리베이스)
    from utils.data_processing import normalize_prices
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
                    rolling_half_life.append(sub_half_life)
                except:
                    rolling_kappa.append(kappa)
                    rolling_half_life.append(half_life)
            else:
                rolling_kappa.append(kappa)
                rolling_half_life.append(half_life)
        
        # 롤링 데이터 날짜 정렬
        rolling_dates = spread.index[window:]
        
        # Z-score 계산 (신호 생성용)
        spread_mean = spread.rolling(window=signal_window).mean()
        spread_std = spread.rolling(window=signal_window).std()
        zscore = (spread - spread_mean) / spread_std
        
    except Exception as e:
        st.error(f"OU 분석 오류: {str(e)}")
        return None
    
    # Plotly 차트 생성 (5개 서브플롯)
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[
            "정규화된 가격",
            "OU 스프레드 & 장기 평균 (θ)",
            "평균회귀속도 κ (30일 롤링)",
            "반감기 (30일 롤링)",
            "Z-Score (진입 신호)"
        ],
        row_heights=[0.25, 0.2, 0.2, 0.2, 0.15]
    )
    
    # 형성 기간 배경색 설정
    formation_start = end_date - timedelta(days=formation_window)
    
    # 1. 정규화된 가격
    fig.add_trace(
        go.Scatter(x=normalized_data.index, y=normalized_data[asset1], 
                   name=asset_mapping.get(asset1, {}).get('name', asset1) if asset_mapping else asset1,
                   line=dict(color='blue', width=1.5)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=normalized_data.index, y=normalized_data[asset2], 
                   name=asset_mapping.get(asset2, {}).get('name', asset2) if asset_mapping else asset2,
                   line=dict(color='red', width=1.5)),
        row=1, col=1
    )
    
    # 2. OU 스프레드와 장기 평균
    fig.add_trace(
        go.Scatter(x=spread.index, y=spread, 
                   name="OU 스프레드", 
                   line=dict(color='purple', width=1.5)),
        row=2, col=1
    )
    fig.add_hline(y=theta, line_dash="dash", line_color="orange", 
                  annotation_text=f"θ={theta:.3f}", row=2, col=1)
    
    # 3. 평균회귀속도 κ
    fig.add_trace(
        go.Scatter(x=rolling_dates, y=rolling_kappa,
                   name="κ (평균회귀속도)", 
                   line=dict(color='green', width=1.5)),
        row=3, col=1
    )
    fig.add_hline(y=0.01, line_dash="dash", line_color="red", 
                  annotation_text="최소 κ=0.01", row=3, col=1)
    
    # 4. 반감기
    fig.add_trace(
        go.Scatter(x=rolling_dates, y=rolling_half_life,
                   name="반감기 (일)", 
                   line=dict(color='brown', width=1.5)),
        row=4, col=1
    )
    fig.add_hline(y=5, line_dash="dash", line_color="green", 
                  annotation_text="최소=5일", row=4, col=1)
    fig.add_hline(y=60, line_dash="dash", line_color="red", 
                  annotation_text="최대=60일", row=4, col=1)
    
    # 5. Z-Score
    fig.add_trace(
        go.Scatter(x=zscore.index, y=zscore, 
                   name="Z-Score", 
                   line=dict(color='darkblue', width=1.5)),
        row=5, col=1
    )
    fig.add_hline(y=2, line_dash="dash", line_color="orange", 
                  annotation_text="진입", row=5, col=1)
    fig.add_hline(y=-2, line_dash="dash", line_color="orange", 
                  annotation_text="진입", row=5, col=1)
    fig.add_hline(y=0, line_dash="solid", line_color="gray", row=5, col=1)
    
    # 형성 기간 하이라이트 (모든 서브플롯에 적용)
    for i in range(1, 6):
        fig.add_vrect(
            x0=formation_start, x1=end_date,
            fillcolor="yellow", opacity=0.1,
            layer="below", line_width=0,
            row=i, col=1
        )
    
    # 차트 제목
    if asset_mapping:
        asset1_name = asset_mapping.get(asset1, {}).get('name', asset1)
        asset2_name = asset_mapping.get(asset2, {}).get('name', asset2)
        chart_title = f"OU 평균회귀 분석: {asset1_name} vs {asset2_name}"
    else:
        chart_title = f"OU 평균회귀 분석: {asset1} vs {asset2}"
    
    # 레이아웃 설정
    fig.update_layout(
        height=1000,
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
    fig.update_yaxes(title_text="Days", row=4, col=1)
    fig.update_yaxes(title_text="Z-Score", row=5, col=1)
    
    # 통계 정보를 주석으로 추가
    fig.add_annotation(
        x=rolling_dates[-1] if len(rolling_dates) > 0 else end_date,
        y=rolling_kappa[-1] if len(rolling_kappa) > 0 else kappa,
        text=f"현재 κ: {rolling_kappa[-1] if len(rolling_kappa) > 0 else kappa:.4f}<br>반감기: {rolling_half_life[-1] if len(rolling_half_life) > 0 else half_life:.1f}일",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="green",
        font=dict(size=12, color="green"),
        row=3, col=1
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
    
    # 사이드바 설정 (TAB 1에서만 활성화)
    st.sidebar.header("Analysis Settings")
    st.sidebar.markdown("### 기간 설정")
    
    formation_window = st.sidebar.slider(
        "Formation Window (일)",
        min_value=100,
        max_value=500,
        value=252,
        step=50,
        help="OU 과정 추정을 위한 과거 데이터 기간"
    )
    
    rolling_window = st.sidebar.slider(
        "Rolling Window (일)", 
        min_value=30, 
        max_value=120, 
        value=60,
        help="OU 파라미터 추정용 롤링 윈도우"
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
    if st.sidebar.button("Run Analysis", type="primary"):
        pass  # 캐시 사용 안 함
    
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
    
    # 캐시 우선 사용 로직 (SSD 페이지와 동일한 방식)
    try:
        cache_data = cache_utils.load_cache('ou')
        if cache_data and cache_utils.parameters_match_default('ou', params):
            st.info("📋 캐시된 결과를 사용합니다 (통합 스크리너와 동일)")
            enter_list = cache_data.get('enter_signals', [])
            watch_list = cache_data.get('watch_signals', [])
            prices = load_price_data()
            asset_mapping = load_asset_names()
        else:
            st.info("🔄 사용자 설정으로 실시간 계산합니다")
            # 메인 콘텐츠
            with st.spinner("OU 평균회귀 페어 분석 중... 잠시만 기다려주세요."):
                # 실시간 분석 실행
                selected_pairs, prices = analyze_pairs(
                    formation_window, rolling_window, enter_threshold, exit_threshold,
                    stop_loss, min_half_life, max_half_life, min_cost_ratio, 
                    min_mean_reversion_speed, max_kappa_cv, data_coverage_threshold, 
                    winsorize_percentile, n_pairs
                )
                # 실시간 분석 결과를 enter_list와 watch_list로 분리
                enter_list = [p for p in selected_pairs if abs(p.get('current_zscore', 0)) >= enter_threshold]
                watch_list = [p for p in selected_pairs if abs(p.get('current_zscore', 0)) < enter_threshold]
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
            st.metric("Entry Signals", f"{len(enter_list)}개", help="Z-스코어 임계값 이상의 페어")
        
        with col2:
            st.metric("Watch List", f"{len(watch_list)}개", help="진입 직전 단계의 페어")
        
        with col3:
            avg_kappa = np.mean([pair.get('kappa_avg', 0.01) for pair in enter_list + watch_list]) if (enter_list + watch_list) else 0
            st.metric("평균 κ", f"{avg_kappa:.4f}", help="평균회귀속도")
            
        with col4:
            avg_half_life = np.mean([pair.get('half_life_avg', 50) for pair in enter_list + watch_list]) if (enter_list + watch_list) else 0
            st.metric("평균 반감기", f"{avg_half_life:.1f}일", help="선별된 페어들의 평균 반감기")
        
        st.markdown("---")
        
        # 진입 신호 테이블
        if enter_list:
            st.subheader("추천 진입 페어")
            
            # 테이블 데이터 준비
            table_data = []
            for i, pair_info in enumerate(enter_list, 1):
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
                    "방향": pair_info.get('direction', 'LONG/SHORT'),
                    "Z-Score": f"{pair_info.get('current_zscore', 0):.2f}",
                    "κ (속도)": f"{pair_info.get('kappa_avg', 0.01):.4f}",
                    "반감기": f"{pair_info.get('half_life_avg', 50):.1f}일",
                    "품질점수": f"{pair_info.get('quality_score', 0.0):.1f}",
                    "헤지비율": f"{pair_info.get('hedge_ratio', 1.0):.4f}"
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
                    "방향": st.column_config.TextColumn("진입 방향", width="small"),
                    "Z-Score": st.column_config.TextColumn("Z-Score", width="small"),
                    "κ (속도)": st.column_config.TextColumn("κ (속도)", width="small"),
                    "반감기": st.column_config.TextColumn("반감기", width="small"),
                    "품질점수": st.column_config.TextColumn("품질점수", width="small"),
                    "헤지비율": st.column_config.TextColumn("헤지비율", width="small")
                }
            )
        else:
            st.warning("❌ 현재 진입 조건을 만족하는 페어가 없습니다")
        
        st.markdown("---")
        
        # 통합 페어 상세 분석 섹션 (진입 페어가 있을 때만 표시)
        if enter_list:
            st.subheader("Pair Detail Analysis")
            
            # 최고 추천 페어 표시
            top_pair = enter_list[0]
            if 'pair' in top_pair:
                top_pair_str = top_pair['pair']
            else:
                top_pair_str = f"{top_pair['asset1']}-{top_pair['asset2']}"
            top_formatted_pair = format_pair_name(top_pair_str, asset_mapping)
            st.success(f"최고 품질 페어: {top_formatted_pair}")
            
            # 모든 진입&관찰 페어를 통합하여 선택 가능하도록 구성
            all_pairs = enter_list + watch_list
            all_pair_options = []
            all_pair_display_names = []
            
            for pair_info in all_pairs:
                if 'pair' in pair_info:
                    pair_str = pair_info['pair']
                else:
                    pair_str = f"{pair_info.get('asset1', '')}-{pair_info.get('asset2', '')}"
                all_pair_options.append(pair_str)
                all_pair_display_names.append(format_pair_name(pair_str, asset_mapping))
            
            # selectbox에서 표시할 옵션들 생성
            if all_pair_display_names:
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
                    if 'pair' in signal:
                        pair_str = signal['pair']
                    else:
                        pair_str = f"{signal.get('asset1', '')}-{signal.get('asset2', '')}"
                    
                    if pair_str == selected_pair:
                        selected_pair_info = signal
                        break
                
                if selected_pair_info:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        direction = selected_pair_info.get('direction', '관찰중')
                        st.metric("진입 방향", direction)
                    with col2:
                        st.metric("현재 Z-Score", f"{selected_pair_info.get('current_zscore', 0):.2f}")
                    with col3:
                        st.metric("κ (평균회귀속도)", f"{selected_pair_info.get('kappa_avg', 0.01):.4f}")
                    with col4:
                        st.metric("반감기", f"{selected_pair_info.get('half_life_avg', 50):.1f}일")
                
                if selected_pair and '-' in selected_pair:
                    asset1, asset2 = selected_pair.split('-')
                    
                    # 차트 생성 및 표시
                    with st.spinner(f"{selected_display_pair} OU 평균회귀 차트 생성 중..."):
                        fig = create_pair_chart(prices, asset1, asset2, formation_window, rolling_window, asset_mapping)
                        
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
                            - **노란색 배경**: 최근 분석 기간
                            - **초록/빨간색 선**: 반감기 기준 (5일/60일)
                            - **주황색 선**: 진입 임계값 (±2.0)
                            """)
        
        else:
            st.warning("현재 진입 조건을 만족하는 OU 평균회귀 페어가 없습니다.")
            st.info("평균회귀속도 κ 최소값을 낮추거나 Z-Score 임계값을 낮춰보세요.")
        
        # 관찰 대상 테이블
        if watch_list:
            st.subheader("관찰 대상 페어")
            
            table_data = []
            for i, pair_info in enumerate(watch_list, 1):
                if 'pair' in pair_info:
                    formatted_pair = format_pair_name(pair_info['pair'], asset_mapping)
                else:
                    pair_str = f"{pair_info.get('asset1', '')}-{pair_info.get('asset2', '')}"
                    formatted_pair = format_pair_name(pair_str, asset_mapping)
                
                table_data.append({
                    "순위": i,
                    "페어": formatted_pair,
                    "Z-Score": f"{pair_info.get('current_zscore', 0):.2f}",
                    "κ (속도)": f"{pair_info.get('kappa_avg', 0.01):.4f}",
                    "반감기": f"{pair_info.get('half_life_avg', 50):.1f}일",
                    "품질점수": f"{pair_info.get('quality_score', 0.0):.1f}"
                })
            
            df_watch = pd.DataFrame(table_data)
            st.dataframe(df_watch, use_container_width=True, hide_index=True)
        else:
            st.info("현재 관찰 대상 페어가 없습니다")
        
        # 캐시 정보 표시 (기본 파라미터 사용시에만)
        if is_default and 'cache_data' in locals():
            st.markdown("---")
            st.caption(f"📅 캐시 생성일: {cache_data.get('generated_at', 'Unknown')}")
            st.caption(f"📊 데이터 기준일: {cache_data.get('data_date', 'Unknown')}")
    
    # =====================================================
    # TAB 2: 상세 작동 과정
    # =====================================================
    with tab2:
        st.markdown("### OU Mean Reversion Methodology 상세 작동 과정")
        
        # STEP별 작동 과정 (OU 특화)
        st.markdown("#### STEP 1: 스프레드 OU 과정 적합")
        st.info("""
        - **AR(1) 근사**: ΔSpread(t) = α + β×Spread(t-1) + ε(t)
        - **OU 변환**: κ = -β/Δt, θ = -α/β
        - 각 페어의 스프레드를 Ornstein-Uhlenbeck 모델로 추정
        """)
        
        st.markdown("#### STEP 2: 평균회귀속도 품질평가")
        st.warning("""
        - **κ > 0.01**: 유의미한 평균회귀 존재 확인
        - **κ 안정성**: 시간에 따른 κ 변동성 최소화
        - **반감기**: Half-Life = ln(2)/κ → 5~60일 범위 선호
        """)
        
        st.markdown("#### STEP 3: 동적 모니터링")
        st.success("""
        - 30일 롤링 윈도우로 κ 실시간 추적
        - κ 증가 → 더 강한 평균회귀 → 좋은 신호
        - κ 감소 → 약화된 평균회귀 → 위험 신호
        """)
        
        st.markdown("#### STEP 4: 품질 필터링")
        st.error("""
        - **평균 κ**: 0.01 이상 (충분한 평균회귀 강도)
        - **κ 일관성**: 시간에 따른 안정적 유지
        - **Half-Life**: 5~60일 (적절한 수렴 속도)
        """)
        
        st.markdown("#### STEP 5: 진입 타이밍 최적화")
        st.info("""
        - Z-Score + κ 강도 조합으로 최적 진입점 포착
        - 높은 κ + 높은 Z-Score = 최상의 진입 기회
        - 수학적으로 검증된 평균회귀 강도 활용
        """)
    
    # =====================================================
    # TAB 3: 상세 설명
    # =====================================================
    with tab3:
        st.markdown("### OU(Ornstein-Uhlenbeck) 평균회귀 기반 페어 선정 방법론")
        
        st.markdown("#### 📍 핵심 원리")
        st.info("""
        스프레드가 평균으로 돌아가는 속도(평균회귀속도 κ)를 수학적으로 모델링하여,
        가장 빠르고 안정적으로 수렴하는 페어를 선정하는 전략
        """)
        
        st.markdown("#### 🎯 OU 과정 수학적 모델")
        st.success("""
        **dX(t) = κ(θ - X(t))dt + σdW(t)**
        
        - **κ (kappa)**: 평균회귀속도 → 클수록 빠른 수렴
        - **θ (theta)**: 장기평균 → 스프레드가 수렴할 목표점
        - **σ (sigma)**: 변동성 → 노이즈 수준
        - **W(t)**: 브라운 운동 (랜덤 충격)
        """)
        
        st.markdown("#### ⚡ OU 파라미터 추정 과정")
        st.markdown("""
        **1. AR(1) 모델로 근사**
        - 스프레드 차분: ΔS(t) = S(t) - S(t-1)
        - AR(1) 회귀: ΔS(t) = α + β×S(t-1) + ε(t)
        - 최소자승법으로 α, β 추정
        
        **2. OU 파라미터 변환**
        - κ = -β/Δt (Δt = 1/252 for daily data)
        - θ = -α/β (장기 평균)
        - σ = std(ε) × √(2κ) (변동성)
        
        **3. 반감기 계산**
        - Half-Life = ln(2)/κ
        - 스프레드가 절반으로 수렴하는 시간
        """)
        
        st.markdown("#### 🎪 활용 시나리오")
        st.markdown("""
        **최적 활용 상황**
        - **안정적 관계**: 구조적으로 연결된 자산 페어
        - **높은 κ 값**: 빠른 평균회귀 속도
        - **낮은 변동성**: 예측 가능한 수렴 패턴
        - **일관된 반감기**: 시간에 따라 안정적
        """)
        
        st.markdown("#### ✅ 장점 vs ❌ 단점")
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("""
            **✅ 장점**
            - 수학적 엄밀성과 이론적 기반
            - 평균회귀 강도 정량화
            - 동적 품질 모니터링
            - 진입/청산 타이밍 최적화
            - 리스크 관리 용이
            """)
        
        with col2:
            st.error("""
            **❌ 단점**
            - 모델 복잡성 높음
            - 구조변화 감지 지연
            - 계산 집약적
            - 파라미터 추정 오류 가능
            - 비정상성 가정 위반 위험
            """)
        
        st.markdown("#### 🔧 품질 필터링 기준")
        st.info("""
        **평균회귀속도 κ**: 0.01 이상 (충분한 평균회귀 강도)
        **반감기**: 5~60일 (너무 빠르거나 느리지 않은 수렴)
        **κ 안정성**: CV < 0.6 (시간에 따른 일관성)
        **데이터 커버리지**: 90% 이상 (데이터 품질)
        **비용비율**: 5.0 이상 (거래비용 대비 수익성)
        """)
    
    # =====================================================
    # TAB 4: 수식 및 계산
    # =====================================================
    with tab4:
        st.markdown("### OU Mean Reversion Methodology 수식 및 계산")
        
        st.markdown("#### 1. OU 확률미분방정식")
        st.latex(r'''
        dX_t = \kappa(\theta - X_t)dt + \sigma dW_t
        ''')
        
        st.markdown("**파라미터 해석:**")
        st.markdown("""
        - $X_t$: 시점 t에서의 스프레드
        - $\kappa$: 평균회귀속도 (mean reversion speed)
        - $\theta$: 장기평균 (long-term mean)
        - $\sigma$: 변동성 (volatility)
        - $W_t$: 브라운 운동 (Brownian motion)
        """)
        
        st.markdown("#### 2. AR(1) 근사 및 파라미터 추정")
        st.latex(r'''
        X_t - X_{t-1} = \alpha + \beta X_{t-1} + \epsilon_t
        ''')
        
        st.latex(r'''
        \kappa = -\frac{\beta}{\Delta t}, \quad \theta = -\frac{\alpha}{\beta}, \quad \sigma = \text{std}(\epsilon) \sqrt{2\kappa}
        ''')
        
        st.markdown("#### 3. 반감기 (Half-Life) 계산")
        st.latex(r'''
        \text{Half-Life} = \frac{\ln(2)}{\kappa}
        ''')
        
        st.markdown("**해석**: 스프레드가 현재 값에서 장기평균까지의 거리를 절반으로 줄이는 데 걸리는 시간")
        
        st.markdown("#### 4. OU 과정의 평균과 분산")
        st.latex(r'''
        E[X_t | X_0] = X_0 e^{-\kappa t} + \theta(1 - e^{-\kappa t})
        ''')
        
        st.latex(r'''
        \text{Var}[X_t | X_0] = \frac{\sigma^2}{2\kappa}(1 - e^{-2\kappa t})
        ''')
        
        st.markdown("#### 5. Z-Score 계산")
        st.latex(r'''
        Z_t = \frac{X_t - \mu_X}{\sigma_X}
        ''')
        
        st.markdown("여기서 $\mu_X$와 $\sigma_X$는 롤링 윈도우에서 계산된 평균과 표준편차")
        
        st.markdown("#### 6. 계산 예시")
        
        if enter_list:
            # 첫 번째 페어를 예시로 사용
            example_pair = enter_list[0]
            
            # 페어 이름 처리
            if 'pair' in example_pair:
                pair_str = example_pair['pair']
            else:
                pair_str = f"{example_pair.get('asset1', 'Asset1')}-{example_pair.get('asset2', 'Asset2')}"
            
            formatted_pair = format_pair_name(pair_str, asset_mapping)
            
            st.markdown(f"**예시 페어: {formatted_pair}**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**OU 파라미터:**")
                st.code(f"""
κ (평균회귀속도): {example_pair.get('kappa_avg', 0.01):.4f}
θ (장기평균): {example_pair.get('theta_avg', 0.0):.3f}
σ (변동성): {example_pair.get('sigma', 0.1):.3f}
반감기: {example_pair.get('half_life_avg', 50):.1f}일
                """)
            
            with col2:
                st.markdown("**거래 신호:**")
                st.code(f"""
현재 Z-Score: {example_pair.get('current_zscore', 0):.3f}
헤지 비율: {example_pair.get('hedge_ratio', 1.0):.4f}
품질 점수: {example_pair.get('quality_score', 0.0):.1f}
진입 신호: {'진입' if abs(example_pair.get('current_zscore', 0)) > 2.0 else '관찰'}
                """)
        
        st.markdown("#### 7. Python 구현 예시")
        st.code("""
import numpy as np
from sklearn.linear_model import LinearRegression

def estimate_ou_parameters(spread, dt=1/252):
    \"\"\"OU 파라미터 추정\"\"\"
    # AR(1) 회귀
    spread_lag = spread[:-1].values.reshape(-1, 1)
    spread_diff = spread.diff()[1:].values
    
    reg = LinearRegression()
    reg.fit(spread_lag, spread_diff)
    
    beta = reg.coef_[0]
    alpha = reg.intercept_
    
    # OU 파라미터 변환
    kappa = -beta / dt
    theta = -alpha / beta if beta != 0 else 0
    
    # 반감기
    half_life = np.log(2) / kappa if kappa > 0 else np.inf
    
    return kappa, theta, half_life
        """, language='python')
        
        st.markdown("#### 8. 최적화 팁")
        st.info("""
        **파라미터 선택 가이드:**
        - **Formation Window**: 252일 (1년) - 충분한 데이터로 안정적 추정
        - **Rolling Window**: 60일 - 최근 변화 반영과 안정성 균형
        - **최소 κ**: 0.01 - 연간 약 2.5회 평균회귀
        - **반감기 범위**: 5-60일 - 너무 빠르거나 느린 수렴 제외
        - **κ CV**: < 0.6 - 시간에 따른 안정성 확보
        """)
    
    # 푸터
    st.markdown("---")

# Streamlit 페이지로 실행
if __name__ == "__main__":
    main()
else:
    main()