"""
Pair Trading Analysis Tool - Clustering Methodology
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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
clustering_module = import_module_from_file(os.path.join(project_root, "methods/6_clustering_pairs.py"), "clustering_pairs")

# 필요한 함수들 import
load_data = common_utils.load_data
normalize_prices = common_utils.normalize_prices
calculate_spread = common_utils.calculate_spread
calculate_zscore = common_utils.calculate_zscore
ClusteringPairTrading = clustering_module.ClusteringPairTrading

# 페이지 설정
st.set_page_config(
    page_title="Clustering Methodology",
    page_icon="🧠",
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
def analyze_pairs(formation_days, signal_days, enter_threshold, n_pairs, n_clusters, clustering_method):
    """페어 분석 실행"""
    prices = load_price_data()
    
    trader = ClusteringPairTrading(
        formation_window=formation_days,
        signal_window=signal_days,
        enter_threshold=enter_threshold,
        exit_threshold=0.5,
        stop_loss=3.0,
        min_half_life=5,
        max_half_life=60,
        min_cost_ratio=5.0,
        n_clusters=n_clusters,
        clustering_method=clustering_method
    )
    
    enter_list, watch_list = trader.screen_pairs(prices, n_pairs=n_pairs)
    cluster_summary = trader.get_cluster_summary(prices)
    return enter_list, watch_list, cluster_summary, prices

def create_cluster_visualization(prices, trader, formation_days):
    """클러스터 시각화 생성"""
    # 최근 데이터 추출
    formation_data = prices.tail(formation_days)
    
    # 유효한 자산만 선택
    valid_assets = [col for col in formation_data.columns 
                   if formation_data[col].notna().sum() >= formation_data.shape[0] * 0.8]
    
    if len(valid_assets) < 4:
        return None
        
    formation_data = formation_data[valid_assets].fillna(method='ffill')
    
    # 특징 추출 및 클러스터링
    features = trader.extract_features(formation_data)
    if len(features) < 4:
        return None
        
    clusters = trader.perform_clustering(features)
    
    # PCA로 2차원 축소
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(trader.scaler.fit_transform(features))
    
    # 클러스터 레이블 생성
    cluster_labels = []
    asset_list = []
    
    for cluster_id, assets in clusters.items():
        for asset in assets:
            if asset in features.index:
                cluster_labels.append(cluster_id)
                asset_list.append(asset)
    
    # 2D 시각화 데이터 생성
    viz_df = pd.DataFrame({
        'PC1': features_2d[:, 0],
        'PC2': features_2d[:, 1],
        'Asset': asset_list,
        'Cluster': cluster_labels
    })
    
    # Plotly scatter plot 생성
    fig = px.scatter(
        viz_df, 
        x='PC1', 
        y='PC2', 
        color='Cluster',
        text='Asset',
        title='클러스터링 결과 (PCA 2D 투영)',
        labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%} 설명력)',
                'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%} 설명력)'},
        color_continuous_scale='viridis'
    )
    
    fig.update_traces(textposition="middle right")
    fig.update_layout(
        height=600,
        showlegend=True,
        template='plotly_white'
    )
    
    return fig

def create_pair_chart(prices, asset1, asset2, formation_days, signal_days, asset_mapping=None):
    """페어 차트 생성 (Clustering Methodology에 맞게 조정)"""
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
            f'{asset1} vs {asset2} - 정규화된 가격 (클러스터링 기반)',
            'Spread (Price Difference)',
            'Z-Score (클러스터링 기반 신호)'
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
        chart_title = f"클러스터링 기반 페어분석: {name1}({asset1}) - {name2}({asset2})"
    else:
        chart_title = f"클러스터링 기반 페어분석: {asset1} - {asset2}"
    
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
    st.title("🧠 Clustering Pair Trading")
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
    
    formation_days = st.sidebar.slider(
        "분석 기간 (일)",
        min_value=252,
        max_value=1260,
        value=756,
        step=126,
        help="페어 선정 및 클러스터링을 위한 과거 데이터 기간"
    )
    
    enter_threshold = st.sidebar.slider(
        "진입 Z-스코어 임계값",
        min_value=1.5,
        max_value=3.0,
        value=2.0,
        step=0.1,
        help="이 값 이상일 때 진입 신호 생성"
    )
    
    n_clusters = st.sidebar.slider(
        "클러스터 개수",
        min_value=3,
        max_value=15,
        value=8,
        step=1,
        help="자산을 몇 개 클러스터로 그룹화할지 설정"
    )
    
    clustering_method = st.sidebar.selectbox(
        "클러스터링 방법",
        ["kmeans", "hierarchical"],
        index=0,
        help="K-Means 또는 계층적 클러스터링 선택"
    )
    
    # 분석 실행 버튼
    if st.sidebar.button("🚀 분석 실행", type="primary"):
        st.cache_data.clear()
    
    # 기본 파라미터 설정
    signal_days = 60
    n_pairs = 10
    
    # 공통 분석 수행
    with st.spinner("🧠 클러스터링 기반 페어 분석 중..."):
        try:
            enter_list, watch_list, cluster_summary, prices = analyze_pairs(
                formation_days, signal_days, enter_threshold, n_pairs, n_clusters, clustering_method
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
            st.metric("진입 신호 개수", f"{len(enter_list)}개", help="Z-스코어 임계값 이상의 페어")
        
        with col2:
            st.metric("관찰 대상 개수", f"{len(watch_list)}개", help="진입 직전 단계의 페어")
        
        with col3:
            st.metric("활성 클러스터 수", f"{len(cluster_summary)}개", help="페어가 발굴된 클러스터 수")
        
        with col4:
            avg_zscore = np.mean([abs(s['current_zscore']) for s in enter_list]) if enter_list else 0
            st.metric("평균 Z-스코어", f"{avg_zscore:.2f}", help="진입 신호들의 평균 Z-스코어")
        
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
                    "클러스터": f"C{signal.get('cluster_id', 'N/A')}",
                    "거리": f"{signal.get('distance', 0):.3f}",
                    "Half-Life": f"{signal.get('half_life', 0):.1f}일"
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
                    "클러스터": st.column_config.TextColumn("클러스터", width="small"),
                    "거리": st.column_config.TextColumn("클러스터내거리", width="small"),
                    "Half-Life": st.column_config.TextColumn("평균회귀속도", width="small")
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
                    "클러스터": f"C{signal.get('cluster_id', 'N/A')}",
                    "거리": f"{signal.get('distance', 0):.3f}"
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
                    "클러스터": st.column_config.TextColumn("클러스터", width="small"),
                    "거리": st.column_config.TextColumn("클러스터내거리", width="small")
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
                    st.metric("진입 방향", selected_pair_info.get('direction', 'N/A'))
                with col3:
                    st.metric("클러스터 ID", f"C{selected_pair_info.get('cluster_id', 'N/A')}")
                with col4:
                    st.metric("클러스터 내 거리", f"{selected_pair_info.get('distance', 0):.3f}")
            
            # 인터랙티브 차트
            if selected_pair:
                asset1, asset2 = selected_pair.split('-')
                
                with st.spinner(f"📊 {selected_display_pair} 차트 생성 중..."):
                    fig = create_pair_chart(prices, asset1, asset2, formation_days, signal_days, asset_mapping)
                    
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 차트 해석 가이드
                        with st.expander("📖 차트 해석 가이드"):
                            st.info("""
                            **🧠 클러스터링 기반 차트 해석:**
                            - **상단 차트**: 두 자산의 정규화된 가격 추이 (동일 클러스터 내 유사 자산)
                            - **중간 차트**: 스프레드 (클러스터링으로 선정된 페어 간 가격 차이)
                            - **하단 차트**: Z-스코어 (클러스터 기반 평균회귀 신호)
                            
                            **💡 신호 해석:**
                            - 노란색 배경: 최근 6개월 기간 (신호 발생 구간)
                            - 주황색 선: 진입 임계값 (±2.0)
                            - Z-스코어 > +2.0: Short 첫째자산, Long 둘째자산
                            - Z-스코어 < -2.0: Long 첫째자산, Short 둘째자산
                            
                            **🎯 클러스터링 특징:**
                            - 펀더멘털이 유사한 자산끼리 매칭
                            - 클러스터 내 거리가 가까울수록 높은 품질
                            """)
        else:
            st.info("💡 분석할 페어가 없습니다. 임계값을 조정해보세요.")
    
    # TAB 2: 📊 상세 작동 과정
    with tab2:
        st.header("Clustering Pair Trading 작동 과정")
        
        # STEP 1
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("### STEP 1")
            st.info("**다차원 특징 추출**")
        with col2:
            st.markdown("""
            **10가지 핵심 특성 계산**
            
            각 자산별로 다음 특징들을 계산합니다:
            - **수익률 통계**: 평균 수익률, 변동성, 왜도, 첨도
            - **가격 동학**: 추세, 최대낙폭, 변동성 클러스터링 효과  
            - **시장 관계**: 베타, 모멘텀 비율, 가격 일관성
            
            이를 통해 각 자산을 10차원 특징벡터로 표현합니다.
            """)
        
        # STEP 2
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("### STEP 2")
            st.warning("**지능형 클러스터링**")
        with col2:
            st.markdown(f"""
            **{clustering_method.upper()} 클러스터링 실행**
            
            - **목표 클러스터 수**: {n_clusters}개
            - **방법**: {'K-Means 알고리즘' if clustering_method == 'kmeans' else '계층적 클러스터링'}
            - **표준화**: StandardScaler로 특징 정규화
            - **결과**: 유사한 특성을 가진 자산들을 그룹화
            
            같은 클러스터 = 비슷한 펀더멘털 특성을 의미합니다.
            """)
        
        # STEP 3
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("### STEP 3")
            st.success("**클러스터 내 페어 발굴**")
        with col2:
            st.markdown("""
            **유클리드 거리 기반 페어 매칭**
            
            동일 클러스터 내에서:
            - 모든 자산 간 10차원 유클리드 거리 계산
            - 거리가 가장 가까운 페어부터 순서대로 선별
            - 거리 최소 = 가격 경로 최대 유사성
            
            **결과**: 펀더멘털 + 기술적 유사성을 모두 만족하는 고품질 페어
            """)
        
        # STEP 4
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("### STEP 4")
            st.error("**품질 필터링**")
        with col2:
            st.markdown("""
            **엄격한 품질 기준 적용**
            
            각 페어에 대해:
            - **Half-Life 체크**: 5~60일 범위 (평균회귀 속도)
            - **비용비율 체크**: 최소 5.0 이상 (수익성)
            - **Z-스코어 계산**: 롤링 윈도우 기반 신호 생성
            
            **최종 결과**: 거래 가능한 고품질 페어만 선별
            """)
        
        # 마무리 요소
        st.success("""
        **🎯 Clustering Methodology의 핵심 전략**
        
        단순한 가격 유사성을 넘어 **펀더멘털 특성의 유사성**까지 고려하여 보다 안정적이고 
        예측 가능한 페어를 발굴하는 지능형 접근법입니다.
        """)
        
        # 방법론별 시각화 (2개 컬럼)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("클러스터 구성 요약")
            if cluster_summary:
                for cluster_id, info in list(cluster_summary.items())[:4]:  # 상위 4개만
                    st.metric(
                        f"클러스터 {cluster_id}",
                        f"{info['n_assets']}개 자산",
                        f"평균 변동성: {info.get('avg_volatility', 0):.3f}"
                    )
        
        with col2:
            st.subheader("클러스터 시각화")
            
            # 클러스터 시각화 생성
            temp_trader = ClusteringPairTrading(
                formation_window=formation_days,
                signal_window=signal_days,
                enter_threshold=enter_threshold,
                exit_threshold=0.5,
                stop_loss=3.0,
                min_half_life=5,
                max_half_life=60,
                min_cost_ratio=5.0,
                n_clusters=n_clusters,
                clustering_method=clustering_method
            )
            
            cluster_fig = create_cluster_visualization(prices, temp_trader, formation_days)
            if cluster_fig:
                # 작은 크기로 조정
                cluster_fig.update_layout(height=400)
                st.plotly_chart(cluster_fig, use_container_width=True)
    
    # TAB 3: 📝 상세 설명
    with tab3:
        st.info("""
        ### 클러스터링 기반 페어 선정 방법론

        **핵심 원리**: 유사한 특성을 가진 자산들을 클러스터로 그룹화한 후, 클러스터 내에서 최근접 이웃 페어를 발굴하는 지능형 전략

        **상세 작동 과정**:
        1. **다차원 특징 추출**: 각 자산별로 10가지 핵심 특성 계산
           - 수익률 통계: 평균 수익률, 변동성, 왜도, 첨도
           - 가격 동학: 추세, 최대낙폭, 변동성 클러스터링 효과
           - 시장 관계: 베타, 모멘텀 비율, 가격 일관성
        2. **지능형 클러스터링**: K-Means 또는 계층적 클러스터링으로 유사 자산군 식별
           - 표준화된 특징벡터 기반으로 클러스터 형성
           - PCA 차원축소를 통한 시각적 확인 가능
        3. **클러스터 내 페어 발굴**: 동일 클러스터 내 자산 간 유클리드 거리 최소화
           - 같은 클러스터 = 비슷한 기본 특성
           - 거리 최소 = 가격 경로 최대 유사성
        4. **품질 필터링**: 
           - **Half-Life**: 5~60일 범위 (클러스터링 효과 + 평균회귀 속도)
           - **비용비율**: 최소 5.0 이상 (거래비용 대비 수익 잠재력)

        **핵심**: **비슷한 펀더멘털 + 비슷한 가격 움직임**을 동시에 만족하는 고품질 페어 발굴

        **장점**: 단순 가격 유사성을 넘어 근본적 특성까지 고려한 정교한 페어 선정, 클러스터별 리스크 분산 효과
        
        **특별한 특징**:
        - 머신러닝 기반의 객관적 페어 선정
        - 다차원 특징공간에서의 유사도 측정
        - 클러스터별 위험 분산 효과
        - PCA 시각화를 통한 직관적 이해 제공
        
        **적용 시나리오**:
        - 대량의 자산에서 체계적 페어 발굴
        - 섹터 내 또는 섹터 간 관계 분석
        - 리스크 관리를 위한 다각화된 페어 포트폴리오 구성
        """)
    
    # TAB 4: 🔍 수식 및 계산
    with tab4:
        # 2개 컬럼 레이아웃
        col1, col2 = st.columns(2)
        
        # 왼쪽: 핵심 수식
        with col1:
            st.subheader("핵심 수식")
            
            st.markdown("**1. 특징벡터 구성**")
            st.latex(r'''
            F_i = [r_{avg}, \sigma, skew, kurt, trend, mdd, \beta, mom, consistency, vol\_cluster]
            ''')
            
            st.markdown("**2. 유클리드 거리**")
            st.latex(r'''
            d(i,j) = \sqrt{\sum_{k=1}^{10} (F_{i,k} - F_{j,k})^2}
            ''')
            
            st.markdown("**3. K-Means 목적함수**")
            st.latex(r'''
            \min J = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2
            ''')
        
        # 오른쪽: 보조 수식
        with col2:
            st.subheader("보조 수식")
            
            st.markdown("**1. Z-Score 계산**")
            st.latex(r'''
            Z_t = \frac{S_t - \mu_S}{\sigma_S}
            ''')
            
            st.markdown("**2. Half-Life 추정**")
            st.latex(r'''
            HL = \frac{\ln(2)}{-\ln(1 + \beta)}
            ''')
            
            st.markdown("**3. 비용비율**")
            st.latex(r'''
            Cost\_Ratio = \frac{\sigma_S \times \sqrt{252}}{2 \times transaction\_cost}
            ''')
        
        st.markdown("---")
        
        # 실제 계산 예시
        col1, col2 = st.columns(2)
        
        # 왼쪽: Python 코드 예시  
        with col1:
            st.subheader("Python 구현 예시")
            
            st.code("""
# 특징 추출 예시
def extract_features(returns):
    features = {}
    
    # 기본 통계량
    features['avg_return'] = returns.mean()
    features['volatility'] = returns.std()
    features['skewness'] = returns.skew()
    features['kurtosis'] = returns.kurtosis()
    
    # 추세 지표
    cumret = (1 + returns).cumprod()
    features['trend'] = np.polyfit(range(len(cumret)), 
                                 cumret, 1)[0]
    
    # 최대낙폭
    roll_max = cumret.expanding().max()
    drawdown = (cumret - roll_max) / roll_max
    features['max_drawdown'] = drawdown.min()
    
    return features

# K-Means 클러스터링
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_df)

kmeans = KMeans(n_clusters=8, random_state=42)
clusters = kmeans.fit_predict(features_scaled)
            """)
        
        # 오른쪽: 해석 및 활용법
        with col2:
            st.subheader("해석 및 활용법")
            
            st.markdown("""
            **특징벡터 해석:**
            - **수익률 통계**: 자산의 기본 수익성과 위험 프로필
            - **가격 동학**: 자산의 움직임 패턴과 변동성 특성  
            - **시장 관계**: 시장 전체와의 상관관계 및 모멘텀
            
            **클러스터링 활용:**
            - **클러스터 수 선택**: 엘보우 방법 또는 실루엣 분석
            - **거리 임계값**: 클러스터 내 페어의 품질 관리
            - **특징 가중치**: 도메인 지식에 따른 조정 가능
            
            **실전 팁:**
            - 클러스터 수는 자산 수의 1/3~1/5 수준 권장
            - PCA 시각화로 클러스터 품질 확인
            - 정기적 재클러스터링으로 시장 변화 반영
            
            **성과 모니터링:**
            - 클러스터별 성과 추적
            - 거리와 수익률의 상관관계 분석
            - 클러스터 안정성 지표 모니터링
            """)

if __name__ == "__main__":
    main()