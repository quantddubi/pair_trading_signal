"""
페어트레이딩 분석 도구 - 클러스터링 기반 방법론
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

# 프로젝트 루트 경로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
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
clustering_module = import_module_from_file(os.path.join(project_root, "methods/5_clustering_pairs.py"), "clustering_pairs")

# 필요한 함수들 import
load_data = common_utils.load_data
normalize_prices = common_utils.normalize_prices
calculate_spread = common_utils.calculate_spread
calculate_zscore = common_utils.calculate_zscore
ClusteringPairTrading = clustering_module.ClusteringPairTrading

# 페이지 설정
st.set_page_config(
    page_title="클러스터링 방법론",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 캐시된 데이터 로딩
@st.cache_data
def load_price_data():
    """가격 데이터 로딩"""
    file_path = "/Users/a/PycharmProjects/pair_trading_signal/data/MU Price(BBG).csv"
    return load_data(file_path)

@st.cache_data
def load_asset_names():
    """자산 이름 매핑 로딩 (CSV 파일의 1행: 티커, 2행: 이름)"""
    file_path = "/Users/a/PycharmProjects/pair_trading_signal/data/MU Price(BBG).csv"
    
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
    """페어 차트 생성 (클러스터링 방법론에 맞게 조정)"""
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
    zscore_window = min(signal_days, len(spread)//2) if len(spread) > 20 else max(20, len(spread)//4)
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
        chart_title = f"클러스터링 기반 페어분석: {asset1}({name1}) - {asset2}({name2})"
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
    st.title("클러스터링 기반 페어트레이딩")
    st.markdown("---")
    
    # 클러스터링 방법론 설명
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
    """)
    
    # 사이드바 설정
    st.sidebar.header("분석 설정")
    st.sidebar.markdown("### 기간 설정")
    
    formation_days = st.sidebar.slider(
        "분석 기간 (일)",
        min_value=252,
        max_value=1260,  # 5년
        value=756,       # 3년
        step=126,        # 6개월 단위
        help="페어 선정 및 클러스터링을 위한 과거 데이터 기간"
    )
    
    signal_days = st.sidebar.slider(
        "Z-스코어 계산 기간 (일)",
        min_value=20,
        max_value=120,
        value=60,
        step=10,
        help="Z-스코어 신호 계산을 위한 롤링 윈도우"
    )
    
    st.sidebar.markdown("### 신호 설정")
    
    enter_threshold = st.sidebar.slider(
        "진입 Z-스코어 임계값",
        min_value=1.5,
        max_value=3.0,
        value=2.0,
        step=0.1,
        help="이 값 이상일 때 진입 신호 생성"
    )
    
    st.sidebar.markdown("### 클러스터링 설정")
    
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
    with st.spinner("클러스터링 기반 페어 분석 중... 잠시만 기다려주세요."):
        try:
            enter_list, watch_list, cluster_summary, prices = analyze_pairs(
                formation_days, signal_days, enter_threshold, n_pairs, n_clusters, clustering_method
            )
            asset_mapping = load_asset_names()  # 자산 이름 매핑 로딩
        except Exception as e:
            st.error(f"분석 중 오류 발생: {str(e)}")
            return
    
    # 클러스터 요약 정보
    if cluster_summary:
        st.header("클러스터 구성 요약")
        
        # 클러스터별 메트릭 표시
        cols = st.columns(min(4, len(cluster_summary)))
        for i, (cluster_id, info) in enumerate(cluster_summary.items()):
            with cols[i % len(cols)]:
                st.metric(
                    f"클러스터 {cluster_id}",
                    f"{info['n_assets']}개 자산",
                    f"평균 변동성: {info.get('avg_volatility', 0):.3f}"
                )
                # 클러스터 내 주요 자산들 표시
                main_assets = info['assets'][:3]
                if len(info['assets']) > 3:
                    asset_text = f"{', '.join(main_assets)}... (+{len(info['assets'])-3}개)"
                else:
                    asset_text = ', '.join(main_assets)
                st.caption(f"주요 자산: {asset_text}")
        
        st.markdown("---")
    
    # 분석 결과 요약
    st.header("분석 결과 요약")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("진입 신호", f"{len(enter_list)}개", help="Z-스코어 임계값 이상의 페어")
    
    with col2:
        st.metric("관찰 대상", f"{len(watch_list)}개", help="진입 직전 단계의 페어")
    
    with col3:
        st.metric("활성 클러스터", f"{len(cluster_summary)}개", help="페어가 발굴된 클러스터 수")
        
    with col4:
        avg_zscore = np.mean([abs(s['current_zscore']) for s in enter_list]) if enter_list else 0
        st.metric("평균 Z-스코어", f"{avg_zscore:.2f}", help="진입 신호들의 평균 Z-스코어")
    
    st.markdown("---")
    
    # 클러스터 시각화
    st.header("클러스터 시각화")
    
    # 임시로 trader 객체 생성 (시각화용)
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
        st.plotly_chart(cluster_fig, use_container_width=True)
        st.info("""
        **클러스터 시각화 설명:**
        - 각 점은 개별 자산을 나타냄
        - 같은 색상 = 같은 클러스터 (유사한 특성)
        - 가까운 거리 = 높은 유사도
        - PCA로 고차원 특징을 2차원으로 투영하여 표시
        """)
    
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
                "클러스터": f"C{signal.get('cluster_id', 'N/A')}",
                "거리": f"{signal.get('distance', 0):.3f}",
                "Half-Life": f"{signal.get('half_life', 0):.1f}일"
            })
        
        df_enter = pd.DataFrame(table_data)
        
        # 스타일링된 테이블 표시
        st.dataframe(
            df_enter,
            use_container_width=True,
            hide_index=True,
            column_config={
                "순위": st.column_config.NumberColumn("순위", width="small"),
                "페어": st.column_config.TextColumn("페어", width="large"),
                "방향": st.column_config.TextColumn("진입 방향", width="large"),
                "Z-Score": st.column_config.TextColumn("Z-Score", width="small"),
                "클러스터": st.column_config.TextColumn("클러스터", width="small"),
                "거리": st.column_config.TextColumn("거리", width="small"),
                "Half-Life": st.column_config.TextColumn("Half-Life", width="small")
            }
        )
        
        st.markdown("---")
        
        # 페어 선택 및 차트 표시
        st.header("페어 상세 분석")
        
        # 최고 추천 페어 표시
        top_pair = enter_list[0]
        top_formatted_pair = format_pair_name(top_pair['pair'], asset_mapping)
        st.success(f"최고 추천 페어 (클러스터 {top_pair.get('cluster_id', 'N/A')}): {top_formatted_pair}")
        
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
                st.metric("클러스터 ID", f"C{selected_pair_info.get('cluster_id', 'N/A')}")
            with col4:
                st.metric("클러스터 내 거리", f"{selected_pair_info.get('distance', 0):.3f}")
        
        if selected_pair:
            asset1, asset2 = selected_pair.split('-')
            
            # 차트 생성 및 표시
            with st.spinner(f"{selected_display_pair} 차트 생성 중..."):
                fig = create_pair_chart(prices, asset1, asset2, formation_days, signal_days, asset_mapping)
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 차트 설명
                    st.info("""
                    **클러스터링 기반 차트 설명:**
                    - 상단: 두 자산의 정규화된 가격 추이 (같은 클러스터 내 유사 자산)
                    - 중간: 스프레드 (클러스터링으로 선정된 페어 간 가격 차이)
                    - 하단: Z-스코어 (클러스터 기반 평균회귀 신호)
                    - 노란색 배경: 최근 6개월 기간
                    - 주황색 선: 진입 임계값 (±2.0)
                    - 특징: 클러스터링을 통해 펀더멘털이 유사한 자산끼리 매칭
                    """)
    
    else:
        st.warning("현재 진입 조건을 만족하는 페어가 없음")
        st.info("임계값을 낮추거나 클러스터 수를 조정.")
    
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
                "클러스터": f"C{signal.get('cluster_id', 'N/A')}",
                "거리": f"{signal.get('distance', 0):.3f}",
                "Half-Life": f"{signal.get('half_life', 0):.1f}일"
            })
        
        df_watch = pd.DataFrame(table_data)
        st.dataframe(df_watch, use_container_width=True, hide_index=True)
    
    # 푸터
    st.markdown("---")

# Streamlit 페이지로 실행
main()