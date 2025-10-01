"""
Monitoring Universe Dendrogram - Anomaly Detection via Hierarchical Clustering
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
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

# 필요한 함수들
load_data = common_utils.load_data

# 페이지 설정
st.set_page_config(
    page_title="Monitoring Universe Dendrogram",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

def calculate_correlation_distance(prices, lookback_weeks=520):
    """
    상관관계 기반 거리 계산
    d = sqrt(1 - ρ) / sqrt(2)

    Args:
        prices: 가격 데이터
        lookback_weeks: 주간 수익률 계산 기간 (기본값: 520주 = 약 10년)

    Returns:
        distance_matrix: 거리 행렬
        correlation_matrix: 상관관계 행렬
    """
    # 주간 수익률 계산 (최근 lookback_weeks 주)
    weekly_prices = prices.resample('W').last()
    weekly_returns = weekly_prices.pct_change().dropna()

    # 최근 N주 데이터만 사용
    if len(weekly_returns) > lookback_weeks:
        weekly_returns = weekly_returns.tail(lookback_weeks)

    # 상관관계 계산
    correlation_matrix = weekly_returns.corr()

    # 거리로 변환: d = sqrt((1 - ρ) / 2)
    # ρ가 +1에 가까우면 d → 0 (가까움)
    # ρ가 -1에 가까우면 d → 1 (멀다)
    distance_matrix = np.sqrt((1 - correlation_matrix) / 2)

    return distance_matrix, correlation_matrix

def perform_hierarchical_clustering(distance_matrix, method='ward'):
    """
    계층적 군집분석 수행

    Args:
        distance_matrix: 거리 행렬
        method: linkage 방법 (ward, complete, average 등)

    Returns:
        linkage_matrix: linkage 결과
    """
    # 거리 행렬을 condensed form으로 변환
    condensed_distance = squareform(distance_matrix, checks=False)

    # Hierarchical clustering
    linkage_matrix = linkage(condensed_distance, method=method)

    return linkage_matrix

def get_cluster_assignments(linkage_matrix, n_clusters):
    """
    클러스터 할당 결과 얻기

    Args:
        linkage_matrix: linkage 결과
        n_clusters: 클러스터 개수

    Returns:
        cluster_labels: 각 자산의 클러스터 레이블
    """
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    return cluster_labels

def detect_anomalies_in_cluster(weekly_returns, cluster_assets, lookback_weeks=4):
    """
    클러스터 내 Anomaly 탐지

    Args:
        weekly_returns: 주간 수익률 데이터
        cluster_assets: 클러스터에 속한 자산 리스트
        lookback_weeks: 최근 수익률 계산 기간

    Returns:
        anomalies: 특이 움직임을 보이는 자산 리스트 {asset: (pc1_dist, direction)}
    """
    if len(cluster_assets) < 2:
        return {}

    # 클러스터 내 자산들의 최근 수익률
    cluster_returns = weekly_returns[cluster_assets].tail(lookback_weeks)

    # 결측치 제거
    cluster_returns = cluster_returns.dropna(axis=1)

    if cluster_returns.shape[1] < 2:
        return {}

    # PCA 수행
    pca = PCA(n_components=1)
    try:
        pca.fit(cluster_returns.T)
        pc1 = pca.components_[0]  # 첫 번째 주성분
        evr = pca.explained_variance_ratio_[0]  # 설명된 분산 비율
    except:
        return {}

    # Threshold 계산: k = 1 - log(EVR)
    # EVR이 높을수록 밴드가 좁아짐 (움직임이 유사)
    # EVR이 낮을수록 밴드가 넓어짐 (움직임이 상이)
    if evr > 0:
        threshold = 1 - np.log(evr)
    else:
        threshold = 3.0  # 기본값

    # 각 자산의 최근 수익률과 PC1 간의 거리 계산
    anomalies = {}
    latest_returns = cluster_returns.iloc[-1]  # 최근 1주 수익률

    for asset in cluster_returns.columns:
        asset_return = latest_returns[asset]

        # PC1 방향과의 내적으로 예상 수익률 계산
        asset_idx = cluster_returns.columns.get_loc(asset)
        expected_direction = pc1[asset_idx]

        # 실제 수익률과 PC1 방향성 비교
        # PC1 * 평균 수익률로 예상 수익률 추정
        mean_return = cluster_returns.mean(axis=1).iloc[-1]
        expected_return = expected_direction * mean_return

        # 편차 계산
        deviation = asset_return - expected_return
        std_dev = cluster_returns.std(axis=1).iloc[-1]

        if std_dev > 0:
            normalized_deviation = abs(deviation) / std_dev
        else:
            normalized_deviation = 0

        # Threshold 초과 시 anomaly로 간주
        if normalized_deviation > threshold:
            direction = '▲' if deviation > 0 else '▼'
            anomalies[asset] = (normalized_deviation, direction)

    return anomalies

def create_dendrogram_plot(linkage_matrix, labels, title="Asset Dendrogram"):
    """
    Interactive Dendrogram 생성

    Args:
        linkage_matrix: linkage 결과
        labels: 자산 레이블 (이름)
        title: 그래프 제목

    Returns:
        fig: plotly figure
    """
    # Dendrogram 생성
    fig = ff.create_dendrogram(
        linkage_matrix,
        labels=labels,
        orientation='left',
        linkagefun=lambda x: linkage_matrix
    )

    fig.update_layout(
        title=title,
        width=1000,
        height=max(400, len(labels) * 20),  # 자산 수에 따라 높이 조정
        xaxis=dict(title="Distance"),
        yaxis=dict(title="Assets"),
        margin=dict(l=200, r=50, t=100, b=50),
        font=dict(size=10)
    )

    return fig

def create_correlation_heatmap(correlation_matrix, cluster_labels, asset_names):
    """
    클러스터별로 정렬된 상관관계 히트맵 생성

    Args:
        correlation_matrix: 상관관계 행렬
        cluster_labels: 클러스터 레이블
        asset_names: 자산 이름 리스트

    Returns:
        fig: plotly figure
    """
    # 클러스터별로 자산 정렬
    cluster_df = pd.DataFrame({
        'asset': asset_names,
        'cluster': cluster_labels
    })
    cluster_df = cluster_df.sort_values('cluster')
    sorted_assets = cluster_df['asset'].tolist()

    # 정렬된 순서대로 상관관계 행렬 재배열
    sorted_indices = [asset_names.index(asset) for asset in sorted_assets]
    sorted_corr = correlation_matrix.iloc[sorted_indices, sorted_indices]

    # 히트맵 생성
    fig = go.Figure(data=go.Heatmap(
        z=sorted_corr.values,
        x=sorted_assets,
        y=sorted_assets,
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        colorbar=dict(title="Correlation"),
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title="Correlation Matrix (Sorted by Clusters)",
        width=1000,
        height=900,
        xaxis=dict(tickangle=45, tickfont=dict(size=8)),
        yaxis=dict(tickfont=dict(size=8), autorange='reversed'),
        margin=dict(l=150, r=50, t=100, b=150)
    )

    return fig

def main():
    st.title("🌳 Monitoring Universe Dendrogram")
    st.markdown("---")

    # 데이터 로딩
    try:
        prices = load_price_data()
        asset_mapping = load_asset_names()
    except Exception as e:
        st.error(f"데이터 로딩 실패: {str(e)}")
        return

    # 방법론 설명
    with st.expander("📖 Methodology Explanation", expanded=False):
        st.markdown("""
        ## Monitoring Universe Dendrogram - Anomaly Detection

        ### 1. 거리 정의 (Distance Definition)

        두 자산 간 상관관계 ρ ∈ [-1, 1]을 거리 d ∈ [0, 1]로 변환:

        **d = √((1 - ρ) / 2)**

        - 상관관계가 +1에 가까울수록 거리는 0에 가까워짐 (매우 유사)
        - 상관관계가 -1에 가까울수록 거리는 1에 가까워짐 (매우 상이)

        이 거리 함수는 수학적으로 정의된 거리(metric)의 4가지 조건을 모두 만족:
        1. d(x,x) = 0 (자기 자신과의 거리는 0)
        2. d(x,y) > 0, ∀x≠y (서로 다른 점 간 거리는 양수)
        3. d(x,y) = d(y,x) (대칭성)
        4. d(x,z) ≤ d(x,y) + d(y,z) (삼각 부등식)

        ---

        ### 2. Anomaly 정의 (Anomaly Definition)

        **군집 내 특이한 움직임 탐지:**

        1. **군집분석**: 거리 기반으로 계층적 군집분석(Hierarchical Clustering) 수행
        2. **대표 수익률**: 각 군집별로 PCA를 통해 1st Principal Component (PC1) 계산
        3. **밴드 설정**: PC1의 Explained Variance Ratio (EVR)를 활용한 동적 밴드

        **Threshold 계산:**

        **k = 1 - log(EVR)**

        - EVR이 높을수록 (자산 간 움직임이 유사) → 밴드가 좁음 → 특이 움직임 민감
        - EVR이 낮을수록 (자산 간 움직임이 상이) → 밴드가 넓음 → 특이 움직임 관대

        ---

        ### 3. 활용 방법 (Usage)

        **적절한 프로세스:**

        1. **이상치 검색**: 특이하게 움직인 자산 식별
        2. **파급력 진단**: 해당 자산의 시장 영향력 평가
        3. **시나리오 설정**: 주요 자산과의 연관성 점검
        4. **리스크 관리**: 포트폴리오 조정 검토

        ⚠️ **주의**: 특이한 움직임을 '매매 신호'로 직접 활용하는 것은 권장하지 않음

        ---

        ### 4. 해석 가이드

        **군집 특성:**
        - **짧은 거리**: 자산 간 높은 유사성 (EVR 높음)
        - **긴 거리**: 자산 간 낮은 유사성 (EVR 낮음)

        **특이 움직임 (Anomaly):**
        - **▲**: 클러스터 대비 상승
        - **▼**: 클러스터 대비 하락
        """)

    st.markdown("---")

    # 파라미터 설정
    st.subheader("⚙️ Analysis Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        lookback_years = st.selectbox(
            "Correlation Lookback Period",
            options=[1, 3, 5, 10],
            index=3,
            help="상관관계 계산을 위한 과거 데이터 기간 (년)"
        )
        lookback_weeks = lookback_years * 52

    with col2:
        n_clusters = st.slider(
            "Number of Clusters",
            min_value=3,
            max_value=15,
            value=8,
            help="자산을 몇 개의 군집으로 나눌지 설정"
        )

    with col3:
        anomaly_weeks = st.slider(
            "Anomaly Detection Window (weeks)",
            min_value=1,
            max_value=12,
            value=4,
            help="최근 몇 주의 수익률로 특이 움직임을 탐지할지 설정"
        )

    st.markdown("---")

    # 계산 수행
    st.subheader("📊 Analysis Results")

    with st.spinner("Calculating correlation distance and performing clustering..."):
        try:
            # 거리 계산
            distance_matrix, correlation_matrix = calculate_correlation_distance(prices, lookback_weeks)

            # 계층적 군집분석
            linkage_matrix = perform_hierarchical_clustering(distance_matrix.values)

            # 클러스터 할당
            cluster_labels = get_cluster_assignments(linkage_matrix, n_clusters)

            # 자산 이름 리스트
            asset_names = [asset_mapping.get(ticker, ticker) for ticker in prices.columns]

            # Dendrogram 생성
            st.markdown("### 🌳 Asset Dendrogram")
            st.info("계층적 군집분석 결과를 나타내는 Dendrogram입니다. 세로축은 자산, 가로축은 거리를 나타냅니다.")

            dendrogram_fig = create_dendrogram_plot(
                linkage_matrix,
                asset_names,
                f"Asset Dendrogram ({lookback_years} Years Correlation, {n_clusters} Clusters)"
            )
            st.plotly_chart(dendrogram_fig, use_container_width=True)

            st.markdown("---")

            # Anomaly Detection
            st.markdown("### 🔍 Anomaly Detection Results")

            # 주간 수익률 계산
            weekly_prices = prices.resample('W').last()
            weekly_returns = weekly_prices.pct_change().dropna()

            # 클러스터별 Anomaly 탐지
            all_anomalies = {}
            cluster_info = []

            for cluster_id in range(1, n_clusters + 1):
                cluster_mask = cluster_labels == cluster_id
                cluster_assets = prices.columns[cluster_mask].tolist()

                # Anomaly 탐지
                anomalies = detect_anomalies_in_cluster(weekly_returns, cluster_assets, anomaly_weeks)

                if anomalies:
                    all_anomalies[cluster_id] = anomalies

                # 클러스터 정보 수집
                cluster_info.append({
                    'Cluster': f"Cluster {cluster_id}",
                    'Assets': len(cluster_assets),
                    'Anomalies': len(anomalies),
                    'Top Assets': ', '.join([asset_mapping.get(a, a) for a in cluster_assets[:3]])
                })

            # 클러스터 요약 표시
            st.markdown("#### 📋 Cluster Summary")
            cluster_df = pd.DataFrame(cluster_info)
            st.dataframe(cluster_df, use_container_width=True, hide_index=True)

            # Anomaly 결과 표시
            if all_anomalies:
                st.markdown("#### ⚠️ Detected Anomalies")

                anomaly_list = []
                for cluster_id, anomalies in all_anomalies.items():
                    for asset, (deviation, direction) in anomalies.items():
                        anomaly_list.append({
                            'Cluster': f"Cluster {cluster_id}",
                            'Asset': asset_mapping.get(asset, asset),
                            'Ticker': asset,
                            'Direction': direction,
                            'Deviation': f"{deviation:.2f}σ"
                        })

                anomaly_df = pd.DataFrame(anomaly_list)
                anomaly_df = anomaly_df.sort_values('Deviation', ascending=False)

                # 특이 상승/하락 분리 표시
                col_up, col_down = st.columns(2)

                with col_up:
                    st.markdown("**특이 상승(▲) 자산:**")
                    up_anomalies = anomaly_df[anomaly_df['Direction'] == '▲']
                    if not up_anomalies.empty:
                        st.dataframe(up_anomalies, use_container_width=True, hide_index=True)

                        # 자산 이름만 표시
                        up_assets = up_anomalies['Asset'].tolist()
                        st.success(f"**{len(up_assets)}** assets: {up_assets}")
                    else:
                        st.info("No significant upward anomalies detected")

                with col_down:
                    st.markdown("**특이 하락(▼) 자산:**")
                    down_anomalies = anomaly_df[anomaly_df['Direction'] == '▼']
                    if not down_anomalies.empty:
                        st.dataframe(down_anomalies, use_container_width=True, hide_index=True)

                        # 자산 이름만 표시
                        down_assets = down_anomalies['Asset'].tolist()
                        st.warning(f"**{len(down_assets)}** assets: {down_assets}")
                    else:
                        st.info("No significant downward anomalies detected")
            else:
                st.success("✅ No significant anomalies detected in any cluster")

            st.markdown("---")

            # 상관관계 히트맵
            st.markdown("### 🔥 Correlation Heatmap (Cluster-Sorted)")
            st.info("클러스터별로 정렬된 상관관계 행렬입니다. 같은 클러스터의 자산들이 인접하게 배치됩니다.")

            heatmap_fig = create_correlation_heatmap(correlation_matrix, cluster_labels, asset_names)
            st.plotly_chart(heatmap_fig, use_container_width=True)

        except Exception as e:
            st.error(f"분석 중 오류 발생: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

    st.markdown("---")

    # 추가 정보
    with st.expander("ℹ️ Technical Notes", expanded=False):
        st.markdown("""
        ### Technical Implementation Details

        **1. Data Processing**
        - Weekly returns calculated using Friday closing prices
        - Missing data handled via forward-fill method
        - Minimum data requirement: 50% of lookback period

        **2. Distance Calculation**
        - Pearson correlation used for similarity measure
        - Distance transformation ensures metric properties
        - Symmetric and non-negative distance matrix

        **3. Clustering Method**
        - Ward linkage for hierarchical clustering
        - Minimizes within-cluster variance
        - Produces balanced, interpretable clusters

        **4. PCA-based Anomaly Detection**
        - First principal component captures main trend
        - EVR-based dynamic threshold adapts to cluster cohesion
        - Normalized deviation measures relative anomaly strength

        **5. Interpretation Cautions**
        - Short-term anomalies may revert quickly
        - Consider fundamental factors before action
        - Use as risk monitoring, not trading signals
        - Combine with other analyses for validation
        """)

if __name__ == "__main__":
    main()
else:
    main()
