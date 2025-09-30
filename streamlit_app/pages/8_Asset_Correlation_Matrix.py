"""
Asset Correlation Matrix - Interactive correlation analysis with methodology pair highlights
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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
cache_utils = import_module_from_file(os.path.join(project_root, "utils/cache_utils.py"), "cache_utils")
common_utils = import_module_from_file(os.path.join(project_root, "utils/common_utils.py"), "common_utils")

# 필요한 함수들
load_data = common_utils.load_data

# 페이지 설정
st.set_page_config(
    page_title="Asset Correlation Matrix",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_asset_categories():
    """Classify assets by categories"""
    return {
        'Major Stock Indices': ['SPX Index', 'RTY Index', 'NDX Index', 'INDU Index', 'MXWD Index', 'MXWO Index', 'MXEA Index'],
        'US Sectors': ['S5FINL Index', 'S5INFT Index', 'S5HLTH Index', 'S5TELS Index', 'S5INDU Index', 'S5COND Index', 'S5CONS Index', 'S5ENRS Index', 'S5MATR Index', 'S5RLST Index', 'S5UTIL Index'],
        'Europe/Asia Stocks': ['SX5E Index', 'SXXP Index', 'UKX Index', 'NKY Index', 'TPX Index', 'DAX Index', 'CAC Index', 'AEX Index', 'FTSEMIB Index', 'IBEX Index'],
        'Emerging Market Stocks': ['MXEF Index', 'XIN9I Index', 'HSI Index', 'KOSPI Index', 'TWSE Index', 'SENSEX Index', 'MXLA Index'],
        'Bonds': ['WN1 Comdty', 'UXY1 Comdty', 'TY1 Comdty', 'FV1 Comdty', 'TU1 Comdty', 'CN1 Comdty', 'UB1 Comdty', 'RX1 Comdty', 'OE1 Comdty', 'DU1 Comdty', 'OAT1 Comdty', 'G 1 Comdty'],
        'Bond Indices': ['LEGATRUH Index', 'LGTRTRUH Index', 'H03454US Index', 'H03450US Index', 'LGCPTRUH Index', 'LG30TRUH Index', 'LG20TRUH Index', 'LUATTRUU Index'],
        'Major Currencies': ['EURUSD Curncy', 'JPYUSD Curncy', 'GBPUSD Curncy', 'CHFUSD Curncy', 'AD1 Curncy', 'CADUSD Curncy'],
        'Emerging Currencies': ['CNYUSD Curncy', 'BRLUSD Curncy', 'MXNUSD Curncy', 'KRWUSD Curncy', 'SEKUSD Curncy', 'NZDUSD Curncy'],
        'Commodities': ['CL1 Comdty', 'HG1 Comdty', 'NG1 Comdty', 'GC1 Comdty', 'SI1 Comdty', 'PL1 Comdty', 'SCO1 Comdty', 'PA1 Comdty'],
        'Agriculture': ['C 1 Comdty', 'W 1 Comdty', 'S 1 Comdty'],
        'Others': ['VIX Index']
    }

def create_correlation_matrix_with_pairs(prices, all_pairs_by_method, asset_mapping, selected_category='전체', category_assets=None, time_period='3년'):
    """
    자산 상관관계 매트릭스 생성 (방법론별 페어 강조)
    
    Args:
        prices: 가격 데이터
        all_pairs_by_method: 방법론별 선정된 페어들 {method: [pairs]}
        asset_mapping: 자산 이름 매핑
        selected_category: 선택된 카테고리 이름
        category_assets: 표시할 자산 리스트 (None이면 전체)
        time_period: 시간 범위 ('1년', '3년', '5년', '10년', '전체')
    """
    # 시간 범위별 날짜 계산
    end_date = prices.index[-1]
    
    time_mapping = {
        '1년': 365,
        '3년': 3 * 365,
        '5년': 5 * 365,
        '10년': 10 * 365,
        '전체': None
    }
    
    if time_period == '전체':
        start_date = prices.index[0]
        period_days = (end_date - start_date).days
    else:
        period_days = time_mapping.get(time_period, 3 * 365)
        start_date = end_date - timedelta(days=period_days)
    
    recent_data = prices.loc[start_date:end_date]
    
    # 데이터 충분성 확인 (최소 50%의 데이터 필요)
    min_data_points = len(recent_data) * 0.5
    sufficient_assets = []
    
    for col in recent_data.columns:
        non_null_count = recent_data[col].notna().sum()
        if non_null_count >= min_data_points:
            sufficient_assets.append(col)
    
    if len(sufficient_assets) < 2:
        st.error(f"{time_period} 기간에 충분한 데이터가 있는 자산이 부족합니다.")
        return None, []
    
    # 카테고리별 자산 필터링 (충분한 데이터가 있는 자산만)
    if category_assets is not None:
        # 선택된 카테고리이면서 충분한 데이터가 있는 자산만 사용
        available_assets = [asset for asset in category_assets if asset in sufficient_assets]
        if len(available_assets) < 2:
            st.error(f"{selected_category} 카테고리에서 {time_period} 기간에 충분한 데이터가 있는 자산이 부족합니다.")
            return None, []
        recent_data = recent_data[available_assets]
    else:
        # 전체 자산 중 충분한 데이터가 있는 자산만 사용
        recent_data = recent_data[sufficient_assets]
    
    # 결측치 처리
    recent_data = recent_data.fillna(method='ffill')
    
    # 수익률 계산
    returns = recent_data.pct_change().dropna()
    
    # 상관관계 매트릭스 계산
    correlation_matrix = returns.corr()
    
    # 상삼각형을 NaN으로 만들어서 하삼각형만 표시 (대각선 포함)
    import numpy as np
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
    correlation_matrix_lower = correlation_matrix.copy()
    correlation_matrix_lower[mask] = np.nan
    
    # 자산 티커와 이름 가져오기
    tickers = correlation_matrix.columns.tolist()
    n_assets = len(tickers)
    
    # 티커를 자산 이름으로 매핑
    display_names = [asset_mapping.get(ticker, ticker) for ticker in tickers]
    
    # 방법론별 색상 정의
    method_colors = {
        'euclidean': '#FF6B6B',      # 빨간색
        'ssd': '#4ECDC4',            # 청록색
        'cointegration': '#45B7D1',  # 파란색
        'regime': '#FFA07A',         # 연어색
        'ou': '#98D8C8',             # 민트색
        'clustering': '#F7DC6F',     # 노란색
        'copula': '#BB8FCE'          # 보라색
    }
    
    # Plotly 히트맵 생성
    fig = go.Figure()
    
    # 기본 상관관계 히트맵 (하삼각형만)
    fig.add_trace(
        go.Heatmap(
            z=correlation_matrix_lower.values,
            x=[name[:12] + '...' if len(name) > 15 else name for name in display_names],  # 자산 이름으로 표시
            y=[name[:12] + '...' if len(name) > 15 else name for name in display_names],
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar=dict(
                title=dict(text="상관계수"),
                tickmode="linear",
                tick0=-1,
                dtick=0.5
            ),
            hovertemplate='<b>%{y}</b> vs <b>%{x}</b><br>상관계수: %{z:.3f}<extra></extra>',
            showscale=True
        )
    )
    
    # 방법론별 페어에 테두리 박스 추가
    shapes = []
    annotations = []
    
    for method, pairs in all_pairs_by_method.items():
        if not pairs:
            continue
            
        color = method_colors.get(method, '#000000')
        
        for pair in pairs[:10]:  # 상위 10개만 표시 (너무 많으면 복잡)
            try:
                asset1, asset2 = pair.split('-')
                if asset1 in tickers and asset2 in tickers:
                    i = tickers.index(asset1)
                    j = tickers.index(asset2)
                    
                    # 대칭 위치에 박스 추가
                    for x, y in [(i, j), (j, i)]:
                        shapes.append(
                            dict(
                                type="rect",
                                x0=x-0.4, y0=y-0.4,
                                x1=x+0.4, y1=y+0.4,
                                line=dict(color=color, width=3),
                                fillcolor="rgba(0,0,0,0)"  # 투명 배경
                            )
                        )
            except:
                continue
    
    # 레이아웃 설정
    fig.update_layout(
        title=dict(
            text=f"{selected_category} Asset Correlation Matrix ({time_period})<br><sub>Assets: {len(display_names)}</sub>",
            x=0.5,
            font=dict(size=18)
        ),
        width=1200,  # 크기 증가
        height=1000,  # 크기 증가
        xaxis=dict(
            tickangle=45,
            tickfont=dict(size=10),  # 폰트 크기 증가
            side='bottom'
        ),
        yaxis=dict(
            tickfont=dict(size=10),  # 폰트 크기 증가
            autorange='reversed'  # y축 뒤집기
        ),
        shapes=shapes,
        annotations=annotations,
        margin=dict(l=100, r=100, t=100, b=100)
    )
    
    return fig, tickers

def display_correlation_legend(all_pairs_by_method, displayed_assets=None):
    """방법론별 색상 범례를 Streamlit 컬럼으로 표시 (실제 매트릭스에 표시된 페어만)"""
    method_colors = {
        'euclidean': '#FF6B6B',
        'ssd': '#4ECDC4', 
        'cointegration': '#45B7D1',
        'regime': '#FFA07A',
        'ou': '#98D8C8',
        'clustering': '#F7DC6F',
        'copula': '#BB8FCE'
    }
    
    method_names = {
        'euclidean': 'Euclidean Distance',
        'ssd': 'SSD Distance',
        'cointegration': 'Cointegration',
        'regime': 'Correlation Regime',
        'ou': 'OU Mean Reversion',
        'clustering': 'Clustering',
        'copula': 'Copula Rank Correlation'
    }
    
    # 실제 매트릭스에 표시된 페어가 있는 방법론만 수집
    active_methods = []
    for method, pairs in all_pairs_by_method.items():
        if pairs:
            # displayed_assets가 제공된 경우, 해당 자산들로 구성된 페어만 필터링
            if displayed_assets is not None:
                visible_pairs = []
                for pair in pairs:
                    try:
                        asset1, asset2 = pair.split('-')
                        if asset1 in displayed_assets and asset2 in displayed_assets:
                            visible_pairs.append(pair)
                    except:
                        continue
                
                if visible_pairs:  # 실제로 표시되는 페어가 있는 경우만
                    active_methods.append((method, visible_pairs))
            else:
                active_methods.append((method, pairs))
    
    if not active_methods:
        st.info("선정된 페어가 없습니다.")
        return
    
    # 범례 제목
    st.markdown("**Matrix Pair Highlight Legend:**")
    
    # 컬럼으로 범례 표시
    cols = st.columns(min(len(active_methods), 4))  # 최대 4개 컬럼
    
    for i, (method, pairs) in enumerate(active_methods):
        col_idx = i % len(cols)
        with cols[col_idx]:
            color = method_colors.get(method, '#000000')
            name = method_names.get(method, method)
            count = len(pairs)
            
            # 색상 박스와 텍스트를 HTML로 표시
            st.markdown(f"""
            <div style='display: flex; align-items: center; gap: 8px; margin: 5px 0;'>
                <div style='width: 16px; height: 16px; border: 2px solid {color}; background: transparent; flex-shrink: 0;'></div>
                <span style='font-size: 13px; font-weight: bold;'>{name} ({count})</span>
            </div>
            """, unsafe_allow_html=True)

# 데이터 로딩 함수들 (캐시 없음 - 항상 최신 데이터 로드)
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

def main():
    st.title("🔍 Asset Correlation Matrix")
    st.markdown("---")
    
    # 데이터 로딩
    try:
        prices = load_price_data()
        asset_mapping = load_asset_names()
    except Exception as e:
        st.error(f"데이터 로딩 실패: {str(e)}")
        return
    
    # 방법론별 진입 페어 수집
    methods = ['euclidean', 'ssd', 'cointegration', 'regime', 'ou', 'clustering', 'copula']
    method_pairs = {}
    for method in methods:
        cache_data = cache_utils.load_cache(method)
        if cache_data:
            enter_signals = cache_data.get('enter_signals', [])
            method_pairs[method] = [signal['pair'] for signal in enter_signals]
        else:
            method_pairs[method] = []
    
    # 페이지 설명
    st.info("""
    **Asset Correlation Matrix Analysis**
    
    이 페이지에서는 전체 자산 간의 상관관계를 시각적으로 분석할 수 있습니다.
    - 각 방법론에서 선정된 페어들이 색상별로 하이라이트됩니다
    - 카테고리별로 자산을 필터링하여 분석할 수 있습니다
    - 다양한 시간 범위로 상관관계 변화를 관찰할 수 있습니다
    """)
    
    st.markdown("---")
    
    # 카테고리 선택 UI
    st.subheader("📊 Asset Category & Period Selection")
    categories = get_asset_categories()
    
    # 카테고리 및 시간 범위 선택
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.markdown("**Asset Category:**")
        category_options = ['전체'] + list(categories.keys())
        selected_category = st.selectbox(
            "Category",
            category_options,
            index=0,
            help="특정 자산군만 선택하여 더 명확한 매트릭스를 확인할 수 있습니다"
        )
    
    with col2:
        st.markdown("**Analysis Period:**")
        time_options = ['1년', '3년', '5년', '10년', '전체']
        selected_time_period = st.selectbox(
            "Period",
            time_options,
            index=1,  # 기본값: 3년
        )
    
    with col3:
        if selected_category == '전체':
            # 카테고리별 자산 수 요약
            with st.expander("📋 Category Asset Count Summary", expanded=False):
                summary_cols = st.columns(3)
                for i, (cat_name, cat_assets) in enumerate(categories.items()):
                    col_idx = i % 3
                    with summary_cols[col_idx]:
                        st.metric(cat_name, f"{len(cat_assets)} assets")
        else:
            category_assets = categories[selected_category]
            st.success(f"**{selected_category}** category: {len(category_assets)} assets, **{selected_time_period}** period")
            
            # 선택된 카테고리의 자산 목록 표시
            with st.expander(f"📋 {selected_category} Asset List", expanded=False):
                asset_names = [f"{asset} ({asset_mapping.get(asset, asset)})" for asset in category_assets if asset in prices.columns]
                if asset_names:
                    st.write(", ".join(asset_names))
                    st.caption(f"💡 Assets with insufficient data in the {selected_time_period} period will be automatically excluded.")
                else:
                    st.warning("해당 카테고리에 사용 가능한 자산이 없습니다.")
    
    st.markdown("---")
    
    # 상관관계 매트릭스 생성 및 표시
    st.subheader("📈 Interactive Correlation Matrix")
    
    try:
        with st.spinner(f"Generating {selected_category} correlation matrix..."):
            if selected_category == '전체':
                result = create_correlation_matrix_with_pairs(
                    prices, method_pairs, asset_mapping, '전체', None, selected_time_period
                )
            else:
                result = create_correlation_matrix_with_pairs(
                    prices, method_pairs, asset_mapping, selected_category, categories[selected_category], selected_time_period
                )
            
            if result and len(result) == 2:
                correlation_fig, displayed_tickers = result
                st.plotly_chart(correlation_fig, use_container_width=True)
                
                # 실제 매트릭스에 표시된 자산들만 고려하여 범례 표시
                display_correlation_legend(method_pairs, displayed_tickers)
                
                # 매트릭스 해석 가이드
                with st.expander("📖 Matrix Interpretation Guide", expanded=False):
                    st.markdown("""
                    **🔍 How to Read the Correlation Matrix:**
                    
                    **Correlation Values:**
                    - **+1.0**: Perfect positive correlation (move together)
                    - **0.0**: No correlation (independent movement) 
                    - **-1.0**: Perfect negative correlation (move opposite)
                    
                    **Color Coding:**
                    - **Red**: Positive correlation
                    - **Blue**: Negative correlation
                    - **White**: Near zero correlation
                    
                    **Methodology Pair Highlights:**
                    - **Colored boxes**: Pairs selected by different methodologies
                    - **Multiple colors**: Pairs selected by multiple methods
                    - **Box intensity**: Shows methodology confidence
                    
                    **Analysis Tips:**
                    - Look for high correlation clusters
                    - Identify diversification opportunities (low/negative correlations)
                    - Compare methodology selections with actual correlations
                    - Use different time periods to see correlation stability
                    """)

    except Exception as e:
        st.error(f"상관관계 매트릭스 생성 중 오류 발생: {str(e)}")
    
    st.markdown("---")
    
    # 추가 분석 도구
    st.subheader("📊 Additional Analysis Tools")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔄 Quick Analysis")
        if st.button("Show High Correlation Pairs (>0.8)", type="secondary"):
            try:
                returns = prices.pct_change().dropna()
                corr_matrix = returns.corr()
                
                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.8:
                            asset1 = corr_matrix.columns[i]
                            asset2 = corr_matrix.columns[j]
                            high_corr_pairs.append({
                                'Asset 1': asset_mapping.get(asset1, asset1),
                                'Asset 2': asset_mapping.get(asset2, asset2), 
                                'Correlation': f"{corr_val:.3f}"
                            })
                
                if high_corr_pairs:
                    df_high_corr = pd.DataFrame(high_corr_pairs)
                    st.dataframe(df_high_corr, use_container_width=True)
                else:
                    st.info("No pairs with correlation > 0.8 found")
            except Exception as e:
                st.error(f"Analysis error: {str(e)}")
    
    with col2:
        st.markdown("#### 📈 Statistics Summary")
        try:
            returns = prices.pct_change().dropna()
            corr_matrix = returns.corr()
            
            # 상관계수 통계
            corr_values = corr_matrix.values
            corr_values = corr_values[~np.eye(corr_values.shape[0], dtype=bool)]  # 대각선 제거
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Average Correlation", f"{np.nanmean(corr_values):.3f}")
                st.metric("Max Correlation", f"{np.nanmax(corr_values):.3f}")
            with col_b:
                st.metric("Min Correlation", f"{np.nanmin(corr_values):.3f}")
                st.metric("Std Correlation", f"{np.nanstd(corr_values):.3f}")
            
        except Exception as e:
            st.error(f"Statistics calculation error: {str(e)}")

if __name__ == "__main__":
    main()
else:
    main()