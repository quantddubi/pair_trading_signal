"""
페어트레이딩 분석 도구 - 통합 스크리너 (캐시 적용)
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
cache_utils = import_module_from_file(os.path.join(project_root, "utils/cache_utils.py"), "cache_utils")
common_utils = import_module_from_file(os.path.join(project_root, "utils/common_utils.py"), "common_utils")

# 필요한 함수들
load_data = common_utils.load_data
normalize_prices = common_utils.normalize_prices
calculate_spread = common_utils.calculate_spread
calculate_zscore = common_utils.calculate_zscore

# 페이지 설정
st.set_page_config(
    page_title="통합 스크리너",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 통합 페어트레이딩 스크리너")
st.markdown("---")

def create_correlation_matrix_with_pairs(prices, all_pairs_by_method, asset_mapping):
    """
    전체 자산 상관관계 매트릭스 생성 (방법론별 페어 강조)
    
    Args:
        prices: 가격 데이터
        all_pairs_by_method: 방법론별 선정된 페어들 {method: [pairs]}
        asset_mapping: 자산 이름 매핑
    """
    # 최근 3년 데이터로 상관관계 계산
    end_date = prices.index[-1]
    start_date = end_date - timedelta(days=3*365)  # 3년
    
    recent_data = prices.loc[start_date:end_date].fillna(method='ffill')
    
    # 수익률 계산
    returns = recent_data.pct_change().dropna()
    
    # 상관관계 매트릭스 계산
    correlation_matrix = returns.corr()
    
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
    
    method_names = {
        'euclidean': '유클리드',
        'ssd': 'SSD',
        'cointegration': '공적분',
        'regime': '상관레짐',
        'ou': 'OU',
        'clustering': '클러스터',
        'copula': '코퓰라'
    }
    
    # Plotly 히트맵 생성
    fig = go.Figure()
    
    # 기본 상관관계 히트맵
    fig.add_trace(
        go.Heatmap(
            z=correlation_matrix.values,
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
            text="전체 자산 상관관계 매트릭스 (최근 3년)<br><sub>박스 테두리: 각 방법론별 선정 페어</sub>",
            x=0.5,
            font=dict(size=16)
        ),
        width=800,
        height=700,
        xaxis=dict(
            tickangle=45,
            tickfont=dict(size=8),
            side='bottom'
        ),
        yaxis=dict(
            tickfont=dict(size=8),
            autorange='reversed'  # y축 뒤집기
        ),
        shapes=shapes,
        annotations=annotations,
        margin=dict(l=100, r=100, t=100, b=100)
    )
    
    return fig

def display_correlation_legend(all_pairs_by_method):
    """방법론별 색상 범례를 Streamlit 컬럼으로 표시"""
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
        'euclidean': '유클리드 거리',
        'ssd': 'SSD 거리',
        'cointegration': '공적분',
        'regime': '상관관계 레짐',
        'ou': 'OU 평균회귀',
        'clustering': '클러스터링',
        'copula': '코퓰라 순위상관'
    }
    
    # 페어가 있는 방법론만 수집
    active_methods = []
    for method, pairs in all_pairs_by_method.items():
        if pairs:
            active_methods.append((method, pairs))
    
    if not active_methods:
        st.info("선정된 페어가 없습니다.")
        return
    
    # 범례 제목
    st.markdown("**매트릭스 페어 하이라이트 범례:**")
    
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
                <span style='font-size: 13px; font-weight: bold;'>{name} ({count}개)</span>
            </div>
            """, unsafe_allow_html=True)

# 캐시된 데이터 로딩 함수들
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

def create_simple_pair_chart(prices, asset1, asset2, method_name, signal_info, asset_mapping=None):
    """간단한 페어 차트 생성 (통합 스크리너용)"""
    # 최근 1년 데이터만 사용
    end_date = prices.index[-1]
    start_date = end_date - timedelta(days=365)
    
    chart_data = prices.loc[start_date:end_date, [asset1, asset2]].dropna()
    
    if len(chart_data) == 0:
        st.error(f"데이터가 없습니다: {asset1}, {asset2}")
        return None
    
    # 가격 정규화 (리베이스)
    normalized_data = normalize_prices(chart_data, method='rebase')
    
    # 스프레드 및 Z-스코어 계산 (최근 252일 기준)
    recent_data = chart_data.tail(252)
    normalized_recent = normalize_prices(recent_data, method='rebase')
    spread = calculate_spread(normalized_recent[asset1], normalized_recent[asset2], hedge_ratio=1.0)
    
    # Z-score 계산
    zscore_window = min(60, len(spread)//2) if len(spread) > 20 else max(20, len(spread)//4)
    zscore = calculate_zscore(spread, window=zscore_window)
    
    if len(zscore.dropna()) == 0:
        st.error(f"Z-score 계산 오류")
        return None
    
    # 서브플롯 생성
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.4, 0.3, 0.3],
        subplot_titles=[
            f'{asset1} vs {asset2} - 정규화된 가격 ({method_name})',
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
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=normalized_data.index,
            y=normalized_data[asset2],
            name=asset2,
            line=dict(color='red', width=2)
        ),
        row=1, col=1
    )
    
    # 2. 스프레드 차트
    fig.add_trace(
        go.Scatter(
            x=spread.index,
            y=spread.values,
            name='Spread',
            line=dict(color='green', width=2)
        ),
        row=2, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)
    
    # 3. Z-스코어 차트
    zscore_clean = zscore.dropna()
    fig.add_trace(
        go.Scatter(
            x=zscore_clean.index,
            y=zscore_clean.values,
            name='Z-Score',
            line=dict(color='purple', width=2)
        ),
        row=3, col=1
    )
    
    # Z-스코어 임계값 라인들
    fig.add_hline(y=2.0, line_dash="dash", line_color="orange", opacity=0.7, row=3, col=1)
    fig.add_hline(y=-2.0, line_dash="dash", line_color="orange", opacity=0.7, row=3, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=3, col=1)
    
    # 차트 제목에 자산 이름 포함
    if asset_mapping:
        name1 = asset_mapping.get(asset1, asset1)
        name2 = asset_mapping.get(asset2, asset2)
        chart_title = f"{method_name}: {name1}({asset1}) - {name2}({asset2})"
    else:
        chart_title = f"{method_name}: {asset1} - {asset2}"
    
    # 레이아웃 설정
    fig.update_layout(
        height=700,
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
    
    return fig

def main():
    st.info("🚀 모든 방법론의 사전 계산된 결과를 통합하여 표시합니다.")
    
    # 필요한 데이터 로딩
    try:
        prices = load_price_data()
        asset_mapping = load_asset_names()
    except Exception as e:
        st.error(f"데이터 로딩 실패: {str(e)}")
        return
    
    # 모든 방법론의 캐시 정보 가져오기
    cache_info = cache_utils.get_cache_info()
    
    # SSD 캐시 없음 알림
    if not cache_info.get('ssd', {}).get('exists', False):
        st.warning("⚠️ SSD 거리 방법론의 캐시가 없습니다. `python generate_cache.py`를 실행하여 전체 캐시를 생성하세요.")
    
    # 캐시 상태 표시
    st.subheader("방법론별 캐시 상태")
    cols = st.columns(7)
    methods = ['euclidean', 'ssd', 'cointegration', 'regime', 'ou', 'clustering', 'copula']
    method_names = ['유클리드 거리', 'SSD 거리', '공적분', '상관관계 레짐', 'OU 평균회귀', '클러스터링', '코퓰라 순위상관']
    method_icons = ['📐', '📊', '🔗', '📈', '🔄', '🧠', '🎲']
    
    for i, (method, name, icon) in enumerate(zip(methods, method_names, method_icons)):
        with cols[i]:
            if cache_info.get(method, {}).get('exists', False):
                st.success(f"✅ {icon} {name}")
                info = cache_info[method]
                st.caption(f"진입: {info['enter_count']}개")
                st.caption(f"관찰: {info['watch_count']}개")
            else:
                st.error(f"❌ {icon} {name}")
    
    # 방법론별 특징 설명
    with st.expander("방법론별 비교 특징", expanded=False):
        st.markdown("""
        | 방법론 | 핵심 특징 | 장점 | 적합한 상황 |
        |--------|-----------|------|-------------|
        | 📐 **유클리드 거리** | 가격 경로 유사성 | 계산 빠름, 직관적 | 단순하고 안정적인 페어 |
        | 📊 **SSD 거리** | 누적수익률 제곱편차 | 실무적, 정교한 측정 | 배당 고려한 실전 트레이딩 |
        | 🔗 **공적분** | 장기 균형관계 | 이론적 근거 강함 | 장기 투자, 펀더멘털 연관 |
        | 📈 **상관관계 레짐** | 동적 상관관계 변화 | 시장 환경 적응 | 변동성 큰 시장 |
        | 🔄 **OU 평균회귀** | 평균회귀 속도 최적화 | 수학적 정교함 | 안정적 평균회귀 |
        | 🧠 **클러스터링** | 다차원 특성 그룹화 | 펀더멘털 고려 | 복합적 특성 분석 |
        | 🎲 **코퓰라 순위상관** | 비선형 꼬리의존성 | 극단위험 고려 | 위기시 동조현상 포착 |
        """)
    
    st.markdown("---")
    
    # 전체 자산 상관관계 매트릭스 (방법론별 페어 하이라이트)
    st.subheader("전체 자산 상관관계 매트릭스 (최근 3년)")
    
    # 방법론별 진입 페어 수집
    method_pairs = {}
    for method in methods:
        cache_data = cache_utils.load_cache(method)
        if cache_data:
            enter_signals = cache_data.get('enter_signals', [])
            method_pairs[method] = [signal['pair'] for signal in enter_signals]
        else:
            method_pairs[method] = []
    
    # 상관관계 매트릭스 생성 및 표시
    try:
        with st.spinner("상관관계 매트릭스 생성 중..."):
            correlation_fig = create_correlation_matrix_with_pairs(prices, method_pairs, asset_mapping)
            st.plotly_chart(correlation_fig, use_container_width=True)
            
            # 방법론별 색상 범례 표시
            display_correlation_legend(method_pairs)
            
            st.info("💡 **매트릭스 해석 가이드:**\n"
                   "- 색상이 진할수록 높은 상관관계 (빨강: 양의 상관, 파랑: 음의 상관)\n"
                   "- 색칠된 테두리 박스: 각 방법론에서 선정된 진입 페어\n"
                   "- 마우스 오버: 두 자산 간 정확한 상관계수 확인 가능")
    except Exception as e:
        st.error(f"상관관계 매트릭스 생성 중 오류 발생: {str(e)}")
    
    st.markdown("---")
    
    # 방법론별 페어 선정 현황
    st.subheader("방법론별 페어 선정 현황")
    
    # 방법론별 페어 리스트 표시
    method_cols = st.columns(len(methods))
    for i, (method, name, icon) in enumerate(zip(methods, method_names, method_icons)):
        with method_cols[i]:
            st.markdown(f"**{icon} {name}**")
            pairs = method_pairs[method]
            if pairs:
                for pair in pairs[:5]:  # 최대 5개만 표시
                    formatted_pair = format_pair_name(pair, asset_mapping)
                    st.caption(f"• {formatted_pair}")
                if len(pairs) > 5:
                    st.caption(f"... (+{len(pairs)-5}개)")
            else:
                if method == 'ssd':
                    st.caption("캐시 없음 (생성 필요)")
                else:
                    st.caption("선정 페어 없음")
    
    st.markdown("---")
    
    # 통합 결과 표시
    all_enter_signals = []
    all_watch_signals = []
    
    for method in methods:
        cache_data = cache_utils.load_cache(method)
        if cache_data:
            enter_signals = cache_data.get('enter_signals', [])
            watch_signals = cache_data.get('watch_signals', [])
            all_enter_signals.extend(enter_signals)
            all_watch_signals.extend(watch_signals)
    
    # 합의 기반 분석
    st.subheader("합의 기반 진입 신호")
    if all_enter_signals:
        # 페어별로 그룹화
        pair_counts = {}
        for signal in all_enter_signals:
            pair = signal['pair']
            if pair not in pair_counts:
                pair_counts[pair] = []
            pair_counts[pair].append(signal)
        
        # 여러 방법론에서 선정된 페어 우선 표시
        consensus_pairs = [(pair, signals) for pair, signals in pair_counts.items() if len(signals) >= 2]
        single_method_pairs = [(pair, signals) for pair, signals in pair_counts.items() if len(signals) == 1]
        
        if consensus_pairs:
            st.info(f"{len(consensus_pairs)}개 페어가 여러 방법론에서 선정되었습니다.")
            
            for pair, signals in consensus_pairs:
                with st.expander(f"{pair} (합의도: {len(signals)}개 방법론)", expanded=True):
                    cols = st.columns(len(signals))
                    for i, signal in enumerate(signals):
                        with cols[i]:
                            method = signal.get('method', 'unknown')
                            method_idx = methods.index(method) if method in methods else -1
                            icon = method_icons[method_idx] if method_idx >= 0 else "❓"
                            name = method_names[method_idx] if method_idx >= 0 else method
                            
                            st.write(f"**{icon} {name}**")
                            st.write(f"**Z-Score:** {signal['current_zscore']:.2f}")
                            st.write(f"**방향:** {signal['direction']}")
                            if 'half_life' in signal:
                                st.write(f"**반감기:** {signal['half_life']:.1f}일")
                            if 'cost_ratio' in signal:
                                st.write(f"**비용비율:** {signal['cost_ratio']:.1f}")
        
        if single_method_pairs:
            st.subheader("단일 방법론 진입 신호")
            for pair, signals in single_method_pairs[:5]:  # 최대 5개만 표시
                signal = signals[0]
                method = signal.get('method', 'unknown')
                method_idx = methods.index(method) if method in methods else -1
                icon = method_icons[method_idx] if method_idx >= 0 else "❓"
                name = method_names[method_idx] if method_idx >= 0 else method
                
                # 페어 이름 포맷팅
                formatted_pair = format_pair_name(pair, asset_mapping)
                
                with st.expander(f"{formatted_pair} ({icon} {name})"):
                    # 기본 정보 표시
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("진입 방향", signal['direction'])
                    with col2:
                        st.metric("Z-Score", f"{signal['current_zscore']:.2f}")
                    with col3:
                        if 'half_life' in signal:
                            st.metric("반감기", f"{signal['half_life']:.1f}일")
                    with col4:
                        if 'cost_ratio' in signal:
                            st.metric("비용비율", f"{signal['cost_ratio']:.1f}")
                    
                    # 차트 생성 및 표시
                    asset1, asset2 = pair.split('-')
                    
                    try:
                        with st.spinner(f"{formatted_pair} 차트 생성 중..."):
                            fig = create_simple_pair_chart(
                                prices, asset1, asset2, name, signal, asset_mapping
                            )
                            
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # 차트 설명
                                st.info(f"""
                                **{name} 방법론 차트 설명:**
                                - 상단: 두 자산의 정규화된 가격 추이 (최근 1년)
                                - 중간: 스프레드 (가격 차이)
                                - 하단: Z-스코어 ({name} 기반 신호)
                                - 주황색 선: 진입 임계값 (±2.0)
                                - 현재 Z-Score: {signal['current_zscore']:.2f}
                                - 진입 방향: {signal['direction']}
                                """)
                            else:
                                st.warning("차트를 생성할 수 없습니다.")
                    
                    except Exception as e:
                        st.error(f"차트 생성 중 오류: {str(e)}")
    else:
        st.info("현재 진입 신호가 없습니다.")
    
    # 관찰 대상
    st.subheader("관찰 대상 (모든 방법론)")
    if all_watch_signals:
        # 최대 10개까지만 표시
        for i, signal in enumerate(all_watch_signals[:10], 1):
            method = signal.get('method', 'unknown')
            method_idx = methods.index(method) if method in methods else -1
            icon = method_icons[method_idx] if method_idx >= 0 else "❓"
            name = method_names[method_idx] if method_idx >= 0 else method
            
            # 페어 이름 포맷팅
            formatted_pair = format_pair_name(signal['pair'], asset_mapping)
            
            with st.expander(f"{i}. {formatted_pair} ({icon} {name}) - Z-Score: {signal['current_zscore']:.2f}"):
                # 기본 정보 표시
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if 'direction' in signal:
                        st.metric("예상 방향", signal['direction'])
                    else:
                        st.metric("상태", "관찰 중")
                with col2:
                    st.metric("Z-Score", f"{signal['current_zscore']:.2f}")
                with col3:
                    if 'half_life' in signal:
                        st.metric("반감기", f"{signal['half_life']:.1f}일")
                with col4:
                    if 'cost_ratio' in signal:
                        st.metric("비용비율", f"{signal['cost_ratio']:.1f}")
                
                # 차트 생성 및 표시
                asset1, asset2 = signal['pair'].split('-')
                
                try:
                    with st.spinner(f"{formatted_pair} 차트 생성 중..."):
                        fig = create_simple_pair_chart(
                            prices, asset1, asset2, name, signal, asset_mapping
                        )
                        
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # 차트 설명
                            st.info(f"""
                            **{name} 방법론 관찰 차트 설명:**
                            - 상단: 두 자산의 정규화된 가격 추이 (최근 1년)
                            - 중간: 스프레드 (가격 차이)
                            - 하단: Z-스코어 ({name} 기반 신호)
                            - 주황색 선: 진입 임계값 (±2.0)
                            - 현재 Z-Score: {signal['current_zscore']:.2f} (진입 대기 중)
                            - 상태: 진입 조건 근접, 지속적 관찰 필요
                            """)
                        else:
                            st.warning("차트를 생성할 수 없습니다.")
                
                except Exception as e:
                    st.error(f"차트 생성 중 오류: {str(e)}")
    else:
        st.info("현재 관찰 대상이 없습니다.")
    
    # 요약 통계
    st.markdown("---")
    st.subheader("분석 요약")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("총 진입 신호", len(all_enter_signals))
    with col2:
        st.metric("총 관찰 대상", len(all_watch_signals))
    with col3:
        consensus_count = len([pair for pair, signals in pair_counts.items() if len(signals) >= 2]) if all_enter_signals else 0
        st.metric("합의 페어", consensus_count)
    with col4:
        active_methods = sum(1 for method in methods if cache_info.get(method, {}).get('exists', False))
        st.metric("활성 방법론", f"{active_methods}/6")

if __name__ == "__main__":
    main()
else:
    main()