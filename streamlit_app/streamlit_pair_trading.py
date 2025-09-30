"""
Pair Trading Analysis Tool - Integrated Screener (Cache Applied)
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
sys.path.insert(0, current_dir)
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
    page_title="Integrated Screener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Find pairs")
st.markdown("---")


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
    """페어 이름을 이름만 표시하는 형태로 포맷팅"""
    asset1, asset2 = pair.split('-')
    
    name1 = asset_mapping.get(asset1, asset1)
    name2 = asset_mapping.get(asset2, asset2)
    
    return f"{name1} - {name2}"

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
    st.subheader("Cache Status by Methodology")
    cols = st.columns(7)
    methods = ['euclidean', 'ssd', 'cointegration', 'regime', 'ou', 'clustering', 'copula']
    method_names = ['Euclidean Distance', 'SSD Distance', 'Cointegration', 'Correlation Regime', 'OU Mean Reversion', 'Clustering', 'Copula Rank Correlation']
    method_icons = ['📐', '📊', '🔗', '📈', '🔄', '🧠', '🎲']
    
    for i, (method, name, icon) in enumerate(zip(methods, method_names, method_icons)):
        with cols[i]:
            if cache_info.get(method, {}).get('exists', False):
                st.success(f"✅ {icon} {name}")
                info = cache_info[method]
                st.caption(f"Entry: {info['enter_count']}")
                st.caption(f"Watch: {info['watch_count']}")
            else:
                st.error(f"❌ {icon} {name}")
    
    # 방법론별 특징 설명
    with st.expander("방법론별 비교 특징", expanded=False):
        st.markdown("""
        | 방법론 | 핵심 특징 | 공식 | 장점 | 적합한 상황 |
        |--------|-----------|------|------|-------------|
        | **유클리드 거리** | 정규화 가격벡터 간 L2 거리 최소화 | d = √Σ(P₁ᵢ-P₂ᵢ)² | 계산 단순, 노이즈에 견고 | 유사한 가격 패턴을 보이는 안정적 페어 |
        | **SSD 거리** | 누적수익률 제곱편차의 합 | SSD = Σ(r₁ᵢ-r₂ᵢ)² | 수익률 기반, 배당 반영 | 수익률 동조성이 높고 실전 거래 적합 |
        | **공적분** | 단위근 검정으로 장기균형 관계 | P₁ - βP₂ = I(0) | 이론적 근거 확실 | 펀더멘털 연관성 강한 장기 페어 |
        | **상관관계 레짐** | DCC-GARCH로 동적 상관계수 | ρₜ = corr(r₁ₜ,r₂ₜ\|Ωₜ₋₁) | 시변 상관관계 포착 | 상관관계가 시간에 따라 변하는 페어 |
        | **OU 평균회귀** | 확률미분방정식, 반감기 | θ(μ-Xₜ)dt, HL = ln(2)/θ | 평균회귀 속도 정량화 | 명확한 평균회귀 특성을 가진 스프레드 |
        | **클러스터링** | K-means로 다차원 거리공간 | min Σ∥xᵢ-cⱼ∥² | 다중 특성 동시 고려 | 가격외 펀더멘털 지표가 유사한 페어 |
        | **코퓰라 순위상관** | 켄달타우 τ와 꼬리의존성 | C(u,v) = P(U≤u,V≤v) | 비선형 의존구조 포착 | 극단상황에서 동반 급락/급등 페어 |
        """)
    
    st.markdown("---")
    
    # 방법론별 페어 선정 현황
    st.subheader("방법론별 페어 선정 현황")
    
    # 방법론별 진입 페어 수집
    method_pairs = {}
    for method in methods:
        cache_data = cache_utils.load_cache(method)
        if cache_data:
            enter_signals = cache_data.get('enter_signals', [])
            method_pairs[method] = [signal['pair'] for signal in enter_signals]
        else:
            method_pairs[method] = []
    
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
    
    # 다중 필터 통과 분석
    st.subheader("다중 필터 통과 진입 신호")
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
                # 페어 이름 포맷팅
                formatted_pair = format_pair_name(pair, asset_mapping)
                
                with st.expander(f"{formatted_pair} (통과: {len(signals)}개 방법론)", expanded=True):
                    # 각 방법론 신호 정보 표시
                    cols = st.columns(len(signals))
                    for i, signal in enumerate(signals):
                        with cols[i]:
                            method = signal.get('method', 'unknown')
                            method_idx = methods.index(method) if method in methods else -1
                            icon = method_icons[method_idx] if method_idx >= 0 else "❓"
                            name = method_names[method_idx] if method_idx >= 0 else method
                            
                            st.write(f"**{icon} {name}**")
                            if 'current_zscore' in signal:
                                st.write(f"**Z-Score:** {signal['current_zscore']:.2f}")
                            elif 'current_deviation' in signal:
                                st.write(f"**편차:** {signal['current_deviation']:.2f}σ")
                            st.write(f"**방향:** {signal['direction']}")
                            if 'half_life' in signal:
                                st.write(f"**반감기:** {signal['half_life']:.1f}일")
                            if 'cost_ratio' in signal:
                                st.write(f"**비용비율:** {signal['cost_ratio']:.1f}")
                    
                    # 차트 생성 및 표시 (첫 번째 신호를 기준으로)
                    asset1, asset2 = pair.split('-')
                    primary_signal = signals[0]  # 첫 번째 신호를 기준으로
                    primary_method = primary_signal.get('method', 'unknown')
                    primary_method_idx = methods.index(primary_method) if primary_method in methods else -1
                    primary_name = method_names[primary_method_idx] if primary_method_idx >= 0 else primary_method
                    
                    try:
                        with st.spinner(f"{formatted_pair} 차트 생성 중..."):
                            fig = create_simple_pair_chart(
                                prices, asset1, asset2, primary_name, primary_signal, asset_mapping
                            )
                            
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # 차트 설명
                                st.info(f"""
                                **다중 필터 통과 페어 차트 설명:**
                                - 상단: 두 자산의 정규화된 가격 추이 (최근 1년)
                                - 중간: 스프레드 (가격 차이)
                                - 하단: Z-스코어 ({primary_name} 기반 차트)
                                - 주황색 선: 진입 임계값 (±2.0)
                                - 현재 Z-Score: {primary_signal.get('current_zscore', primary_signal.get('current_deviation', 0)):.2f}{'σ' if 'current_deviation' in primary_signal else ''}
                                - 통과 방법론: {len(signals)}개 ({', '.join([method_names[methods.index(s.get('method', 'unknown'))] if s.get('method', 'unknown') in methods else s.get('method', 'unknown') for s in signals])})
                                """)
                            else:
                                st.warning("차트를 생성할 수 없습니다.")
                    
                    except Exception as e:
                        st.error(f"차트 생성 중 오류: {str(e)}")
        
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
                
                with st.expander(f"{formatted_pair} ({name})"):
                    # 기본 정보 표시
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("진입 방향", signal['direction'])
                    with col2:
                        if 'current_zscore' in signal:
                            st.metric("Z-Score", f"{signal['current_zscore']:.2f}")
                        elif 'current_deviation' in signal:
                            st.metric("편차", f"{signal['current_deviation']:.2f}σ")
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
                                - 현재 Z-Score: {signal.get('current_zscore', signal.get('current_deviation', 0)):.2f}{'σ' if 'current_deviation' in signal else ''}
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
            
            score_text = f"Z-Score: {signal['current_zscore']:.2f}" if 'current_zscore' in signal else f"편차: {signal['current_deviation']:.2f}σ" if 'current_deviation' in signal else "점수: N/A"
            with st.expander(f"{i}. {formatted_pair} ({name}) - {score_text}"):
                # 기본 정보 표시
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if 'direction' in signal:
                        st.metric("예상 방향", signal['direction'])
                    else:
                        st.metric("상태", "관찰 중")
                with col2:
                    if 'current_zscore' in signal:
                        st.metric("Z-Score", f"{signal['current_zscore']:.2f}")
                    elif 'current_deviation' in signal:
                        st.metric("편차", f"{signal['current_deviation']:.2f}σ")
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
                            - 현재 Z-Score: {signal.get('current_zscore', signal.get('current_deviation', 0)):.2f}{'σ' if 'current_deviation' in signal else ''} (진입 대기 중)
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
        st.metric("다중 필터 통과 페어", consensus_count)
    with col4:
        active_methods = sum(1 for method in methods if cache_info.get(method, {}).get('exists', False))
        st.metric("활성 방법론", f"{active_methods}/6")

if __name__ == "__main__":
    main()
else:
    main()
