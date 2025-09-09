"""
5) 클러스터링/최근접 이웃(간단 ML) 기반 - 동류끼리 묶고, 그 안에서 페어
핵심: 비슷한 특성의 자산을 클러스터로 묶고, 클러스터 내 최근접 이웃을 페어로
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster
import os
import sys
import importlib.util

# 공통 유틸리티 동적 import
def import_common_utils():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    utils_path = os.path.join(os.path.dirname(current_dir), "utils", "common_utils.py")
    spec = importlib.util.spec_from_file_location("common_utils", utils_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# 공통 함수들 import
common_utils = import_common_utils()
load_data = common_utils.load_data
normalize_prices = common_utils.normalize_prices
calculate_spread = common_utils.calculate_spread
calculate_zscore = common_utils.calculate_zscore
calculate_half_life = common_utils.calculate_half_life
generate_trading_signals = common_utils.generate_trading_signals
calculate_transaction_cost_ratio = common_utils.calculate_transaction_cost_ratio
euclidean_distance_matrix = common_utils.euclidean_distance_matrix

class ClusteringPairTrading:
    def __init__(self, formation_window: int = 252, signal_window: int = 60,
                 enter_threshold: float = 2.0, exit_threshold: float = 0.5,
                 stop_loss: float = 3.0, min_half_life: int = 5, max_half_life: int = 60,
                 min_cost_ratio: float = 5.0, n_clusters: int = 8, 
                 clustering_method: str = 'kmeans'):
        """
        클러스터링 기반 페어트레이딩 파라미터
        
        Args:
            formation_window: 페어 선정 기간 (영업일)
            signal_window: z-스코어 계산 롤링 윈도우
            enter_threshold: 진입 z-스코어 임계값
            exit_threshold: 청산 z-스코어 임계값
            stop_loss: 손절 z-스코어 임계값
            min_half_life: 최소 반감기 (영업일)
            max_half_life: 최대 반감기 (영업일)
            min_cost_ratio: 최소 1σ/거래비용 비율
            n_clusters: 클러스터 개수
            clustering_method: 'kmeans' 또는 'hierarchical'
        """
        self.formation_window = formation_window
        self.signal_window = signal_window
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss = stop_loss
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.min_cost_ratio = min_cost_ratio
        self.n_clusters = n_clusters
        self.clustering_method = clustering_method
        self.scaler = StandardScaler()
    
    def extract_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        클러스터링을 위한 특징 추출
        
        Args:
            prices: 가격 데이터
            
        Returns:
            특징 데이터프레임 (자산별 특징 벡터)
        """
        features_dict = {}
        
        for asset in prices.columns:
            price_series = prices[asset].dropna()
            
            if len(price_series) < 50:  # 최소 데이터 요구사항
                continue
            
            returns = price_series.pct_change().dropna()
            
            if len(returns) < 30:
                continue
            
            # 1. 수익률 통계
            mean_return = returns.mean()
            volatility = returns.std()
            skewness = returns.skew() if len(returns) > 2 else 0
            kurtosis = returns.kurtosis() if len(returns) > 3 else 0
            
            # 2. 가격 추세
            normalized_prices = normalize_prices(price_series.to_frame(), method='rebase')[asset]
            price_trend = (normalized_prices.iloc[-1] - normalized_prices.iloc[0]) / len(normalized_prices)
            
            # 3. 최대 낙폭 (Maximum Drawdown)
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # 4. 변동성 클러스터링 (GARCH 효과 근사)
            returns_squared = returns ** 2
            vol_autocorr = returns_squared.autocorr(lag=1) if len(returns_squared) > 1 else 0
            
            # 5. 베타 (시장 대비) - SPX Index를 벤치마크로 사용
            market_beta = 1.0  # 기본값
            if 'SPX Index' in prices.columns and asset != 'SPX Index':
                market_returns = prices['SPX Index'].pct_change().dropna()
                common_idx = returns.index.intersection(market_returns.index)
                if len(common_idx) > 30:
                    asset_ret_common = returns[common_idx]
                    market_ret_common = market_returns[common_idx]
                    covariance = np.cov(asset_ret_common, market_ret_common)[0, 1]
                    market_variance = np.var(market_ret_common)
                    market_beta = covariance / market_variance if market_variance > 0 else 1.0
            
            # 6. 거래량 프록시 (가격 변화 크기의 역수로 근사)
            # 실제 거래량 데이터가 없으므로 가격 변화의 일관성으로 대체
            price_consistency = 1 / (volatility + 1e-6)  # 변동성이 낮을수록 일관성 높음
            
            # 7. 최근 모멘텀 (단기 vs 중기 수익률)
            if len(returns) >= 60:
                short_momentum = returns.tail(20).sum()  # 최근 1개월
                medium_momentum = returns.tail(60).sum()  # 최근 3개월
                momentum_ratio = short_momentum / (medium_momentum + 1e-6)
            else:
                momentum_ratio = 0
            
            features_dict[asset] = {
                'mean_return': mean_return,
                'volatility': volatility,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'price_trend': price_trend,
                'max_drawdown': max_drawdown,
                'vol_autocorr': vol_autocorr,
                'market_beta': market_beta,
                'price_consistency': price_consistency,
                'momentum_ratio': momentum_ratio
            }
        
        features_df = pd.DataFrame(features_dict).T
        
        # 무한값이나 NaN 제거
        features_df = features_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        return features_df
    
    def perform_clustering(self, features: pd.DataFrame) -> Dict[int, List[str]]:
        """
        클러스터링 수행
        
        Args:
            features: 특징 데이터프레임
            
        Returns:
            {cluster_id: [asset_list]} 형태의 클러스터 정보
        """
        if len(features) < self.n_clusters:
            # 클러스터 수보다 자산이 적으면 모든 자산을 하나의 클러스터로
            return {0: list(features.index)}
        
        # 특징 표준화
        features_scaled = self.scaler.fit_transform(features)
        
        if self.clustering_method == 'kmeans':
            # K-Means 클러스터링
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, 
                          n_init=10, max_iter=300)
            cluster_labels = kmeans.fit_predict(features_scaled)
            
        elif self.clustering_method == 'hierarchical':
            # 계층적 클러스터링
            linkage_matrix = linkage(features_scaled, method='ward')
            cluster_labels = fcluster(linkage_matrix, self.n_clusters, criterion='maxclust')
            cluster_labels = cluster_labels - 1  # 0부터 시작하도록 조정
            
        else:
            raise ValueError("clustering_method must be 'kmeans' or 'hierarchical'")
        
        # 클러스터별 자산 그룹화
        clusters = {}
        for i, asset in enumerate(features.index):
            cluster_id = cluster_labels[i]
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(asset)
        
        return clusters
    
    def find_pairs_within_cluster(self, cluster_assets: List[str], prices: pd.DataFrame) -> List[Tuple[str, str, float]]:
        """
        클러스터 내에서 최근접 이웃 페어 찾기
        
        Args:
            cluster_assets: 클러스터에 속한 자산 리스트
            prices: 가격 데이터
            
        Returns:
            (asset1, asset2, distance) 튜플 리스트
        """
        if len(cluster_assets) < 2:
            return []
        
        # 클러스터 내 자산들의 가격 데이터 추출
        cluster_prices = prices[cluster_assets].dropna()
        
        if len(cluster_prices) < 30:  # 충분한 데이터 없음
            return []
        
        # 표준화된 가격으로 거리 계산
        normalized_prices = normalize_prices(cluster_prices, method='rebase')
        distance_matrix = euclidean_distance_matrix(normalized_prices)
        
        # 최근접 이웃 페어 추출
        pairs = []
        n_assets = len(cluster_assets)
        
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                distance = distance_matrix[i][j]
                pairs.append((cluster_assets[i], cluster_assets[j], distance))
        
        # 거리 기준 오름차순 정렬
        pairs.sort(key=lambda x: x[2])
        
        return pairs
    
    def select_pairs(self, prices: pd.DataFrame, n_pairs: int = 20) -> List[Dict]:
        """
        클러스터링 기반 페어 선정
        
        Args:
            prices: 가격 데이터
            n_pairs: 선정할 페어 개수
            
        Returns:
            선정된 페어 정보 리스트
        """
        # 최근 formation_window 기간 데이터 추출
        formation_data = prices.tail(self.formation_window)
        
        # 결측치가 많은 자산 제외
        valid_assets = []
        for col in formation_data.columns:
            if formation_data[col].notna().sum() >= self.formation_window * 0.8:
                valid_assets.append(col)
        
        if len(valid_assets) < 2:
            return []
            
        formation_data = formation_data[valid_assets].fillna(method='ffill')
        
        # 특징 추출
        features = self.extract_features(formation_data)
        
        if len(features) < 2:
            return []
        
        # 클러스터링 수행
        clusters = self.perform_clustering(features)
        
        # 각 클러스터 내에서 페어 찾기
        all_pairs = []
        
        for cluster_id, cluster_assets in clusters.items():
            cluster_pairs = self.find_pairs_within_cluster(cluster_assets, formation_data)
            
            for asset1, asset2, distance in cluster_pairs:
                # 스프레드 계산 및 품질 검사
                spread = calculate_spread(
                    formation_data[asset1], 
                    formation_data[asset2], 
                    hedge_ratio=1.0
                )
                
                half_life = calculate_half_life(spread)
                cost_ratio = calculate_transaction_cost_ratio(spread)
                
                # 품질 필터
                if (self.min_half_life <= half_life <= self.max_half_life and 
                    cost_ratio >= self.min_cost_ratio):
                    
                    all_pairs.append({
                        'asset1': asset1,
                        'asset2': asset2,
                        'cluster_id': cluster_id,
                        'distance': distance,
                        'half_life': half_life,
                        'cost_ratio': cost_ratio,
                        'hedge_ratio': 1.0,
                        'method': 'clustering'
                    })
        
        # 거리 기준 정렬 후 중복 제거
        all_pairs.sort(key=lambda x: x['distance'])
        
        selected_pairs = []
        used_assets = set()
        
        for pair_info in all_pairs:
            if len(selected_pairs) >= n_pairs:
                break
                
            asset1, asset2 = pair_info['asset1'], pair_info['asset2']
            if asset1 not in used_assets and asset2 not in used_assets:
                selected_pairs.append(pair_info)
                used_assets.add(asset1)
                used_assets.add(asset2)
        
        return selected_pairs
    
    def get_cluster_summary(self, prices: pd.DataFrame) -> Dict:
        """
        클러스터링 결과 요약 정보
        
        Returns:
            클러스터별 요약 정보
        """
        formation_data = prices.tail(self.formation_window)
        
        valid_assets = [col for col in formation_data.columns 
                       if formation_data[col].notna().sum() >= self.formation_window * 0.8]
        
        if len(valid_assets) < 2:
            return {}
            
        formation_data = formation_data[valid_assets].fillna(method='ffill')
        features = self.extract_features(formation_data)
        
        if len(features) < 2:
            return {}
        
        clusters = self.perform_clustering(features)
        
        cluster_summary = {}
        for cluster_id, assets in clusters.items():
            # 클러스터 내 자산들의 평균 특성
            cluster_features = features.loc[assets].mean()
            
            cluster_summary[cluster_id] = {
                'assets': assets,
                'n_assets': len(assets),
                'avg_volatility': cluster_features['volatility'],
                'avg_return': cluster_features['mean_return'],
                'avg_beta': cluster_features['market_beta'],
                'avg_momentum': cluster_features['momentum_ratio']
            }
        
        return cluster_summary
    
    def generate_signals(self, prices: pd.DataFrame, pair_info: Dict) -> Dict:
        """
        특정 페어에 대한 트레이딩 신호 생성
        
        Args:
            prices: 전체 가격 데이터
            pair_info: 페어 정보 딕셔너리
            
        Returns:
            신호 정보 딕셔너리
        """
        asset1, asset2 = pair_info['asset1'], pair_info['asset2']
        
        # 최근 데이터 확보
        recent_data = prices[[asset1, asset2]].tail(self.signal_window * 2).fillna(method='ffill')
        
        if len(recent_data) < self.signal_window:
            return {'status': 'insufficient_data'}
        
        # 스프레드 계산
        spread = calculate_spread(
            recent_data[asset1],
            recent_data[asset2],
            hedge_ratio=pair_info['hedge_ratio']
        )
        
        # Z-스코어 계산
        zscore = calculate_zscore(spread, window=self.signal_window)
        current_zscore = zscore.iloc[-1] if not zscore.empty else 0
        
        # 신호 생성
        signals = generate_trading_signals(
            zscore, 
            enter_threshold=self.enter_threshold,
            exit_threshold=self.exit_threshold,
            stop_loss=self.stop_loss
        )
        
        current_signal = signals.iloc[-1] if not signals.empty else 0
        
        # 신호 해석
        if current_signal == 1:
            signal_type = "ENTER_LONG"
            direction = f"Long {asset1}, Short {asset2}"
        elif current_signal == -1:
            signal_type = "ENTER_SHORT"
            direction = f"Short {asset1}, Long {asset2}"
        else:
            signal_type = "EXIT_OR_WAIT"
            direction = "Exit or Wait"
        
        return {
            'status': 'success',
            'pair': f"{asset1}-{asset2}",
            'signal_type': signal_type,
            'direction': direction,
            'current_zscore': current_zscore,
            'cluster_id': pair_info['cluster_id'],
            'distance': pair_info['distance'],
            'half_life': pair_info['half_life'],
            'cost_ratio': pair_info['cost_ratio'],
            'method': 'clustering'
        }
    
    def screen_pairs(self, prices: pd.DataFrame, n_pairs: int = 10) -> Tuple[List[Dict], List[Dict]]:
        """
        전체 페어 스크리닝 및 신호 생성
        
        Returns:
            (enter_signals, watch_signals): 진입 신호와 관찰 대상 리스트
        """
        # 페어 선정
        selected_pairs = self.select_pairs(prices, n_pairs * 2)
        
        enter_signals = []
        watch_signals = []
        
        for pair_info in selected_pairs:
            signal_result = self.generate_signals(prices, pair_info)
            
            if signal_result['status'] != 'success':
                continue
                
            current_z = abs(signal_result['current_zscore'])
            
            # 진입 신호 (|z| >= 2.0)
            if current_z >= self.enter_threshold:
                enter_signals.append(signal_result)
            # 관찰 대상 (1.5 <= |z| < 2.0)
            elif 1.5 <= current_z < self.enter_threshold:
                watch_signals.append(signal_result)
        
        # 거리가 가까운 순으로 정렬 (클러스터 내 유사성 높은 순)
        enter_signals.sort(key=lambda x: x['distance'])
        watch_signals.sort(key=lambda x: x['distance'])
        
        return enter_signals[:n_pairs], watch_signals[:n_pairs]

def main():
    """
    클러스터링 기반 페어트레이딩 실행 예제
    """
    # 데이터 로딩
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    file_path = os.path.join(project_root, "data", "MU Price(BBG).csv")
    prices = load_data(file_path)
    
    # 클러스터링 기반 페어트레이딩 객체 생성
    clustering_trader = ClusteringPairTrading(
        formation_window=252,      # 1년
        signal_window=60,          # 3개월
        enter_threshold=2.0,
        exit_threshold=0.5,
        stop_loss=3.0,
        min_half_life=5,
        max_half_life=60,
        min_cost_ratio=5.0,
        n_clusters=8,              # 8개 클러스터
        clustering_method='kmeans'  # K-Means 방법
    )
    
    # 클러스터링 요약 정보
    cluster_summary = clustering_trader.get_cluster_summary(prices)
    
    print("=" * 70)
    print("클러스터링 기반 페어트레이딩 신호")
    print("=" * 70)
    
    print(f"\n🎯 클러스터 요약 ({len(cluster_summary)}개):")
    print("-" * 50)
    for cluster_id, info in cluster_summary.items():
        print(f"클러스터 {cluster_id}: {info['n_assets']}개 자산")
        print(f"  자산: {', '.join(info['assets'][:3])}{'...' if len(info['assets']) > 3 else ''}")
        print(f"  평균 변동성: {info['avg_volatility']:.4f} | 평균 베타: {info['avg_beta']:.3f}")
        print()
    
    # 페어 스크리닝
    enter_list, watch_list = clustering_trader.screen_pairs(prices, n_pairs=10)
    
    print(f"\n📈 진입 신호 ({len(enter_list)}개):")
    print("-" * 60)
    for i, signal in enumerate(enter_list, 1):
        print(f"{i:2d}. {signal['pair']:20s} | {signal['direction']:25s}")
        print(f"     Z-Score: {signal['current_zscore']:6.2f} | 클러스터: {signal['cluster_id']}")
        print(f"     거리: {signal['distance']:6.3f} | Half-Life: {signal['half_life']:4.1f}D")
        print(f"     Cost Ratio: {signal['cost_ratio']:5.1f}")
        print()
    
    print(f"\n👀 관찰 대상 ({len(watch_list)}개):")
    print("-" * 60)
    for i, signal in enumerate(watch_list, 1):
        print(f"{i:2d}. {signal['pair']:20s} | Z-Score: {signal['current_zscore']:6.2f}")
        print(f"     클러스터: {signal['cluster_id']} | 거리: {signal['distance']:6.3f}")
        print(f"     Half-Life: {signal['half_life']:4.1f}D | Cost Ratio: {signal['cost_ratio']:5.1f}")
        print()

if __name__ == "__main__":
    main()