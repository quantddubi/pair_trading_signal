"""
7) 통합 스크리너 - 합의 기반 및 앙상블 점수
핵심: 여러 방법론의 결과를 종합하여 최종 페어트레이딩 신호 생성
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

import sys
import importlib.util

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

def import_module_from_file(file_name, module_name):
    """같은 폴더 내의 모듈을 동적으로 import"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, file_name)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# 각 방법론 모듈 import (같은 폴더 내)
euclidean_module = import_module_from_file("1_euclidean_distance_pairs.py", "euclidean_distance_pairs")
cointegration_module = import_module_from_file("2_cointegration_pairs.py", "cointegration_pairs")
regime_module = import_module_from_file("3_correlation_regime_pairs.py", "correlation_regime_pairs")
ou_module = import_module_from_file("4_ou_mean_reversion_pairs.py", "ou_mean_reversion_pairs")
clustering_module = import_module_from_file("5_clustering_pairs.py", "clustering_pairs")
copula_module = import_module_from_file("6_copula_rank_correlation_pairs.py", "copula_rank_correlation_pairs")

EuclideanDistancePairTrading = euclidean_module.EuclideanDistancePairTrading
CointegrationPairTrading = cointegration_module.CointegrationPairTrading
CorrelationRegimePairTrading = regime_module.CorrelationRegimePairTrading
OUMeanReversionPairTrading = ou_module.OUMeanReversionPairTrading
ClusteringPairTrading = clustering_module.ClusteringPairTrading
CopulaRankCorrelationPairTrading = copula_module.CopulaRankCorrelationPairTrading

class IntegratedPairTradingScreener:
    def __init__(self, formation_window: int = 252, signal_window: int = 60,
                 enter_threshold: float = 2.0, exit_threshold: float = 0.5,
                 stop_loss: float = 3.0, consensus_threshold: int = 2,
                 enable_methods: List[str] = None):
        """
        통합 페어트레이딩 스크리너
        
        Args:
            formation_window: 페어 선정 기간 (영업일)
            signal_window: z-스코어 계산 롤링 윈도우
            enter_threshold: 진입 z-스코어 임계값
            exit_threshold: 청산 z-스코어 임계값
            stop_loss: 손절 z-스코어 임계값
            consensus_threshold: 합의 기준 (최소 몇 개 방법론에서 선정되어야 하는지)
            enable_methods: 활성화할 방법론 리스트 (None이면 빠른 방법론만)
        """
        self.formation_window = formation_window
        self.signal_window = signal_window
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss = stop_loss
        self.consensus_threshold = consensus_threshold
        
        # 기본은 빠른 방법론들만 (시간 절약)
        if enable_methods is None:
            enable_methods = ['euclidean', 'cointegration', 'regime']
        self.enable_methods = enable_methods
        
        # 각 방법론별 객체 초기화
        self.methods = {}
        self._initialize_methods()
        
        # 앙상블 가중치 (각 방법론별)
        self.ensemble_weights = {
            'euclidean': 0.25,      # 25% - 빠른 벤치마크
            'cointegration': 0.30,  # 30% - 이론적 근거 강함
            'regime': 0.25,         # 25% - 레짐 전환 포착
            'ou': 0.20,            # 20% - 평균회귀 속도
        }
    
    def _initialize_methods(self):
        """각 방법론별 객체 초기화"""
        
        if 'euclidean' in self.enable_methods:
            self.methods['euclidean'] = EuclideanDistancePairTrading(
                formation_window=self.formation_window,
                signal_window=self.signal_window,
                enter_threshold=self.enter_threshold,
                exit_threshold=self.exit_threshold,
                stop_loss=self.stop_loss
            )
        
        if 'cointegration' in self.enable_methods:
            self.methods['cointegration'] = CointegrationPairTrading(
                formation_window=self.formation_window,
                signal_window=self.signal_window,
                enter_threshold=self.enter_threshold,
                exit_threshold=self.exit_threshold,
                stop_loss=self.stop_loss
            )
        
        if 'regime' in self.enable_methods:
            self.methods['regime'] = CorrelationRegimePairTrading(
                formation_window=self.formation_window,
                signal_window=self.signal_window,
                enter_threshold=self.enter_threshold,
                exit_threshold=self.exit_threshold,
                stop_loss=self.stop_loss
            )
        
        if 'ou' in self.enable_methods:
            self.methods['ou'] = OUMeanReversionPairTrading(
                formation_window=self.formation_window,
                signal_window=self.signal_window,
                enter_threshold=self.enter_threshold,
                exit_threshold=self.exit_threshold,
                stop_loss=self.stop_loss
            )
    
    def collect_all_method_results(self, prices: pd.DataFrame, n_pairs_per_method: int = 15) -> Dict[str, List[Dict]]:
        """
        모든 방법론별 페어 선정 결과 수집
        
        Returns:
            {method_name: [pair_results]} 형태의 결과 딕셔너리
        """
        all_results = {}
        
        print("🔍 각 방법론별 페어 분석 중...")
        
        for method_name, method_obj in self.methods.items():
            try:
                print(f"  - {method_name} 방법론 실행...")
                enter_list, watch_list = method_obj.screen_pairs(prices, n_pairs_per_method)
                
                all_results[method_name] = enter_list + watch_list
                print(f"    → {len(enter_list)}개 진입 + {len(watch_list)}개 관찰 = {len(all_results[method_name])}개 페어")
                
            except Exception as e:
                print(f"    ⚠️ {method_name} 방법론 오류: {str(e)}")
                all_results[method_name] = []
        
        return all_results
    
    def find_consensus_pairs(self, all_results: Dict[str, List[Dict]]) -> List[Dict]:
        """
        합의 기반 페어 선정 (여러 방법론에서 공통으로 선정된 페어)
        
        Returns:
            합의된 페어 리스트
        """
        # 모든 페어 조합 수집
        pair_counts = {}  # {(asset1, asset2): [method_list]}
        pair_details = {}  # {(asset1, asset2): {method: details}}
        
        for method_name, method_results in all_results.items():
            for result in method_results:
                if 'pair' not in result:
                    continue
                    
                try:
                    asset1, asset2 = result['pair'].split('-')
                    # 정규화된 페어 키 (알파벳 순서)
                    pair_key = tuple(sorted([asset1, asset2]))
                    
                    if pair_key not in pair_counts:
                        pair_counts[pair_key] = []
                        pair_details[pair_key] = {}
                    
                    pair_counts[pair_key].append(method_name)
                    pair_details[pair_key][method_name] = result
                except:
                    continue
        
        # 합의 임계값 이상인 페어들 선정
        consensus_pairs = []
        
        for pair_key, methods in pair_counts.items():
            if len(methods) >= self.consensus_threshold:
                asset1, asset2 = pair_key
                
                # 방법론별 결과 통합
                consensus_info = {
                    'asset1': asset1,
                    'asset2': asset2,
                    'pair': f"{asset1}-{asset2}",
                    'consensus_methods': methods,
                    'consensus_count': len(methods),
                    'method_details': pair_details[pair_key]
                }
                
                # 대표 값들 계산 (평균/중앙값)
                z_scores = []
                half_lives = []
                
                for method in methods:
                    detail = pair_details[pair_key][method]
                    if 'current_zscore' in detail:
                        z_scores.append(abs(detail['current_zscore']))
                    if 'half_life' in detail:
                        half_lives.append(detail['half_life'])
                
                consensus_info['avg_zscore'] = np.mean(z_scores) if z_scores else 0
                consensus_info['avg_half_life'] = np.mean(half_lives) if half_lives else 0
                consensus_info['method'] = 'consensus'
                
                # 방향 결정
                pos_count = sum(1 for detail in pair_details[pair_key].values() 
                              if detail.get('current_zscore', 0) > 0)
                neg_count = len(methods) - pos_count
                
                if pos_count > neg_count:
                    consensus_info['direction'] = f"Long {asset1}, Short {asset2}"
                elif neg_count > pos_count:
                    consensus_info['direction'] = f"Short {asset1}, Long {asset2}"
                else:
                    consensus_info['direction'] = "Direction TBD"
                
                consensus_pairs.append(consensus_info)
        
        # 합의 강도(count) 및 z-스코어 기준 정렬
        consensus_pairs.sort(key=lambda x: (x['consensus_count'], x['avg_zscore']), reverse=True)
        
        return consensus_pairs
    
    def screen_integrated_pairs(self, prices: pd.DataFrame, n_pairs: int = 10, 
                              strategy: str = 'consensus') -> Tuple[List[Dict], List[Dict]]:
        """
        통합 페어 스크리닝
        
        Args:
            prices: 가격 데이터
            n_pairs: 최종 선정할 페어 개수
            strategy: 'consensus' (합의 기반)
            
        Returns:
            (enter_signals, watch_signals): 진입 신호와 관찰 대상 리스트
        """
        # 1. 모든 방법론 결과 수집
        all_results = self.collect_all_method_results(prices, n_pairs_per_method=15)
        
        # 2. 합의 기반 페어 선정
        integrated_pairs = self.find_consensus_pairs(all_results)
        print(f"\n✅ 합의 기반: {len(integrated_pairs)}개 페어 (최소 {self.consensus_threshold}개 방법론 동의)")
        
        # 3. 진입/관찰 신호 분류
        enter_signals = []
        watch_signals = []
        
        for pair_info in integrated_pairs:
            current_z = pair_info.get('avg_zscore', 0)
            
            # 기본 신호 정보 생성
            signal_info = {
                'pair': pair_info['pair'],
                'direction': pair_info['direction'],
                'current_zscore': current_z,
                'avg_half_life': pair_info.get('avg_half_life', 0),
                'method': 'consensus',
                'consensus_count': pair_info.get('consensus_count', 0),
                'consensus_methods': pair_info.get('consensus_methods', [])
            }
            
            # 진입/관찰 분류
            if current_z >= self.enter_threshold:
                signal_info['signal_type'] = 'ENTER'
                enter_signals.append(signal_info)
            elif 1.5 <= current_z < self.enter_threshold:
                signal_info['signal_type'] = 'WATCH'
                watch_signals.append(signal_info)
        
        return enter_signals[:n_pairs], watch_signals[:n_pairs]

def main():
    """
    통합 스크리너 실행 예제
    """
    # 데이터 로딩
    file_path = "/Users/a/PycharmProjects/pair_trading_signal/data/MU Price(BBG).csv"
    prices = load_data(file_path)
    
    print("=" * 90)
    print("📊 통합 페어트레이딩 스크리너 (Integrated Multi-Method Screener)")
    print("=" * 90)
    
    print(f"📈 데이터 로딩 완료: {len(prices.columns)}개 자산, {len(prices)}일 데이터")
    print(f"📅 분석 기간: {prices.index[0].strftime('%Y-%m-%d')} ~ {prices.index[-1].strftime('%Y-%m-%d')}")
    
    # 통합 스크리너 생성 (빠른 방법론들만)
    integrated_screener = IntegratedPairTradingScreener(
        formation_window=252,
        signal_window=60,
        enter_threshold=2.0,
        exit_threshold=0.5,
        stop_loss=3.0,
        consensus_threshold=2,  # 최소 2개 방법론에서 동의
        enable_methods=['euclidean', 'cointegration', 'regime']  # 빠른 방법론들
    )
    
    # 합의 기반 분석
    print(f"\n🎯 합의 기반 분석 (Consensus)")
    print("-" * 60)
    consensus_enter, consensus_watch = integrated_screener.screen_integrated_pairs(
        prices, n_pairs=10, strategy='consensus'
    )
    
    print(f"\n📈 진입 신호 ({len(consensus_enter)}개):")
    if consensus_enter:
        for i, signal in enumerate(consensus_enter, 1):
            print(f"{i:2d}. {signal['pair']:25s} | {signal['direction']:30s}")
            print(f"     Z-Score: {signal['current_zscore']:6.2f} | 합의도: {signal['consensus_count']}/{len(integrated_screener.enable_methods)} 방법론")
            print(f"     동의 방법론: {', '.join(signal['consensus_methods'])}")
            print(f"     Half-Life: {signal['avg_half_life']:4.1f}일")
            print()
    else:
        print("     현재 합의된 진입 신호가 없습니다.")
    
    print(f"\n👀 관찰 대상 ({len(consensus_watch)}개):")
    if consensus_watch:
        for i, signal in enumerate(consensus_watch, 1):
            print(f"{i:2d}. {signal['pair']:25s} | Z-Score: {signal['current_zscore']:6.2f}")
            print(f"     합의도: {signal['consensus_count']}/{len(integrated_screener.enable_methods)} | 방법: {', '.join(signal['consensus_methods'])}")
    else:
        print("     현재 합의된 관찰 대상이 없습니다.")
    
    print("\n" + "="*90)
    print(f"📋 분석 요약:")
    print(f"   • 사용 방법론: {', '.join(integrated_screener.enable_methods)}")
    print(f"   • 합의 진입 신호: {len(consensus_enter)}개")
    print(f"   • 합의 관찰 대상: {len(consensus_watch)}개")
    print(f"   • 합의 기준: 최소 {integrated_screener.consensus_threshold}개 방법론 동의")
    print("✨ 합의 기반 신호는 여러 방법론이 동의하는 신호로 신뢰성이 높습니다!")

if __name__ == "__main__":
    main()