"""
Streamlit 페이지들의 한국어 제목을 영어로 일괄 변경
"""
import os
import re

def update_page_titles():
    pages_dir = "/Users/a/PycharmProjects/pair_trading_signal/streamlit_app/pages"
    
    # 각 페이지별 제목 매핑
    title_mappings = {
        "2_Euclidean_Distance.py": {
            "korean_titles": [
                "페어트레이딩 분석 도구 - 유클리드 거리 기반 방법론",
                "유클리드 거리 방법론",
                "유클리드 거리 기반 페어트레이딩",
                "분석 설정",
                "기간 설정"
            ],
            "english_titles": [
                "Pair Trading Analysis Tool - Euclidean Distance Methodology",
                "Euclidean Distance Methodology", 
                "Euclidean Distance Pair Trading",
                "Analysis Settings",
                "Period Settings"
            ]
        },
        "3_SSD_Distance.py": {
            "korean_titles": [
                "페어트레이딩 분석 도구 - SSD 거리 기반 방법론",
                "SSD 거리 방법론",
                "SSD 거리 기반 페어트레이딩"
            ],
            "english_titles": [
                "Pair Trading Analysis Tool - SSD Distance Methodology",
                "SSD Distance Methodology",
                "SSD Distance Pair Trading"
            ]
        },
        "4_Cointegration.py": {
            "korean_titles": [
                "페어트레이딩 분석 도구 - 공적분 기반 방법론",
                "공적분 방법론",
                "공적분 기반 페어트레이딩"
            ],
            "english_titles": [
                "Pair Trading Analysis Tool - Cointegration Methodology",
                "Cointegration Methodology",
                "Cointegration Pair Trading"
            ]
        },
        "6_OU_Mean_Reversion.py": {
            "korean_titles": [
                "페어트레이딩 분석 도구 - OU 평균회귀 기반 방법론",
                "OU 평균회귀 방법론",
                "OU 평균회귀 기반 페어트레이딩"
            ],
            "english_titles": [
                "Pair Trading Analysis Tool - OU Mean Reversion Methodology",
                "OU Mean Reversion Methodology",
                "OU Mean Reversion Pair Trading"
            ]
        },
        "7_Clustering.py": {
            "korean_titles": [
                "페어트레이딩 분석 도구 - 클러스터링 기반 방법론",
                "클러스터링 방법론",
                "클러스터링 기반 페어트레이딩"
            ],
            "english_titles": [
                "Pair Trading Analysis Tool - Clustering Methodology",
                "Clustering Methodology",
                "Clustering Pair Trading"
            ]
        },
        "8_Copula_Rank_Correlation.py": {
            "korean_titles": [
                "페어트레이딩 분석 도구 - 코퓰라 순위상관 기반 방법론",
                "코퓰라 순위상관 방법론",
                "코퓰라 순위상관 기반 페어트레이딩"
            ],
            "english_titles": [
                "Pair Trading Analysis Tool - Copula Rank Correlation Methodology",
                "Copula Rank Correlation Methodology",
                "Copula Rank Correlation Pair Trading"
            ]
        }
    }
    
    # 공통 한국어-영어 매핑
    common_mappings = {
        "분석 설정": "Analysis Settings",
        "기간 설정": "Period Settings", 
        "신호 설정": "Signal Settings",
        "품질 필터": "Quality Filter",
        "분석 실행": "Run Analysis",
        "분석 결과 요약": "Analysis Results Summary",
        "상세 작동 과정": "Detailed Process",
        "상세 설명": "Detailed Description", 
        "수식 및 계산": "Formulas & Calculations",
        "진입 신호": "Entry Signals",
        "관찰 대상": "Watch List",
        "페어 상세 분석": "Pair Detail Analysis"
    }
    
    for filename, mappings in title_mappings.items():
        file_path = os.path.join(pages_dir, filename)
        
        if not os.path.exists(file_path):
            print(f"파일이 없습니다: {filename}")
            continue
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 제목별 치환
            korean_titles = mappings["korean_titles"]
            english_titles = mappings["english_titles"]
            
            for korean, english in zip(korean_titles, english_titles):
                content = content.replace(f'"{korean}"', f'"{english}"')
                content = content.replace(f"'{korean}'", f"'{english}'")
                content = content.replace(korean, english)
            
            # 공통 매핑 적용
            for korean, english in common_mappings.items():
                content = content.replace(f'"{korean}"', f'"{english}"')
                content = content.replace(f"'{korean}'", f"'{english}'")
            
            # 파일에 다시 쓰기
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ {filename} 업데이트 완료")
            
        except Exception as e:
            print(f"❌ {filename} 업데이트 실패: {str(e)}")

if __name__ == "__main__":
    update_page_titles()
    print("🎉 모든 페이지 제목 업데이트 완료!")