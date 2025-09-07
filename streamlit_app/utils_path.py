"""
경로 관련 유틸리티 함수
Streamlit Cloud와 로컬 환경 모두에서 작동하도록 함
"""
import os
import sys

def get_project_root():
    """프로젝트 루트 경로를 반환"""
    # streamlit_app/utils_path.py 기준으로 상위 폴더가 프로젝트 루트
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)

def get_data_file_path():
    """데이터 파일 경로 반환"""
    project_root = get_project_root()
    return os.path.join(project_root, "data", "MU Price(BBG).csv")

def setup_path():
    """프로젝트 경로 설정"""
    project_root = get_project_root()
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return project_root