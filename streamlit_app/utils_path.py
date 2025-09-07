"""
경로 관련 유틸리티 함수
Streamlit Cloud와 로컬 환경 모두에서 작동하도록 함
"""
import os
import sys

def get_project_root():
    """프로젝트 루트 경로를 반환"""
    # Streamlit Cloud 환경 체크
    if os.path.exists('/mount/src/pair_trading_signal'):
        return '/mount/src/pair_trading_signal'
    
    # 로컬 환경: streamlit_app/utils_path.py 기준으로 상위 폴더가 프로젝트 루트
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)

def get_data_file_path():
    """데이터 파일 경로 반환"""
    project_root = get_project_root()
    
    # 여러 가능한 경로 시도
    possible_paths = [
        os.path.join(project_root, "data", "MU Price(BBG).csv"),
        os.path.join(project_root, "streamlit_app", "MU Price(BBG).csv"),  # streamlit_app 내부
        os.path.join(os.path.dirname(__file__), "MU Price(BBG).csv"),  # utils_path.py와 같은 폴더
        os.path.join(project_root, "data", "MU_Price_BBG.csv"),  # 괄호 없는 버전
        os.path.join(project_root, "MU Price(BBG).csv"),  # 루트에 직접
        "/mount/src/pair_trading_signal/data/MU Price(BBG).csv",  # Streamlit Cloud 절대 경로
        "/mount/src/pair_trading_signal/streamlit_app/MU Price(BBG).csv",  # Streamlit Cloud streamlit_app 내부
    ]
    
    # 디버깅용 경로 출력
    print(f"[DEBUG] Project root: {project_root}")
    print(f"[DEBUG] Current working directory: {os.getcwd()}")
    
    # 데이터 디렉토리 확인
    data_dir = os.path.join(project_root, "data")
    print(f"[DEBUG] Data directory exists: {os.path.exists(data_dir)}")
    if os.path.exists(data_dir):
        print(f"[DEBUG] Files in data directory: {os.listdir(data_dir)}")
    
    for path in possible_paths:
        print(f"[DEBUG] Trying path: {path}")
        print(f"[DEBUG] File exists: {os.path.exists(path)}")
        if os.path.exists(path):
            print(f"[DEBUG] Using path: {path}")
            return path
    
    # 모든 경로가 실패한 경우
    print(f"[ERROR] No data file found in any of these paths: {possible_paths}")
    return possible_paths[0]  # 첫 번째 경로 반환 (오류 메시지용)

def setup_path():
    """프로젝트 경로 설정"""
    project_root = get_project_root()
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return project_root