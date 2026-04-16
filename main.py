import streamlit as st
import numpy as np
import pandas as pd
import os
from streamlit_echarts import st_echarts

# 1. 페이지 설정
st.set_page_config(page_title="KNN 데이터 시뮬레이터", layout="wide")

# 2. 데이터 로드 (캐싱 및 인덱스 기반 추출로 에러 방지)
DATA_FILE = "knn_data.csv"

@st.cache_data
def load_data():
    if os.path.exists(DATA_FILE):
        # 한글 깨짐 방지 및 CSV 로드
        df = pd.read_csv(DATA_FILE, encoding='utf-8-sig')
        try:
            # 컬럼명에 상관없이 0, 1번째 열은 좌표, 마지막 열은 라벨
            X = df.iloc[:, [0, 1]].values.astype(float)
            y = df.iloc[:, -1].values
            return X, y, df.columns[0], df.columns[1]
        except Exception as e:
            st.error(f"데이터 로드 중 오류 발생: {e}")
            return None, None, None, None
    return None, None, None, None

X_knn, y_knn, x_label, y_label = load_data()

# 3. 세션 상태 초기화 (앱 실행 시 한 번만)
if "knn_step" not in st.session_state:
    st.session_state.update({
        "knn_new_point": None, 
        "knn_step": 0, 
        "knn_sorted_idx": None, 
        "knn_distances": None
    })

# 4. 색상 맵 자동 설정
COLOR_MAP = {}
if y_knn is not None:
    unique_labels = np.unique(y_knn)
    palette = ["#e74c3c", "#3498db", "#f1c40f", "#9b59b6", "#e67e22"]
    COLOR_MAP = {label: palette[i % len(palette)] for i, label in enumerate(unique_labels)}
    COLOR_MAP["새 데이터"] = "#2ecc71"

# 5. 차트 옵션 생성 함수
def build_chart_options(k_val):
    series = []
    
    # 배경 데이터 (152개 샘플)
    for label in np.unique(y_knn):
        mask = (y_knn == label)
        series.append({
            "name": str(label),
            "type": "scatter",
            "symbolSize": 10,
            "data": X_knn[mask].tolist(),
            "itemStyle": {"color": COLOR_MAP[label], "opacity": 0.5}
        })

    # 새 데이터 투입 시 시각화
    if st.session_state.knn_new_point is not None:
        p = st.session_state.knn_new_point.tolist()
        
        # 새 데이터 포인트 (삼각형)
        series.append({
            "name": "새 데이터",
            "type": "scatter",
            "symbol": "triangle",
            "symbolSize": 20,
            "data": [p],
            "itemStyle": {"color": COLOR_MAP["새 데이터"]},
            "z": 10
        })

        # 다음 찾기 단계별 강조 및 연결선
        for i in range(st.session_state.knn_step):
            idx = st.session_state.knn_sorted_idx[i]
            neighbor = X_knn[idx].tolist()
            
            # 연결 점선
            series.append({
                "type": "line",
                "data": [p, neighbor],
                "symbol": "none",
                "lineStyle": {"type": "dashed", "width": 1.5, "color": "#444"},
                "z": 5
            })
            
            # 선택된 이웃 강조 원
            series.append({
                "type": "scatter",
                "symbolSize": 22,
                "data": [neighbor],
                "itemStyle": {
                    "color": "none",
                    "borderColor": COLOR_MAP[y_knn[idx]],
                    "borderWidth": 3
                },
                "label": {
                    "show": True,
                    "formatter": f"{i+1}NN",
                    "position": "top",
                    "color": "#333",
                    "fontWeight": "bold"
                },
                "z": 11
            })

    return {
        "tooltip": {"trigger": "item"},
        "legend": {"top": "5%", "data": list(COLOR_MAP.keys())},
