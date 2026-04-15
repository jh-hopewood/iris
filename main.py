import streamlit as st
import numpy as np
import pandas as pd
import os
from streamlit_echarts import st_echarts

st.set_page_config(
    page_title="KNN 시뮬레이터",
    layout="centered"
)

# ============================================================
# 데이터 로드 (같은 폴더의 CSV 파일 읽기)
# ============================================================
DATA_FILE = "knn_data.csv"

@st.cache_data
def load_data():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        # CSV 컬럼명에 맞춰 추출 (x, y, label)
        X = df[['꽃받침 길이', '꽃받침 너비']].values.astype(float)
        y = df['종류'].values
        return X, y
    else:
        return None, None

X_knn, y_knn = load_data()

# ============================================================
# 상태 초기화 및 스타일 설정
# ============================================================
if "knn_initialized" not in st.session_state:
    st.session_state.knn_new_point = None
    st.session_state.knn_step = 0
    st.session_state.knn_distances = None
    st.session_state.knn_sorted_idx = None
    st.session_state.knn_final_label = None
    st.session_state.knn_initialized = True

CLASS_COLORS = {
    "빨강": "#e74c3c",
    "파랑": "#3498db",
    "새 데이터": "#2ecc71"
}

# ------------------------------------------------------------
# 핵심 알고리즘 및 헬퍼 함수 (기존 로직 유지)
# ------------------------------------------------------------
def predict_knn(point, k):
    dists = np.linalg.norm(X_knn - point, axis=1)
    idx = np.argsort(dists)[:k]
    neighbor_labels = y_knn[idx]
    unique, counts = np.unique(neighbor_labels, return_counts=True)
    max_count = np.max(counts)
    winners = unique[counts == max_count]
    return winners[0] if len(winners) == 1 else y_knn[np.argsort(dists)[0]]

def prepare_knn(new_point, k):
    st.session_state.knn_step = 0
    st.session_state.knn_new_point = np.array(new_point, dtype=float)
    distances = np.linalg.norm(X_knn - st.session_state.knn_new_point, axis=1)
    st.session_state.knn_distances = distances
    st.session_state.knn_sorted_idx = np.argsort(distances)
    st.session_state.knn_final_label = predict_knn(st.session_state.knn_new_point, k)

# (build_knn_option 함수는 기존과 동일하여 생략 가능하지만, 전체 흐름을 위해 유지)
def build_knn_option(k):
    if X_knn is None: return {}
    
    red_points = [{"value": [p[0], p[1]], "label": {"show": True, "formatter": str(i)}} 
                  for i, (p, l) in enumerate(zip(X_knn, y_knn)) if l == "빨강"]
    blue_points = [{"value": [p[0], p[1]], "label": {"show": True, "formatter": str(i)}} 
                   for i, (p, l) in enumerate(zip(X_knn, y_knn)) if l == "파랑"]

    series = [
        {"name": "빨강", "type": "scatter", "data": red_points, "itemStyle": {"color": CLASS_COLORS["빨강"]}},
        {"name": "파랑", "type": "scatter", "data": blue_points, "itemStyle": {"color": CLASS_COLORS["파랑"]}}
    ]

    if st.session_state.knn_new_point is not None:
        p = st.session_state.knn_new_point
        series.append({
            "name": "새 데이터", "type": "scatter", "symbol": "triangle", "symbolSize": 18,
            "data": [{"value": [p[0], p[1]], "label": {"show": True, "formatter": "새", "position": "top"}}],
            "itemStyle": {"color": CLASS_COLORS["새 데이터"]}
        })

        if st.session_state.knn_step > 0:
            for i in range(min(st.session_state.knn_step, k)):
                idx = st.session_state.knn_sorted_idx[i]
                neighbor = X_knn[idx]
                # 선 및 강조 원 그리기 로직 (생략된 기존 코드와 동일)
                series.append({"type": "line", "data": [[p[0], p[1]], [neighbor[0], neighbor[1]]], "lineStyle": {"type": "dashed", "color": "#888"}})
    
    return {
        "tooltip": {"trigger": "item"},
        "legend": {"data": ["빨강", "파랑", "새 데이터"]},
        "xAxis": {"type": "value", "min": 0, "max": 10},
        "yAxis": {"type": "value", "min": 0, "max": 10},
        "series": series
    }

# ============================================================
# UI 메인 화면
# ============================================================
st.title("KNN 데이터셋 시뮬레이터")

if X_knn is None:
    st.error(f"'{DATA_FILE}' 파일을 찾을 수 없습니다. 같은 폴더에 파일을 생성해 주세요.")
else:
    st.success(f"성공: {len(X_knn)}개의 데이터를 불러왔습니다.")
    
    @st.fragment
    def knn_panel():
        with st.form("knn_input"):
            c1, c2, c3 = st.columns([1, 1, 1.2])
            x_val = c1.number_input("X", 0.0, 10.0, 5.0, key="kx")
            y_val = c2.number_input("Y", 0.0, 10.0, 5.0, key="ky")
            k_val = c3.slider("K값", 1, 7, 3, key="kk")
            
            btn_c1, btn_c2, btn_c3 = st.columns(3)
            if btn_c1.form_submit_button("점 찍기"): prepare_knn([x_val, y_val], k_val)
            if btn_c2.form_submit_button("다음 단계"):
                if st.session_state.knn_new_point is not None and st.session_state.knn_step < k_val:
                    st.session_state.knn_step += 1
            if btn_c3.form_submit_button("초기화"):
                for key in ["knn_new_point", "knn_step", "knn_distances"]: st.session_state[key] = None

        option = build_knn_option(k_val)
        st_echarts(options=option, height="500px")

    knn_panel()
