import streamlit as st
import numpy as np
import pandas as pd
import os
from streamlit_echarts import st_echarts

# 1. 페이지 설정 (최상단에 한 번만)
st.set_page_config(page_title="KNN 시뮬레이터", layout="wide")

# 2. 데이터 로드 (캐싱을 통해 리로드 시 속도 최적화)
DATA_FILE = "knn_data.csv"

@st.cache_data
def load_data():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE, encoding='utf-8-sig')
        X = df.iloc[:, [0, 1]].values.astype(float)
        y = df.iloc[:, -1].values
        return X, y, df.columns[0], df.columns[1]
    return None, None, None, None

X_knn, y_knn, x_label, y_label = load_data()

# 3. 세션 상태 초기화
if "knn_step" not in st.session_state:
    st.session_state.update({
        "knn_new_point": None, 
        "knn_step": 0, 
        "knn_sorted_idx": None, 
        "knn_distances": None
    })

# 4. 색상 맵 설정
if y_knn is not None:
    unique_labels = np.unique(y_knn)
    palette = ["#e74c3c", "#3498db", "#f1c40f", "#9b59b6", "#e67e22"]
    COLOR_MAP = {label: palette[i % len(palette)] for i, label in enumerate(unique_labels)}
    COLOR_MAP["새 데이터"] = "#2ecc71"

# ---------------------------------------------------------
# 5. 메인 시각화 패널 (@st.fragment 적용)
# ---------------------------------------------------------
@st.fragment
def knn_main_panel():
    # 내부 서브 레이아웃 설정
    ctrl_col, chart_col = st.columns([1, 2.5])

    with ctrl_col:
        st.subheader("📍 제어판")
        new_x = st.number_input(f"{x_label}", value=float(np.mean(X_knn[:,0])))
        new_y = st.number_input(f"{y_label}", value=float(np.mean(X_knn[:,1])))
        k_val = st.slider("확인할 이웃 수(K)", 1, 15, 3)
        
        # 버튼들
        if st.button("1. 데이터 투입", use_container_width=True):
            st.session_state.knn_new_point = np.array([new_x, new_y])
            st.session_state.knn_step = 0
            dists = np.linalg.norm(X_knn - st.session_state.knn_new_point, axis=1)
            st.session_state.knn_sorted_idx = np.argsort(dists)
            st.session_state.knn_distances = dists

        if st.button("2. 다음 찾기 (Step)", use_container_width=True):
            if st.session_state.knn_new_point is not None and st.session_state.knn_step < k_val:
                st.session_state.knn_step += 1

        if st.button("3. 초기화", use_container_width=True):
            st.session_state.knn_new_point = None
            st.session_state.knn_step = 0
            st.rerun()

        # 결과 텍스트 표시
        if st.session_state.knn_new_point is not None:
            step = st.session_state.knn_step
            if step > 0:
                current_label = y_knn[st.session_state.knn_sorted_idx[step-1]]
                st.info(f"**{step}번째 이웃:** {current_label}")
            
            if step == k_val:
                final_neighbors = y_knn[st.session_state.knn_sorted_idx[:k_val]]
                u, c = np.unique(final_neighbors, return_counts=True)
                winner = u[np.argmax(c)]
                st.success(f"### 🎯 결과: {winner}")

    with chart_col:
        # 차트 옵션 생성
        option = build_chart_options(k_val)
        st_echarts(option, height="650px", key="knn_chart")

def build_chart_options(k_val):
    series = []
    # 배경 데이터 (152개)
    for label in np.unique(y_knn):
        mask = (y_knn == label)
        series.append({
            "name": str(label), "type": "scatter", "symbolSize": 9,
            "data": X_knn[mask].tolist(),
            "itemStyle": {"color": COLOR_MAP[label], "opacity": 0.4}
        })

    if st.session_state.knn_new_point is not None:
        p = st.session_state.knn_new_point.tolist()
        # 새 데이터 포인트
        series.append({
            "name": "새 데이터", "type": "scatter", "symbol": "triangle", "symbolSize": 22,
            "data": [p], "itemStyle": {"color": COLOR_MAP["새 데이터"]}, "z": 10
        })

        # 이웃 연결 강조
        for i in range(st.session_state.knn_step):
            idx = st.session_state.knn_sorted_idx[i]
            neighbor = X_knn[idx].tolist()
            series.append({
                "type": "line", "data": [p, neighbor],
                "lineStyle": {"type": "dashed", "width": 1.5, "color": "#444"}, "z": 5
            })
            series.append({
                "type": "scatter", "symbolSize": 20, "data": [neighbor],
                "itemStyle": {"color": "none", "borderColor": COLOR_MAP[y_knn[idx]], "borderWidth":
