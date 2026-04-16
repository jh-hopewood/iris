import streamlit as st
import numpy as np
import pandas as pd
import os
from streamlit_echarts import st_echarts

# 1. 페이지 설정
st.set_page_config(page_title="KNN 실시간 투표 시뮬레이터", layout="wide")

# 2. 데이터 로드
DATA_FILE = "knn_data.csv"
@st.cache_data
def load_data():
    if os.path.exists(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE, encoding='utf-8-sig')
            X = df.iloc[:, [0, 1]].values.astype(float)
            y = df.iloc[:, -1].values
            return X, y, df.columns[0], df.columns[1]
        except Exception as e:
            st.error(f"데이터 로드 실패: {e}")
            return None, None, None, None
    return None, None, None, None

X_knn, y_knn, x_label, y_label = load_data()

# 3. 세션 상태 관리
if "knn_step" not in st.session_state:
    st.session_state.update({"knn_new_point": None, "knn_step": 0, "knn_sorted_idx": None, "knn_distances": None})

# 4. 색상 설정
COLOR_MAP = {}
if y_knn is not None:
    unique_labels = np.unique(y_knn)
    palette = ["#e74c3c", "#3498db", "#f1c40f", "#9b59b6", "#e67e22"]
    COLOR_MAP = {label: palette[i % len(palette)] for i, label in enumerate(unique_labels)}
    COLOR_MAP["새 데이터"] = "#2ecc71"

# 5. 차트 옵션
def build_chart_options(k_val):
    series = []
    for label in np.unique(y_knn):
        mask = (y_knn == label)
        series.append({
            "name": str(label), "type": "scatter", "symbolSize": 10,
            "data": X_knn[mask].tolist(),
            "itemStyle": {"color": COLOR_MAP[label], "opacity": 0.4}
        })
    if st.session_state.knn_new_point is not None:
        p = st.session_state.knn_new_point.tolist()
        series.append({
            "name": "새 데이터", "type": "scatter", "symbol": "triangle", "symbolSize": 22,
            "data": [p], "itemStyle": {"color": COLOR_MAP["새 데이터"]}, "z": 10
        })
        for i in range(st.session_state.knn_step):
            idx = st.session_state.knn_sorted_idx[i]
            neighbor = X_knn[idx].tolist()
            series.append({
                "type": "line", "data": [p, neighbor], "symbol": "none",
                "lineStyle": {"type": "dashed", "width": 1.5, "color": "#555"}, "z": 5
            })
            series.append({
                "type": "scatter", "symbolSize": 24, "data": [neighbor],
                "itemStyle": {"color": "none", "borderColor": COLOR_MAP[y_knn[idx]], "borderWidth": 3},
                "label": {"show": True, "formatter": f"{i+1}NN", "position": "top"}, "z": 11
            })
    return {"tooltip": {"trigger": "item"}, "legend": {"top": "5%"}, "xAxis": {"type": "value", "scale": True}, "yAxis": {"type": "value", "scale": True}, "series": series}

# 6. 메인 패널
@st.fragment
def run_knn_app():
    st.title("🔬 KNN 실시간 투표 시뮬레이터")
    col_ctrl, col_chart = st.columns([1, 3])

    with col_ctrl:
        st.subheader("⚙️ 컨트롤러")
        x_val = st.number_input(f"{x_label}", value=float(np.median(X_knn[:, 0])))
        y_val = st.number_input(f"{y_label}", value=float(np.median(X_knn[:, 1])))
        k_val = st.slider("K (이웃 수)", 1, 15, 3)

        if st.button("📍 데이터 투입", use_container_width=True):
            st.session_state.knn_new_point = np.array([x_val, y_val])
            st.session_state.knn_step = 0
            dists = np.linalg.norm(X_knn - st.session_state.knn_new_point, axis=1)
            st.session_state.knn_sorted_idx = np.argsort(dists)
            st.session_state.knn_distances = dists

        if st.button("🔍 다음 찾기 (Step)", use_container_width=True):
            if st.session_state.knn_new_point is not None and st.session_state.knn_step < k_val:
                st.session_state.knn_step += 1

        if st.button("🔄 초기화", use_container_width=True):
            st.session_state.knn_new_point = None
            st.session_state.knn_step = 0
            st.rerun()

        st.divider()
        
        # --- 실시간 투표 현황판 ---
        if st.session_state.knn_new_point is not None:
            step = st.session_state.knn_step
            if step > 0:
                current_indices = st.session_state.knn_sorted_idx[:step]
                found_labels = y_knn[current_indices]
                unique, counts = np.unique(found_labels, return_counts=True)
                vote_status = dict(zip(unique, counts))
                
                st.write(f"#### 📊 투표 현황 ({step}/{k_val})")
                for label in unique_labels: # 모든 라벨 표시 (0개라도 표시)
                    count = vote_status.get(label, 0)
                    color = COLOR_MAP.get(label, "#333")
                    st.markdown(f"- <span style='color:{color}; font-weight:bold;'>{label}</span>: **{count}**표", unsafe_allow_html=True)
            
            if step == k_val:
                winner = unique[np.argmax(counts)]
                st.markdown("---")
                st.success(f"### 🏆 최종 판정: {winner}")

    with col_chart:
        st_echarts(options=build_chart_options(k_val), height="700px", key="knn_v2")

if X_knn is not None:
    run_knn_app()
