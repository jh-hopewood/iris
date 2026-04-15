import streamlit as st
import numpy as np
import pandas as pd
import os
from streamlit_echarts import st_echarts

# 1. 데이터 로드 (이전과 동일)
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

if "knn_step" not in st.session_state:
    st.session_state.update({"knn_new_point": None, "knn_step": 0, "knn_sorted_idx": None, "knn_distances": None})

# 색상 맵 설정
if y_knn is not None:
    unique_labels = np.unique(y_knn)
    palette = ["#e74c3c", "#3498db", "#f1c40f", "#9b59b6", "#e67e22"]
    COLOR_MAP = {label: palette[i % len(palette)] for i, label in enumerate(unique_labels)}
    COLOR_MAP["새 데이터"] = "#2ecc71"

def build_knn_chart(k_val):
    if X_knn is None: return {}
    
    # 152개 전체 데이터 시리즈 구성
    series = []
    for label in np.unique(y_knn):
        mask = (y_knn == label)
        series.append({
            "name": str(label), "type": "scatter", "symbolSize": 8,
            "data": X_knn[mask].tolist(),
            "itemStyle": {"color": COLOR_MAP[label], "opacity": 0.6}
        })

    # 새 데이터 (삼각형)
    if st.session_state.knn_new_point is not None:
        p = st.session_state.knn_new_point.tolist()
        series.append({
            "name": "새 데이터", "type": "scatter", "symbol": "triangle", "symbolSize": 20,
            "data": [p], "itemStyle": {"color": COLOR_MAP["새 데이터"]}, "z": 10
        })

        # 선택된 이웃들 강조 (선 및 외곽선)
        step = st.session_state.knn_step
        if step > 0:
            for i in range(min(step, k_val)):
                idx = st.session_state.knn_sorted_idx[i]
                neighbor = X_knn[idx].tolist()
                label_name = y_knn[idx]
                
                # 1. 새 데이터와 이웃을 잇는 점선
                series.append({
                    "type": "line", "symbol": "none",
                    "data": [p, neighbor],
                    "lineStyle": {"type": "dashed", "width": 1.5, "color": "#555"},
                    "z": 5
                })
                
                # 2. 선택된 샘플 위에 큰 강조 원 그리기
                series.append({
                    "type": "scatter", "symbolSize": 20,
                    "data": [neighbor],
                    "itemStyle": {
                        "color": "none",
                        "borderColor": COLOR_MAP[label_name],
                        "borderWidth": 3
                    },
                    "label": {"show": True, "formatter": f"{i+1}NN", "position": "top", "color": "#000"},
                    "z": 10
                })

    return {
        "title": {"text": f"KNN 시각화 (현재 이웃: {st.session_state.knn_step}개)", "left": "center"},
        "tooltip": {"trigger": "item"},
        "legend": {"bottom": 5},
        "xAxis": {"type": "value", "scale": True, "name": x_label},
        "yAxis": {"type": "value", "scale": True, "name": y_label},
        "series": series,
        "animationDurationUpdate": 500 # 단계별 변화 시 부드러운 애니메이션
    }

# --- UI 레이아웃 ---
st.title("KNN 탐색 애니메이션")

if X_knn is not None:
    with st.sidebar:
        st.write("### 설정")
        new_x = st.number_input(f"{x_label}", value=float(np.mean(X_knn[:,0])))
        new_y = st.number_input(f"{y_label}", value=float(np.mean(X_knn[:,1])))
        k_val = st.slider("최종 확인 이웃 수(K)", 1, 15, 3)
        
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

    # 결과 메시지 출력
    if st.session_state.knn_new_point is not None:
        step = st.session_state.knn_step
        if step > 0:
            current_idx = st.session_state.knn_sorted_idx[step-1]
            current_label = y_knn[current_idx]
            st.info(f"📍 {step}번째로 가까운 이웃 탐색 완료: **{current_label}** (샘플 번호: {current_idx})")
        
        if step == k_val:
            final_neighbors = y_knn[st.session_state.knn_sorted_idx[:k_val]]
            unique, counts = np.unique(final_neighbors, return_counts=True)
            winner = unique[np.argmax(counts)]
            st.success(f"### 🎯 최종 판정 결과: **{winner}**")

    # 차트 출력
    st_echarts(build_knn_chart(k_val), height="600px")
