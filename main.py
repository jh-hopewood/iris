import streamlit as st
import numpy as np
import pandas as pd
import os
from streamlit_echarts import st_echarts

st.set_page_config(page_title="KNN 152 데이터 시뮬레이터", layout="wide")

# ============================================================
# 1. 데이터 로드 (오타 방지를 위해 iloc 사용)
# ============================================================
DATA_FILE = "knn_data.csv"

@st.cache_data
def load_data():
    if os.path.exists(DATA_FILE):
        # 한글 깨짐 방지를 위해 utf-8-sig 사용
        df = pd.read_csv(DATA_FILE, encoding='utf-8-sig')
        try:
            # 첫 번째, 두 번째 컬럼을 X로, 마지막 컬럼을 y로 선택
            X = df.iloc[:, [0, 1]].values.astype(float)
            y = df.iloc[:, -1].values
            cols = df.columns.tolist()
            return X, y, cols[0], cols[1]
        except Exception as e:
            st.error(f"데이터 형식 오류: {e}")
            return None, None, None, None
    return None, None, None, None

X_knn, y_knn, x_label, y_label = load_data()

# ============================================================
# 2. 상태 및 색상 설정
# ============================================================
if "knn_step" not in st.session_state:
    st.session_state.update({
        "knn_new_point": None, "knn_step": 0, "knn_distances": None,
        "knn_sorted_idx": None, "knn_final_label": None
    })

# 동적 색상 할당 (라벨이 무엇이든 대응 가능)
if y_knn is not None:
    unique_labels = np.unique(y_knn)
    # 기본 색상 팔레트
    palette = ["#e74c3c", "#3498db", "#f1c40f", "#9b59b6", "#e67e22"]
    COLOR_MAP = {label: palette[i % len(palette)] for i, label in enumerate(unique_labels)}
    COLOR_MAP["새 데이터"] = "#2ecc71"

# ============================================================
# 3. 로직 함수
# ============================================================
def prepare_knn(new_point, k):
    st.session_state.knn_step = 0
    st.session_state.knn_new_point = np.array(new_point, dtype=float)
    dists = np.linalg.norm(X_knn - st.session_state.knn_new_point, axis=1)
    st.session_state.knn_distances = dists
    st.session_state.knn_sorted_idx = np.argsort(dists)
    
    # 예측
    neighbors = y_knn[st.session_state.knn_sorted_idx[:k]]
    u, c = np.unique(neighbors, return_counts=True)
    st.session_state.knn_final_label = u[np.argmax(c)]

def build_option(k):
    if X_knn is None: return {}
    
    # 축 범위 자동 설정
    x_min, x_max = X_knn[:, 0].min() - 0.5, X_knn[:, 0].max() + 0.5
    y_min, y_max = X_knn[:, 1].min() - 0.5, X_knn[:, 1].max() + 0.5

    series = []
    # 그룹별로 점 찍기
    for label in np.unique(y_knn):
        mask = (y_knn == label)
        series.append({
            "name": str(label), "type": "scatter", "symbolSize": 10,
            "data": X_knn[mask].tolist(),
            "itemStyle": {"color": COLOR_MAP[label]}
        })

    # 새 데이터 및 이웃 연결 선
    if st.session_state.knn_new_point is not None:
        p = st.session_state.knn_new_point
        series.append({
            "name": "새 데이터", "type": "scatter", "symbol": "triangle", "symbolSize": 15,
            "data": [p.tolist()], "itemStyle": {"color": COLOR_MAP["새 데이터"]}
        })
        
        step = st.session_state.knn_step
        if step > 0:
            for i in range(min(step, k)):
                idx = st.session_state.knn_sorted_idx[i]
                neighbor = X_knn[idx]
                series.append({
                    "type": "line", "symbol": "none", "lineStyle": {"type": "dashed", "width": 1, "color": "#999"},
                    "data": [p.tolist(), neighbor.tolist()]
                })

    return {
        "legend": {"bottom": 10},
        "tooltip": {"trigger": "item"},
        "xAxis": {"type": "value", "min": round(x_min, 1), "max": round(x_max, 1), "name": x_label},
        "yAxis": {"type": "value", "min": round(y_min, 1), "max": round(y_max, 1), "name": y_label},
        "series": series
    }

# ============================================================
# 4. 화면 구성
# ============================================================
st.title("KNN 데이터 시각화 (152개 샘플)")

if X_knn is not None:
    st.sidebar.info(f"📊 총 {len(X_knn)}개의 데이터를 성공적으로 불러왔습니다.")
    
    with st.form("input_form"):
        col1, col2, col3 = st.columns(3)
        # 데이터 범위에 맞춰 기본값 설정
        x_mid = float(np.median(X_knn[:, 0]))
        y_mid = float(np.median(X_knn[:, 1]))
        
        new_x = col1.number_input(f"새 {x_label}", value=x_mid)
        new_y = col2.number_input(f"새 {y_label}", value=y_mid)
        k_val = col3.slider("K (이웃 수)", 1, 15, 3)
        
        c1, c2, c3 = st.columns(3)
        if c1.form_submit_button("데이터 투입"): prepare_knn([new_x, new_y], k_val)
        if c2.form_submit_button("다음 이웃 찾기"):
            if st.session_state.knn_new_point is not None: st.session_state.knn_step += 1
        if c3.form_submit_button("초기화"):
            for key in ["knn_new_point", "knn_step"]: st.session_state[key] = None
            st.rerun()

    if st.session_state.knn_final_label:
        st.write(f"### 🎯 분류 결과: **{st.session_state.knn_final_label}**")

    st_echarts(build_option(k_val), height="600px")
else:
    st.error(f"'{DATA_FILE}' 파일이 없습니다. 코드가 있는 폴더에 파일을 넣어주세요.")
