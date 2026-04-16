import streamlit as st
import numpy as np
import pandas as pd
import os
from streamlit_echarts import st_echarts

# 1. 페이지 설정
st.set_page_config(page_title="KNN 시뮬레이터", layout="wide")

# 2. 데이터 로드 함수 (캐싱 적용)
DATA_FILE = "knn_data.csv"

@st.cache_data
def load_data():
    if os.path.exists(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE, encoding='utf-8-sig')
            # 컬럼명에 상관없이 첫 두 열을 X, 마지막 열을 y로 가져옴
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
    st.session_state.update({
        "knn_new_point": None, 
        "knn_step": 0, 
        "knn_sorted_idx": None, 
        "knn_distances": None
    })

# 4. 색상 맵 설정
COLOR_MAP = {}
if y_knn is not None:
    unique_labels = np.unique(y_knn)
    palette = ["#e74c3c", "#3498db", "#f1c40f", "#9b59b6", "#e67e22"]
    COLOR_MAP = {label: palette[i % len(palette)] for i, label in enumerate(unique_labels)}
    COLOR_MAP["새 데이터"] = "#2ecc71"

# 5. ECharts 옵션 생성 (괄호 구조 정밀 검수 완료)
def build_chart_options(k_val):
    series = []
    
    # 배경 데이터 (학습 데이터 152개)
    for label in np.unique(y_knn):
        mask = (y_knn == label)
        series.append({
            "name": str(label),
            "type": "scatter",
            "symbolSize": 10,
            "data": X_knn[mask].tolist(),
            "itemStyle": {"color": COLOR_MAP[label], "opacity": 0.4}
        })

    # 새 데이터 투입 시 시각화 로직
    if st.session_state.knn_new_point is not None:
        p = st.session_state.knn_new_point.tolist()
        
        # 새 데이터 포인트
        series.append({
            "name": "새 데이터",
            "type": "scatter",
            "symbol": "triangle",
            "symbolSize": 22,
            "data": [p],
            "itemStyle": {"color": COLOR_MAP["새 데이터"]},
            "z": 10
        })

        # 탐색 단계별 강조
        for i in range(st.session_state.knn_step):
            idx = st.session_state.knn_sorted_idx[i]
            neighbor = X_knn[idx].tolist()
            
            # 1순위: 이웃 연결 점선
            series.append({
                "type": "line",
                "data": [p, neighbor],
                "symbol": "none",
                "lineStyle": {"type": "dashed", "width": 1.5, "color": "#555"},
                "z": 5
            })
            
            # 2순위: 선택된 이웃 강조 원
            series.append({
                "type": "scatter",
                "symbolSize": 24,
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
                    "fontWeight": "bold"
                },
                "z": 11
            })

    # 최종 옵션 딕셔너리 반환
    chart_dict = {
        "tooltip": {"trigger": "item"},
        "legend": {"top": "5%", "data": list(COLOR_MAP.keys())},
        "xAxis": {"type": "value", "scale": True, "name": x_label},
        "yAxis": {"type": "value", "scale": True, "name": y_label},
        "series": series
    }
    return chart_dict

# 6. 메인 시뮬레이터 패널 (@st.fragment로 깜박임 제거)
@st.fragment
def run_knn_app():
    st.title("🔬 KNN 알고리즘 시각화 도구 (152개 샘플)")
    
    col_ctrl, col_chart = st.columns([1, 3])

    with col_ctrl:
        st.subheader("⚙️ 컨트롤러")
        
        # 초기값 설정 (중앙값)
        x_val = st.number_input(f"입력 {x_label}", value=float(np.median(X_knn[:, 0])))
        y_val = st.number_input(f"입력 {y_label}", value=float(np.median(X_knn[:, 1])))
        k_val = st.slider("K (이웃 수)", 1, 15, 3)

        st.divider()
        
        if st.button("📍 데이터 투입", use_container_width=True):
            st.session_state.knn_new_point = np.array([x_val, y_val])
            st.session_state.knn_step = 0
            dists = np.linalg.norm(X_knn - st.session_state.knn_new_point, axis=1)
            st.session_state.knn_sorted_idx = np.argsort(dists)
            st.session_state.knn_distances = dists

        if st.button("🔍 다음 찾기 (Step)", use_container_width=True):
            if st.session_state.knn_new_point is not None:
                if st.session_state.knn_step < k_val:
                    st.session_state.knn_step += 1
                else:
                    st.warning("K개의 이웃을 모두 찾았습니다.")
            else:
                st.error("데이터를 먼저 투입하세요.")

        if st.button("🔄 초기화", use_container_width=True):
            st.session_state.knn_new_point = None
            st.session_state.knn_step = 0
            st.rerun()

        st.divider()
        
        # 실시간 판정 리포트
        if st.session_state.knn_new_point is not None:
            step = st.session_state.knn_step
            if step > 0:
                cur_label = y_knn[st.session_state.knn_sorted_idx[step-1]]
                st.write(f"현재 탐색된 이웃: **{cur_label}**")
            
            if step == k_val:
                final_votes = y_knn[st.session_state.knn_sorted_idx[:k_val]]
                unique, counts = np.unique(final_votes, return_counts=True)
                winner = unique[np.argmax(counts)]
                st.success(f"### 최종 판정: {winner}")

    with col_chart:
        options = build_chart_options(k_val)
        st_echarts(options=options, height="700px", key="knn_chart_component")

# 7. 실행부
if X_knn is not None:
    run_knn_app()
else:
    st.error("같은 폴더 내 'knn_data.csv' 파일을 확인해 주세요.")
