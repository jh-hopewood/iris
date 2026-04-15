import streamlit as st
import numpy as np
import pandas as pd
import os
from streamlit_echarts import st_echarts

# 데이터 로드 및 초기 설정 (위의 코드와 동일)
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

# 상태 관리
if "knn_step" not in st.session_state:
    st.session_state.update({"knn_new_point": None, "knn_step": 0})

# --- 화면 구성 ---
st.title("KNN 분류 판정 시뮬레이터")

if X_knn is not None:
    # 사이드바에 데이터 정보 요약
    st.sidebar.header("📊 데이터 요약")
    st.sidebar.write(f"전체 샘플 수: {len(X_knn)}개")
    unique_labels = np.unique(y_knn)
    for l in unique_labels:
        st.sidebar.write(f"- {l}: {list(y_knn).count(l)}개")

    # 입력 폼
    with st.form("knn_form"):
        c1, c2, c3 = st.columns(3)
        new_x = c1.number_input(f"{x_label}", value=float(np.mean(X_knn[:,0])))
        new_y = c2.number_input(f"{y_label}", value=float(np.mean(X_knn[:,1])))
        k_val = c3.slider("K값 (이웃 수)", 1, 15, 3)
        
        b1, b2, b3 = st.columns(3)
        if b1.form_submit_button("1. 데이터 투입"):
            st.session_state.knn_new_point = np.array([new_x, new_y])
            st.session_state.knn_step = 0
            # 거리 계산 및 정렬
            dists = np.linalg.norm(X_knn - st.session_state.knn_new_point, axis=1)
            st.session_state.knn_sorted_idx = np.argsort(dists)
            st.session_state.knn_distances = dists

        if b2.form_submit_button("2. 다음 이웃 찾기"):
            if st.session_state.knn_step < k_val:
                st.session_state.knn_step += 1
        
        if b3.form_submit_button("3. 초기화"):
            st.session_state.knn_new_point = None
            st.session_state.knn_step = 0
            st.rerun()

    # --- 결과 분석 섹션 ---
    if st.session_state.knn_new_point is not None:
        step = st.session_state.knn_step
        indices = st.session_state.knn_sorted_idx[:step]
        
        col_res1, col_res2 = st.columns([1, 1])
        
        with col_res1:
            st.write(f"### 🔍 탐색 중 (K={step}/{k_val})")
            if step > 0:
                # 현재 선택된 이웃들 정보 표로 보여주기
                neighbor_data = []
                for i in range(step):
                    idx = st.session_state.knn_sorted_idx[i]
                    neighbor_data.append({
                        "순위": i+1,
                        "샘플번호": idx,
                        "품종": y_knn[idx],
                        "거리": round(st.session_state.knn_distances[idx], 3)
                    })
                st.table(neighbor_data)
        
        with col_res2:
            if step == k_val:
                # 최종 결과 계산
                final_neighbors = y_knn[indices]
                unique, counts = np.unique(final_neighbors, return_counts=True)
                result_dict = dict(zip(unique, counts))
                winner = unique[np.argmax(counts)]
                
                st.success(f"### 🎯 최종 판정: **[{winner}]**")
                st.write("**투표 결과:**")
                for label, count in result_dict.items():
                    st.write(f"- {label}: {count}표")
            else:
                st.info(" '다음 이웃 찾기' 버튼을 눌러 가장 가까운 샘플을 하나씩 확인하세요.")

        # 시각화 (기존 build_option 로직 실행)
        # ... (생략된 시각화 코드는 이전과 동일)
