def build_chart_options(k_val):
    series = []
    
    # 1. 배경 데이터 샘플들
    for label in np.unique(y_knn):
        mask = (y_knn == label)
        series.append({
            "name": str(label),
            "type": "scatter",
            "symbolSize": 10,
            "data": X_knn[mask].tolist(),
            "itemStyle": {"color": COLOR_MAP[label], "opacity": 0.5}
        })

    # 2. 새 데이터 및 탐색 단계 표시
    if st.session_state.knn_new_point is not None:
        p = st.session_state.knn_new_point.tolist()
        
        # 새 데이터 포인트
        series.append({
            "name": "새 데이터",
            "type": "scatter",
            "symbol": "triangle",
            "symbolSize": 20,
            "data": [p],
            "itemStyle": {"color": COLOR_MAP["새 데이터"]},
            "z": 10
        })

        # 다음 찾기 단계별 강조
        for i in range(st.session_state.knn_step):
            idx = st.session_state.knn_sorted_idx[i]
            neighbor = X_knn[idx].tolist()
            
            # 연결 선
            series.append({
                "type": "line",
                "data": [p, neighbor],
                "symbol": "none",
                "lineStyle": {"type": "dashed", "width": 1.5, "color": "#444"},
                "z": 5
            })
            
            # 선택된 이웃 강조 (이 부분의 중괄호 개수를 확인하세요)
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
                    "position": "top"
                },
                "z": 11
            })

    # 최종 반환문 (에러가 발생했던 지점)
    return {
        "tooltip": {"trigger": "item"},
        "legend": {"top": "5%", "data": list(COLOR_MAP.keys())},
        "xAxis": {"type": "value", "scale": True, "name": x_label},
        "yAxis": {"type": "value", "scale": True, "name": y_label},
        "series": series
    }
