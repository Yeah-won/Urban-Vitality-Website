import streamlit as st
from streamlit_folium import st_folium
import folium
import geopandas as gpd
from folium.features import GeoJsonTooltip
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import subprocess
import matplotlib as mpl
import io
import numpy as np
import json
import numpy as np
from pytorch_tabnet.tab_model import TabNetRegressor
import torch

# 나눔고딕 설치 (Streamlit Cloud에서 리눅스 명령 실행)
subprocess.run(["apt-get", "install", "-y", "fonts-nanum"], check=True)
subprocess.run(["fc-cache", "-fv"], check=True)

# 폰트 경로 지정 후 설정
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
font_name = fm.FontProperties(fname=font_path).get_name()

plt.rcParams['font.family'] = font_name
mpl.rcParams['axes.unicode_minus'] = False

@st.cache_resource
def load_tabnet_model():
    """TabNet 모델과 관련 정보를 로드"""
    try:
        # 모델 파일 경로 설정
        model_path = "tabnet_model.zip"
        feature_names_path = "feature_names.json"
        
        # 특성 이름 로드
        with open(feature_names_path, 'r', encoding='utf-8') as f:
            feature_info = json.load(f)
        feature_names = feature_info['feature_names']
        
        # TabNet 모델 로드
        model = TabNetRegressor()
        model.load_model(model_path)
        
        return model, feature_names
    
    except Exception as e:
        st.error(f"모델 로딩 실패: {e}")
        return None, None

# 앱 시작 시 모델 로드
tabnet_model, feature_names = load_tabnet_model()

# Feature 지정
FEATURES = ['sub_500m', 'ind_cluste', 'pop', 'worker',
       'road_den', 'RAR', 'BAR', 'area(km2)', 'FAR', 
       'train_dist', 'com_cluste', 'pharmacy', 'hospital', 'restaurant',
       'cafe', 'CVS', 'school', 'center_dis', 'entropy', 'park_area', 'central_du', 'center_dis',
        'agriculture_forest_ratio', 'commercial_ratio', 'environmental_ratio',
       'green_ratio', 'industrial_ratio', 'management_ratio',
       'residential_ratio'
]
# ✅ 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # 또는 'NanumGothic'
mpl.rcParams['axes.unicode_minus'] = False
# ✅ 마크다운 설정
st.markdown("""
<style>
.table-style {
    width: 100%;
    table-layout: fixed;
    border-collapse: collapse;
}
.table-style td {
    padding: 6px 12px;
    border-bottom: 1px solid #ddd;
    word-wrap: break-word;
}
.table-style th {
    text-align: left;
    background-color: #f0f0f0;
    padding: 6px 12px;
}
.tooltip {
  position: relative;
  display: inline-block;
  cursor: help;
  border-bottom: 1px dotted #555;
}
.tooltip .tooltiptext {
  visibility: hidden;
  width: 260px;
  background-color: #444;
  color: #fff;
  text-align: left;
  border-radius: 6px;
  padding: 6px;
  position: absolute;
  z-index: 1;
  top: -5px;
  left: 105%;
  opacity: 0;
  transition: opacity 0.3s;
  white-space: normal;
}
.tooltip:hover .tooltiptext {
  visibility: visible;
  opacity: 1;
}
</style>
""", unsafe_allow_html=True)

# 페이지 설정
st.set_page_config(layout="wide")
st.title("택지개발지구 생활인구 예측 UI")


# 1. 세션 상태 초기화 부분에 추가 (기존 초기화 코드들 다음에)
if "selected_index" not in st.session_state:
    st.session_state.selected_index = 367
if "map_center" not in st.session_state:
    st.session_state.map_center = None
if "map_zoom" not in st.session_state:
    st.session_state.map_zoom = None
if "result_df" not in st.session_state:
    st.session_state.result_df = None
if "predicted_vitality" not in st.session_state:  # 추가
    st.session_state.predicted_vitality = None
if "need_prediction" not in st.session_state:     # 추가
    st.session_state.need_prediction = False

# 새로운 7개 비율에 대한 세션 상태 초기화
ratio_keys = ['agriculture_forest_ratio', 'commercial_ratio', 'environmental_ratio',
              'green_ratio', 'industrial_ratio', 'management_ratio', 'residential_ratio']
for key in ratio_keys:
    if key not in st.session_state:
        st.session_state[key] = 0.0

# ✅ 데이터 불러오기
# ✅ shp 파일 불러온 직후
gdf = gpd.read_file("data.gpkg", layer="data", encoding="utf-8").to_crs(epsg=4326)
gdf["geometry"] = gdf["geometry"].simplify(0.0005, preserve_topology=True)

# ✅ 고정 BE 비율 컬럼 처리
be_ratio_cols = ["주거용지_구성비", "상업용지_구성비", "산업시설_구성비","기반시설용지_구성비", "관광휴양용지_구성비"]
for col in be_ratio_cols:
    gdf[col] = pd.to_numeric(gdf[col], errors='coerce').fillna(0.0).astype(float)
# ✅ 사업단계/지구 선택
district_col = "zoneName"
district_code = "zoneCode"
step_col = "사업단계_단순화"
area_col = "area(km2)"
with st.sidebar:
    st.markdown("### 1️⃣ 사업단계를 선택하세요")
    step_list = gdf[step_col].dropna().unique().tolist()
    step_list = ["전체"] + sorted(step_list)
    selected_step = st.selectbox("사업단계", step_list, index=0)

    if selected_step == "전체":
        filtered_gdf = gdf
    else:
        filtered_gdf = gdf[gdf[step_col] == selected_step]

    district_names = filtered_gdf[district_col].tolist()
    default_name = "성남분당"
    default_selected = default_name if default_name in district_names else district_names[0]

    st.markdown("### 2️⃣ 지구명을 검색하세요")
    search_query = st.text_input(" 지구명 검색", value="", placeholder="예: 판교")
    filtered_names = [name for name in district_names if search_query.lower() in name.lower()]

    if not filtered_names:
        st.warning("검색 결과가 없습니다.")
        selected_name = st.selectbox("택지 선택", district_names, index=district_names.index(default_selected))
    else:
        default_index = filtered_names.index(default_selected) if default_selected in filtered_names else 0
        selected_name = st.selectbox("택지 선택", filtered_names, index=default_index)

# ✅ 선택 인덱스 및 폴리곤
selected_index = gdf[gdf[district_col] == selected_name].index[0]

# ✅ 택지 변경 시 상태 초기화
if st.session_state.get("last_index", -1) != selected_index:
    st.session_state.selected_index = selected_index
    st.session_state.map_center = None
    st.session_state.result_df = None
    st.session_state.predicted_vitality = None  # 추가
    st.session_state.need_prediction = False    # 추가

    selected_row = gdf.loc[selected_index]
    selected_poly = selected_row.geometry
    center = selected_poly.centroid
    area = selected_row.get("area(km2)", 1)

    # 🔥 sub_500m은 이미 절대값이므로 그대로 사용, 나머지는 밀도×면적
    st.session_state["sub_500m"] = int(round(selected_row.get("sub_500m", 0)))  # 절대값 그대로

    facility_cols = ["pharmacy", "hospital", "restaurant", "cafe", "CVS", "school", "pop", "worker"]
    for col in facility_cols:
        st.session_state[col] = int(round(selected_row.get(col, 0) * area))
    
    # 새로운 7개 비율 초기화 (0~1 범위의 값을 0~100으로 변환)
    for ratio_key in ratio_keys:
        st.session_state[ratio_key] = int(round(selected_row.get(ratio_key, 0) * 100))

    st.session_state.map_center = [center.y, center.x]
    st.session_state["last_index"] = selected_index
else:
    selected_row = gdf.loc[st.session_state.selected_index]
    selected_poly = selected_row.geometry
    center = selected_poly.centroid

# ✅ 지도 생성
if st.session_state.map_zoom is None:
    st.session_state.map_zoom = 14

m = folium.Map(location=st.session_state.map_center, zoom_start=st.session_state.map_zoom)

buffer = center.buffer(0.01)
display_gdf = gdf[gdf.intersects(buffer)]

folium.GeoJson(
    selected_poly,
    name="선택지구",
    style_function=lambda x: {"color": "black", "weight": 4, "fillOpacity": 0.1}
).add_to(m)

# ✅ 주변 택지지구 GeoJson 표시
if not display_gdf.empty:
    folium.GeoJson(
        display_gdf,
        name="주변택지지구",
        tooltip=GeoJsonTooltip(
            fields=["이름","사업단계_단순화", "area(km2)", "계획인구"],
            aliases=["지구명: ","사업단계: ", "면적(km²): ", "계획인구(명): "],
            localize=True,
            labels=True,
            sticky=True
        ),
        style_function=lambda x: {"color": "blue", "weight": 1, "fillOpacity": 0.2}
    ).add_to(m)


from math import radians, cos, sin, asin, sqrt

def haversine(lat1, lon1, lat2, lon2):
    # 지구 반지름 (km)
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return R * 2 * asin(sqrt(a))


def add_cluster_markers(csv_path, color, name, center_lat, center_lon, radius_km=10):
    df = pd.read_csv(csv_path, encoding="utf-8")
    for _, row in df.iterrows():
        cluster_id = int(row["cluster"])
        x, y = row["center_x"], row["center_y"]
        size = row["size"]

        # 거리 계산
        dist = haversine(center_lat, center_lon, y, x)
        if dist > radius_km:
            continue  # 5km 넘는 건 무시

        marker_radius = np.log(size + 1) * 3
        folium.CircleMarker(
            location=[y, x],
            radius=marker_radius,
            weight=1,                 # ✅ 테두리 두께 줄이기 (기본은 3)
            opacity=0.5,   
            
            color=color,
            fill=True,
            fill_opacity=0.3,
            tooltip=f"{name} 클러스터 {cluster_id}<br>Size: {int(size)}"
        ).add_to(m)

add_cluster_markers(
    "industry_cluster_centers.csv",
    color="green",
    name="산업",
    center_lat=center.y,
    center_lon=center.x
)
add_cluster_markers(
    "restaurant_cluster_centers.csv",
    color="red",
    name="음식점",
    center_lat=center.y,
    center_lon=center.x
)

# ✅ 지도 + 입력폼 + 고정BE값
col1, col2, col3 = st.columns([4, 2, 2])

# 👉 col1: 지도
with col1:
    st_folium(m, width=1500, height=1060)

# 👉 col2: 고정된 BE값
with col2:
    st.subheader("계획 용지 비율")

    # ✅ 파이차트 데이터 정의
    pie_labels = ["주거용지", "상업용지", "산업용지", "기반시설용지", "관광휴양용지"]
    pie_keys = ["주거용지_구성비", "상업용지_구성비", "산업시설_구성비", "기반시설용지_구성비", "관광휴양용지_구성비"]
    pie_values = [selected_row.get(key, 0) or 0 for key in pie_keys]

    # ✅ 0인 항목 제거
    pie_data = [(label, value) for label, value in zip(pie_labels, pie_values) if value > 0]

    if pie_data:
        labels, values = zip(*pie_data)

        fig, ax = plt.subplots(figsize=(3.0, 3.0))
        ax.pie(
            values,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 9}
        )
        ax.axis('equal')

        # ✅ 고해상도 저장
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
        buf.seek(0)

         # ✅ Streamlit 출력
        st.image(buf, use_container_width=True)
    else:
        st.info("표시할 용지 구성비가 없습니다.")

    # 지표 테이블
    st.subheader("고정 변수")
    # 스타일 넣기
    # ✅ 마우스 오버 툴팁 + CSS로 테이블 너비 조정
            
    # 설명 툴팁 정의
    tooltips = {
        "인구수": "인구수",
        "종사자수": "종사자수",        
        "FAR": "용적률 = 연면적 / 대지면적",
        "BAR": "건폐율 = 건축면적 / 대지면적",
        "RAR": "주거비율 = 주거 연면적 / 전체 연면적",
        "Road density": "단위 면적당 도로 길이 (m/km²)",
        "산업클러스터 거리 점수": "5km 반경 내 Σ(클러스터 크기 ÷ 거리)<br><span style='font-size:11px;'>※ 거리 하한값: 500m</span>",
        "상업클러스터 거리 점수": "5km 반경 내 Σ(클러스터 크기 ÷ 거리)<br><span style='font-size:11px;'>※ 거리 하한값: 500m</span>"
    }
        
    # 지표 값
    indicators = {
        "인구수": st.session_state["pop"],
        "종사자수": st.session_state["worker"],
        "FAR": selected_row.get("FAR", 0),
        "BAR": selected_row.get("BAR", 0),
        "RAR": selected_row.get("RAR", 0),
        "Road density": selected_row.get("road_den", 0),
        "산업클러스터 거리 점수": selected_row.get("ind_cluste", 0),
        "상업클러스터 거리 점수": selected_row.get("com_cluste", 0)
    }

    
    # 툴팁 span 생성
    def tooltip_cell(label, explanation):
        return f'''
        <div class="tooltip">{label}
          <div class="tooltiptext">{explanation}</div>
        </div>
        '''
     
    # HTML 테이블 수동 생성 (꽉 차게)
    table_html = '<table class="table-style">'
    table_html += '<tr><th>지표</th><th>값</th></tr>'
    for label, value in indicators.items():
        tooltip_html = tooltip_cell(label, tooltips[label])
        table_html += f'<tr><td>{tooltip_html}</td><td>{value:.4f}</td></tr>'
    table_html += '</table>'
    
    st.markdown(table_html, unsafe_allow_html=True)
    
    # 🔥 현재 도시활력 표시 (하이라이트)
    
    # day, night 컬럼에서 도시활력 값 가져오기
    day_vitality = selected_row.get("day", 0)
    night_vitality = selected_row.get("night", 0)
    
    # Day 도시활력 표시 (노란색 계열)
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #FFD54F 0%, #FFA726 100%);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            color: #333;
            font-weight: bold;
            margin: 10px 0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        ">
            <div style="font-size: 14px; margin-bottom: 5px;">☀️ Day 도시활력(9-16)</div>
            <div style="font-size: 24px; color: #E65100;">{day_vitality:,.0f}</div>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Night 도시활력 표시 (어두운 밤하늘색)
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #1A237E 0%, #3949AB 100%);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            color: white;
            font-weight: bold;
            margin: 10px 0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        ">
            <div style="font-size: 16px; margin-bottom: 5px;">🌙 Night 도시활력(19-6)</div>
            <div style="font-size: 24px; color: #B39DDB;">{night_vitality:,.0f}</div>
        </div>
        """, 
        unsafe_allow_html=True
    )
# 👉 col3: 입력 폼
with col3:
    st.subheader("가변 BE값")
    
    # 새로운 7개 비율 설정
    st.subheader("토지 이용 비율 설정 (%)")
    
    # 비율 이름 매핑 (한글 표시용)
    ratio_labels = {
        'residential_ratio': '🏠 주거',
        'commercial_ratio': '🏢 상업', 
        'industrial_ratio': '🏭 공업',
        'agriculture_forest_ratio': '🌲 농업/산림',
        'environmental_ratio': '🌿 환경',
        'green_ratio': '🌳 녹지',
        'management_ratio': '🏛️ 관리'
    }
    
    # 중요한 변수들 정의
    important_ratios = ['residential_ratio', 'commercial_ratio', 'industrial_ratio']
    
    # 현재 비율들의 합계 계산을 위한 변수들
    ratio_values = {}
    
    # 중요한 비율들을 먼저 표시 (하이라이트)
    st.markdown("#### 🎯 **핵심 토지이용 비율**")
    
    important_cols = st.columns(3)
    for i, key in enumerate(important_ratios):
        with important_cols[i]:
            current_val = st.session_state.get(key, 0)
            
            # 하이라이트된 스타일로 표시
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
                padding: 10px;
                border-radius: 8px;
                border-left: 4px solid #2196F3;
                margin-bottom: 10px;
            ">
                <div style="font-weight: bold; color: #1565C0; margin-bottom: 5px;">
                    {ratio_labels[key]}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            ratio_values[key] = st.slider(
                f"", 
                0, 100, 
                current_val, 
                key=f"{key}_slider",
                help=f"{ratio_labels[key]} 비율 - 도시활력에 큰 영향을 미치는 핵심 변수입니다."
            )
            st.session_state[key] = ratio_values[key]
    
    # 나머지 비율들
    st.markdown("#### 📋 **기타 토지이용 비율**")
    other_ratios = [key for key in ratio_labels.keys() if key not in important_ratios]
    
    other_cols = st.columns(2)
    for i, key in enumerate(other_ratios):
        with other_cols[i % 2]:
            current_val = st.session_state.get(key, 0)
            ratio_values[key] = st.slider(
                ratio_labels[key], 
                0, 100, 
                current_val, 
                key=f"{key}_slider"
            )
            st.session_state[key] = ratio_values[key]
    
    # 현재 비율들의 합계 표시
    total_ratio = sum(ratio_values.values())
    
    # 합계에 따른 색상 변경
    if total_ratio == 100:
        color = "green"
        status = "✅ 완벽!"
    elif 95 <= total_ratio <= 105:
        color = "orange" 
        status = "⚠️ 거의 맞음"
    else:
        color = "red"
        status = "❌ 조정 필요"
    
    st.markdown(f"**총 합계: <span style='color: {color}'>{total_ratio}% ({status})</span>**", unsafe_allow_html=True)
    
    # 각 비율을 막대그래프로 시각화
    if st.checkbox("📊 비율 시각화", value=False):
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # 중요한 변수와 일반 변수 분리
        important_labels = [ratio_labels[key] for key in important_ratios]  
        important_values = [ratio_values[key] for key in important_ratios]
        
        other_keys = [key for key in ratio_labels.keys() if key not in important_ratios]
        other_labels = [ratio_labels[key] for key in other_keys]
        other_values = [ratio_values[key] for key in other_keys]
        
        # 모든 라벨과 값을 합치기 (중요한 것들이 앞에 오도록)
        all_labels = important_labels + other_labels
        all_values = important_values + other_values
        
        # 색상 설정 (중요한 변수는 진한 색, 나머지는 연한 색)
        colors = ['#1976D2', '#1976D2', '#1976D2'] + ['#90CAF9'] * len(other_values)
        
        bars = ax.bar(all_labels, all_values, color=colors)
        ax.set_ylabel('비율 (%)')
        ax.set_title('토지 이용 비율 분포 (파란색: 핵심변수)')
        ax.set_ylim(0, max(100, max(all_values) * 1.1))
        
        # 값 표시
        for bar, value in zip(bars, all_values):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                       f'{value}%', ha='center', va='bottom', fontweight='bold' if value in important_values else 'normal')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
    
    # 시설 개수 입력 폼
    with st.form("입력폼"):
        facility_cols = ["sub_500m","pharmacy", "hospital", "restaurant", "cafe", "CVS", "school"]
        
        st.subheader("시설 개수 입력")
        for name in facility_cols:
            st.session_state[name] = st.number_input(
                f"{name} 개수", min_value=0, step=1,
                value=st.session_state.get(name, 0), key=f"input_{name}"
            )
        
        submitted = st.form_submit_button("적용")

    if submitted:
        area = selected_row[area_col]
        result_data = {
            "택지코드": selected_row[district_code],
            "택지명": selected_row[district_col],
            "사업단계": selected_row.get(step_col, ""),
            "subway": st.session_state["sub_500m"],
            "pharmacy": st.session_state["pharmacy"]/area,
            "hospital": st.session_state["hospital"]/area,
            "restaurant": st.session_state["restaurant"]/area,
            "cafe": st.session_state["cafe"]/area,
            "CVS": st.session_state["CVS"]/area,
            "school": st.session_state["school"]/area,
        }
        
        # 새로운 7개 비율 추가 (0~100을 0~1로 변환)
        for key in ratio_keys:
            result_data[key] = st.session_state[key] / 100.0
            
        st.session_state["result_df"] = pd.DataFrame([result_data])
        st.session_state["need_prediction"] = True  # 예측 필요 플래그 설정
        st.success(f"입력값이 적용되었습니다.")

# 4. 🔥 적용결과 표시 부분 바로 앞에 이 코드를 추가하세요
# (result_container = st.container() 바로 앞에)

# AI 예측 실행
if st.session_state.get("need_prediction", False) and st.session_state.get("result_df") is not None:
    if tabnet_model is not None:
        with st.spinner("🤖 AI 모델로 도시활력을 예측 중..."):
            try:
                # 1. 기존 데이터 복사
                simulation_data = selected_row.copy()
                result_row = st.session_state["result_df"].iloc[0]
                
                # 2. 시설 밀도값 업데이트
                facility_mapping = {
                    'subway': 'sub_500m',
                    'pharmacy': 'pharmacy',
                    'hospital': 'hospital', 
                    'restaurant': 'restaurant',
                    'cafe': 'cafe',
                    'CVS': 'CVS',
                    'school': 'school'
                }
                
                for result_key, sim_key in facility_mapping.items():
                    if result_key in result_row.index:
                        try:
                            simulation_data[sim_key] = float(result_row[result_key])
                        except (ValueError, TypeError):
                            pass
                
                # 3. 새로운 7개 비율 업데이트
                for ratio_key in ratio_keys:
                    if ratio_key in result_row.index:
                        try:
                            simulation_data[ratio_key] = float(result_row[ratio_key])
                        except (ValueError, TypeError):
                            pass
                
                # 4. 모델 입력 데이터 생성
                model_input = []
                for feature in FEATURES:
                    try:
                        if feature in simulation_data.index and pd.notna(simulation_data[feature]):
                            value = float(simulation_data[feature])
                        else:
                            value = 0.0
                    except (ValueError, TypeError):
                        value = 0.0
                    model_input.append(value)
                
                # 5. 모델 예측 실행
                model_input_array = np.array(model_input).reshape(1, -1)
                prediction = tabnet_model.predict(model_input_array)
                
                if hasattr(prediction, '__len__') and len(prediction) > 0:
                    predicted_vitality = float(prediction[0])
                else:
                    predicted_vitality = float(prediction)
                
                st.session_state["predicted_vitality"] = predicted_vitality
                st.session_state["need_prediction"] = False  # 플래그 리셋
                st.success(f"🤖 AI 예측이 완료되었습니다! (예측값: {predicted_vitality:.2f})")
                
            except Exception as e:
                st.session_state["predicted_vitality"] = None
                st.session_state["need_prediction"] = False
                st.error(f"❌ 예측 실행 중 오류: {str(e)}")
    else:
        st.session_state["predicted_vitality"] = None
        st.session_state["need_prediction"] = False
        st.error("❌ TabNet 모델이 로드되지 않았습니다.")
# ✅ AI 예측 결과 표시 (적용 결과보다 먼저)
if st.session_state.get("predicted_vitality") is not None and st.session_state.get("result_df") is not None:
    st.markdown("### 🤖 AI 도시활력 예측 결과")
    
    current_day = selected_row.get("day", 0)
    predicted_value = st.session_state["predicted_vitality"]
    difference = predicted_value - current_day
    change_pct = (difference / current_day) * 100 if current_day != 0 else 0
    
    # 3개 컬럼으로 비교 표시
    col_current, col_predicted, col_change = st.columns(3)
    
    with col_current:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #FFD54F 0%, #FFA726 100%);
                padding: 20px;
                border-radius: 12px;
                text-align: center;
                color: #333;
                font-weight: bold;
                margin: 10px 0;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            ">
                <div style="font-size: 16px; margin-bottom: 8px;">📊 기존 도시활력</div>
                <div style="font-size: 32px; color: #E65100;">{current_day:,.0f}</div>
                <div style="font-size: 12px; color: #666; margin-top: 5px;">Day (9-16시)</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col_predicted:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #4CAF50 0%, #66BB6A 100%);
                padding: 20px;
                border-radius: 12px;
                text-align: center;
                color: white;
                font-weight: bold;
                margin: 10px 0;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            ">
                <div style="font-size: 16px; margin-bottom: 8px;">🎯 AI 예측값</div>
                <div style="font-size: 32px;">{predicted_value:,.0f}</div>
                <div style="font-size: 12px; color: #E8F5E8; margin-top: 5px;">TabNet 모델</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col_change:
        # 변화량에 따른 색상 및 아이콘 설정
        if difference > 0:
            bg_color = "linear-gradient(135deg, #2196F3 0%, #42A5F5 100%)"
            icon = "📈"
            change_text = "증가"
            change_symbol = "+"
        elif difference < 0:
            bg_color = "linear-gradient(135deg, #FF5722 0%, #FF7043 100%)"
            icon = "📉"
            change_text = "감소"
            change_symbol = ""
        else:
            bg_color = "linear-gradient(135deg, #9E9E9E 0%, #BDBDBD 100%)"
            icon = "➡️"
            change_text = "동일"
            change_symbol = ""
        
        st.markdown(
            f"""
            <div style="
                background: {bg_color};
                padding: 20px;
                border-radius: 12px;
                text-align: center;
                color: white;
                font-weight: bold;
                margin: 10px 0;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            ">
                <div style="font-size: 16px; margin-bottom: 8px;">{icon} 변화량</div>
                <div style="font-size: 32px;">{change_symbol}{difference:,.0f}</div>
                <div style="font-size: 12px; color: #F0F0F0; margin-top: 5px;">
                    {change_text} ({change_pct:+.1f}%)
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    # 해석 메시지
    st.markdown("#### 💡 예측 해석")
    if abs(difference) < 5:
        st.info("ℹ️ **안정적 변화**: 제안된 변경사항의 도시활력 영향은 미미할 것으로 예상됩니다.")
    elif difference > 0:
        if difference > 20:
            st.success(f"🚀 **큰 폭 향상**: 제안된 변경으로 도시활력이 **{difference:.0f}점** 크게 향상될 것으로 예상됩니다!")
        else:
            st.success(f"💡 **긍정적 영향**: 제안된 변경으로 도시활력이 **{difference:.0f}점** 향상될 것으로 예상됩니다!")
    else:
        if abs(difference) > 20:
            st.error(f"⚠️ **큰 폭 감소**: 제안된 변경으로 도시활력이 **{abs(difference):.0f}점** 크게 감소할 것으로 예상됩니다.")
        else:
            st.warning(f"⚠️ **부정적 영향**: 제안된 변경으로 도시활력이 **{abs(difference):.0f}점** 감소할 것으로 예상됩니다.")
    
    st.markdown("---")  # 구분선

# ✅ 적용 결과 출력
if st.session_state.result_df is not None:
    st.markdown("### 📄 적용 결과 (이전값 vs 현재값 비교)")
    
    # 이전값 계산
    area = selected_row[area_col]
    
    # 기존 지표들 (subway는 절대값, 나머지는 밀도×면적으로 절대값 변환)
    original_facility_data = [
        selected_row.get("sub_500m", 0),  # subway는 이미 절대값
        selected_row.get("pharmacy", 0) * area,  # 밀도 × 면적 = 절대값
        selected_row.get("hospital", 0) * area,
        selected_row.get("restaurant", 0) * area,
        selected_row.get("cafe", 0) * area,
        selected_row.get("CVS", 0) * area,
        selected_row.get("school", 0) * area
    ]
    
    # 새로운 7개 비율 (0~1 값을 그대로 사용)
    original_ratio_data = [selected_row.get(key, 0) for key in ratio_keys]
    
    original_data = {
        "지표": ["subway", "pharmacy", "hospital", "restaurant", "cafe", "CVS", "school"] + 
                [ratio_labels[key] for key in ratio_keys],
        "이전값": original_facility_data + original_ratio_data
    }
    
    # 현재값 (result_df에서) - 시설 개수들은 면적을 곱해서 절대값으로 표시
    result_row = st.session_state.result_df.iloc[0]
    
    current_facility_values = [
        result_row.get("subway", 0),  # subway는 이미 절대값
        result_row.get("pharmacy", 0) * area,  # 밀도 × 면적 = 절대값
        result_row.get("hospital", 0) * area, 
        result_row.get("restaurant", 0) * area,
        result_row.get("cafe", 0) * area,
        result_row.get("CVS", 0) * area,
        result_row.get("school", 0) * area
    ]
    
    current_ratio_values = [result_row.get(key, 0) for key in ratio_keys]
    
    current_values = current_facility_values + current_ratio_values
    
    # 변화량 계산
    changes = []
    change_pcts = []
    
    for i, (original, current) in enumerate(zip(original_data["이전값"], current_values)):
        change = current - original
        changes.append(change)
        
        if original != 0:
            change_pct = (change / original) * 100
        else:
            change_pct = 0 if change == 0 else float('inf')
        change_pcts.append(change_pct)
    
    # 비교 데이터프레임 생성
    comparison_df = pd.DataFrame({
        "지표": original_data["지표"],
        "이전값": original_data["이전값"],
        "현재값": current_values,
        "변화량": changes,
        "변화율(%)": change_pcts
    })
    
    # 소수점 자리수 조정
    comparison_df["이전값"] = comparison_df["이전값"].round(4)
    comparison_df["현재값"] = comparison_df["현재값"].round(4)
    comparison_df["변화량"] = comparison_df["변화량"].round(4)
    comparison_df["변화율(%)"] = comparison_df["변화율(%)"].round(2)
    
    # 변화가 있는 항목만 하이라이트
    def highlight_changes(row):
        if abs(row["변화량"]) > 0.001:  # 변화가 있는 경우
            if row["변화량"] > 0:
                return ['background-color: #e8f5e8'] * len(row)  # 초록색 (증가)
            else:
                return ['background-color: #ffeaea'] * len(row)  # 빨간색 (감소)
        else:
            return [''] * len(row)  # 변화 없음
    
    # 스타일 적용해서 표시
    styled_df = comparison_df.style.apply(highlight_changes, axis=1)
    st.dataframe(styled_df, use_container_width=True)
    
    # 변화 요약 정보
    changed_items = comparison_df[abs(comparison_df["변화량"]) > 0.001]
    if not changed_items.empty:
        st.markdown("#### 📊 주요 변화 사항")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔺 증가한 항목:**")
            increased = changed_items[changed_items["변화량"] > 0]
            if not increased.empty:
                for _, row in increased.iterrows():
                    st.write(f"• {row['지표']}: {row['이전값']:.4f} → {row['현재값']:.4f} (+{row['변화율(%)']:.1f}%)")
            else:
                st.write("없음")
        
        with col2:
            st.markdown("**🔻 감소한 항목:**")
            decreased = changed_items[changed_items["변화량"] < 0]
            if not decreased.empty:
                for _, row in decreased.iterrows():
                    st.write(f"• {row['지표']}: {row['이전값']:.4f} → {row['현재값']:.4f} ({row['변화율(%)']:.1f}%)")
            else:
                st.write("없음")
    else:
        st.info("ℹ️ 이전 값과 동일합니다.")
    
    # 기존 result_df도 함께 표시 (접을 수 있게)
    with st.expander("📋 상세 데이터 보기"):
        st.markdown("**기본 정보:**")
        basic_info = st.session_state.result_df[["택지코드", "택지명", "사업단계"]].copy()
        st.dataframe(basic_info, use_container_width=True)
        
        st.markdown("**전체 수치 데이터:**")
        numeric_data = st.session_state.result_df.drop(["택지코드", "택지명", "사업단계"], axis=1, errors='ignore')
        st.dataframe(numeric_data, use_container_width=True)

# 6. 사이드바에 디버깅 정보 추가 (기존 사이드바 마지막에)
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🔧 AI 모델 상태")
    
    if tabnet_model is not None:
        st.success("✅ TabNet 모델 로딩됨")
    else:
        st.error("❌ TabNet 모델 로딩 실패")
    
    if st.session_state.get("predicted_vitality") is not None:
        st.info(f"🎯 최근 예측: {st.session_state.predicted_vitality:.2f}")
    else:
        st.info("📊 예측 결과 없음")
    
    if st.session_state.get("need_prediction", False):
        st.warning("⏳ 예측 대기 중...")
    
    # 현재 설정된 비율들 요약 표시
    st.markdown("### 📊 현재 토지이용 비율")
    
    # 중요한 비율들을 먼저 하이라이트해서 표시
    st.markdown("#### 🎯 **핵심 변수**")
    important_summary = {}
    for key in important_ratios:
        important_summary[ratio_labels[key]] = st.session_state.get(key, 0)
    
    for label, value in important_summary.items():
        if value > 0:
            st.markdown(f"**• {label}: {value}%**")
        else:
            st.write(f"• {label}: {value}%")
    
    # 나머지 비율들
    st.markdown("#### 📋 **기타 변수**")
    other_summary = {}
    for key in ratio_labels.keys():
        if key not in important_ratios:
            other_summary[ratio_labels[key]] = st.session_state.get(key, 0)
    
    for label, value in other_summary.items():
        if value > 0:
            st.write(f"• {label}: {value}%")
    
    # 총합 계산
    all_ratios = {**important_summary, **other_summary}
    total_current = sum(all_ratios.values())
