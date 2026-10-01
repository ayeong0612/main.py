
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# =====================================
# 1. 기본 설정
# =====================================
st.set_page_config(
    page_title="서울 기온 예측기",
    page_icon="🌤️",
    layout="wide"
)

st.title("🌤️ 서울 기온 예측기")
st.write(
    "서울의 과거 기온 데이터로 연평균기온을 분석하고 "
    "회귀 직선을 이용해 예상 기온을 확인합니다."
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/"
    "modudata/bb860932644270ad1199f10d3e7670e30231bce4/"
    "data/seoul.csv"
)

기준연도 = 2025
기준시작연도 = 1908

# =====================================
# 2. 데이터 불러오기
# =====================================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["날짜", "평균기온"]
    ).copy()

    df["연도"] = df["날짜"].dt.year

    return df


try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

# =====================================
# 3. 연도별 평균기온 계산
# =====================================
annual = (
    df.groupby("연도")
    .agg(
        평균기온=("평균기온", "mean"),
        관측일수=("날짜", "nunique")
    )
    .reset_index()
)

# 2025년까지, 관측일 300일 이상인 해만 사용
annual = annual[
    (annual["연도"] >= 기준시작연도) &
    (annual["연도"] <= 기준연도) &
    (annual["관측일수"] >= 300)
].copy()

annual = annual.sort_values(
    "연도"
).reset_index(drop=True)

if len(annual) < 2:
    st.error("회귀 분석에 사용할 데이터가 부족합니다.")
    st.stop()

# =====================================
# 4. 회귀 분석 함수
# =====================================
def 회귀분석(data):
    # 독립 변수: 1908년부터 지난 연수
    x = (
        data["연도"].to_numpy(dtype=float)
        - 기준시작연도
    )

    y = data["평균기온"].to_numpy(dtype=float)

    기울기, 절편 = np.polyfit(x, y, 1)

    # 상관계수
    if np.std(x) > 0 and np.std(y) > 0:
        상관계수 = np.corrcoef(x, y)[0, 1]
    else:
        상관계수 = np.nan

    # 1년당 기울기를 100년당 변화량으로 환산
    백년변화량 = 기울기 * 100

    return 기울기, 절편, 상관계수, 백년변화량


# 전체 기간 회귀 분석
(
    전체_기울기,
    전체_절편,
    전체_상관계수,
    전체_백년변화
) = 회귀분석(annual)

# 최근 20개 유효 연도 회귀 분석
최근20 = annual.tail(20).copy()

if len(최근20) < 2:
    st.error("최근 기간 분석에 사용할 데이터가 부족합니다.")
    st.stop()

(
    최근_기울기,
    최근_절편,
    최근_상관계수,
    최근_백년변화
) = 회귀분석(최근20)

# 분석 기간
시작연도 = int(annual["연도"].min())
끝연도 = int(annual["연도"].max())
사용개수 = len(annual)

최근시작연도 = int(최근20["연도"].min())
최근끝연도 = int(최근20["연도"].max())

# =====================================
# 5. 100년당 기온 변화량 비교
# =====================================
st.subheader("🌡️ 100년당 기온 변화량 비교")

st.write(
    "회귀 직선의 기울기를 100년당 몇 °C 변하는지로 "
    "환산한 결과입니다."
)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="전체 기간",
        value=f"{전체_백년변화:+.2f} °C"
    )

    st.write(
        f"기간: {시작연도}~{끝연도}년"
    )

    st.write(
        f"사용한 연도: {사용개수}개"
    )

with col2:
    st.metric(
        label="최근 20개 유효 연도",
        value=f"{최근_백년변화:+.2f} °C"
    )

    st.write(
        f"기간: {최근시작연도}~{최근끝연도}년"
    )

    st.write(
        f"사용한 연도: {len(최근20)}개"
    )

st.caption(
    "양수(+)는 상승 추세, 음수(-)는 하강 추세를 뜻합니다. "
    "각 값은 해당 기간의 직선을 100년 기준으로 환산한 것입니다."
)

st.divider()

# =====================================
# 6. 상관관계 비교
# =====================================
st.subheader("📊 연도와 평균기온의 상관관계")

st.write(
    "상관계수는 연도와 연평균기온 사이의 "
    "선형 관계의 방향과 강도를 나타냅니다."
)

col3, col4 = st.columns(2)

with col3:
    st.metric(
        label="전체 기간 상관계수",
        value=f"{전체_상관계수:.4f}"
    )

with col4:
    st.metric(
        label="최근 20개 유효 연도 상관계수",
        value=f"{최근_상관계수:.4f}"
    )

st.caption(
    "상관계수는 -1부터 +1까지입니다. "
    "+1에 가까우면 양의 상관관계, "
    "-1에 가까우면 음의 상관관계, "
    "0에 가까우면 선형 상관관계가 약합니다."
)

# =====================================
# 7. 회귀식 및 분석 기간
# =====================================
st.subheader("🧮 회귀 분석 정보")

st.write(
    f"**전체 기간 회귀식**  \n"
    f"평균기온 = {전체_기울기:.5f} × "
    f"(연도 - {기준시작연도}) "
    f"+ {전체_절편:.3f}"
)

st.write(
    f"**최근 기간 회귀식**  \n"
    f"평균기온 = {최근_기울기:.5f} × "
    f"(연도 - {기준시작연도}) "
    f"+ {최근_절편:.3f}"
)

st.write(
    f"**전체 기간:** {시작연도}~{끝연도}년 "
    f"({사용개수}개 연도)"
)

st.write(
    f"**최근 기간:** {최근시작연도}~"
    f"{최근끝연도}년 "
    f"({len(최근20)}개 연도)"
)

st.caption(
    "최근 20년은 관측일 300일 이상인 해 중 "
    "가장 최근의 유효 연도 20개입니다."
)

# =====================================
# 8. 연도 슬라이더와 예상 기온
# =====================================
st.subheader("🔎 예상 기온 확인")

선택연도 = st.slider(
    "예측할 연도",
    min_value=1900,
    max_value=2100,
    value=2026,
    step=1,
    key="year_slider"
)

# 전체 기간 회귀 직선으로 예상 기온 계산
예측지난연수 = 선택연도 - 기준시작연도

예상기온 = (
    전체_기울기 * 예측지난연수
    + 전체_절편
)

st.metric(
    label=f"{선택연도}년 예상 연평균기온",
    value=f"{예상기온:.2f} °C"
)

if 선택연도 > 끝연도:
    st.info(
        "선택한 연도는 실제 관측 자료의 마지막 연도 이후입니다. "
        "전체 기간 회귀 직선을 연장한 추정값입니다."
    )
elif 선택연도 < 시작연도:
    st.info(
        "선택한 연도는 분석 시작 연도 이전입니다. "
        "전체 기간 회귀 직선을 과거로 연장한 추정값입니다."
    )

# =====================================
# 9. 산점도와 회귀 직선
# =====================================
st.subheader("📈 연도별 평균기온과 회귀 직선")

fig = go.Figure()

# 실제 연평균기온 산점도
fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(
            size=7,
            opacity=0.75
        ),
        customdata=annual["관측일수"],
        hovertemplate=(
            "연도: %{x}년<br>"
            "평균기온: %{y:.2f} °C<br>"
            "관측일수: %{customdata}일"
            "<extra></extra>"
        )
    )
)

# 전체 기간 회귀 직선
선연도 = np.arange(1900, 2101)

선지난연수 = (
    선연도 - 기준시작연도
)

선기온 = (
    전체_기울기 * 선지난연수
    + 전체_절편
)

fig.add_trace(
    go.Scatter(
        x=선연도,
        y=선기온,
        mode="lines",
        name="전체 기간 회귀 직선",
        line=dict(width=3)
    )
)

# 최근 20개 유효 연도 회귀 직선
최근선연도 = np.arange(
    최근시작연도,
    최근끝연도 + 1
)

최근선지난연수 = (
    최근선연도 - 기준시작연도
)

최근선기온 = (
    최근_기울기 * 최근선지난연수
    + 최근_절편
)

fig.add_trace(
    go.Scatter(
        x=최근선연도,
        y=최근선기온,
        mode="lines",
        name="최근 20개 연도 회귀 직선",
        line=dict(
            width=3,
            dash="dash"
        )
    )
)

# 선택한 연도의 예상 기온
fig.add_trace(
    go.Scatter(
        x=[선택연도],
        y=[예상기온],
        mode="markers",
        name="선택 연도 예상 기온",
        marker=dict(
            size=13,
            symbol="diamond",
            line=dict(
                width=2,
                color="black"
            )
        ),
        hovertemplate=(
            "선택 연도: %{x}년<br>"
            "예상 기온: %{y:.2f} °C"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    xaxis=dict(
        title="연도",
        tickmode="linear",
        dtick=10,
        showgrid=True
    ),
    yaxis=dict(
        title="연평균기온 (°C)",
        showgrid=True
    ),
    hovermode="closest",
    height=550,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    ),
    margin=dict(
        l=20,
        r=20,
        t=60,
        b=20
    )
)

st.plotly_chart(
    fig,
    width="stretch"
)

# =====================================
# 10. 연도별 분석 데이터
# =====================================
with st.expander("📋 연도별 분석 데이터 보기"):
    보여줄표 = annual[
        ["연도", "평균기온", "관측일수"]
    ].copy()

    보여줄표["평균기온"] = (
        보여줄표["평균기온"].round(2)
    )

    st.dataframe(
        보여줄표,
        hide_index=True,
        width="stretch"
    )

# =====================================
# 11. 안내
# =====================================
st.caption(
    "데이터: 서울 일별 기온 자료 | "
    "분석 기준: 2025년까지, 연간 관측일 300일 이상"
)

st.caption(
    "※ 회귀 직선은 과거의 선형 추세를 나타냅니다. "
    "미래의 실제 기온을 정확히 예측하는 모델은 아닙니다."
)
