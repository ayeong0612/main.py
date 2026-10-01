
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -------------------------
# 기본 설정
# -------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌤️",
    layout="wide"
)

st.title("🌤️ 기온 예측기")
st.write(
    "서울의 과거 기온 데이터를 바탕으로 "
    "연평균기온을 분석하고 미래 기온을 예측합니다."
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/"
    "modudata/bb860932644270ad1199f10d3e7670e30231bce4/"
    "data/seoul.csv"
)

기준연도 = 2025
기준시작연도 = 1908

# -------------------------
# 데이터 불러오기
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    df["날짜"] = pd.to_datetime(
        df["날짜"], errors="coerce"
    )
    df["평균기온"] = pd.to_numeric(
        df["평균기온"], errors="coerce"
    )

    df = df.dropna(subset=["날짜", "평균기온"]).copy()
    df["연도"] = df["날짜"].dt.year

    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

# -------------------------
# 연도별 평균기온 계산
# -------------------------
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
    (annual["연도"] <= 기준연도) &
    (annual["관측일수"] >= 300)
].copy()

# 회귀 계산에 필요한 기간만 사용
annual = annual[
    annual["연도"] >= 기준시작연도
].copy()

annual = annual.sort_values("연도").reset_index(drop=True)

if len(annual) < 2:
    st.error("회귀 분석에 사용할 데이터가 부족합니다.")
    st.stop()

# -------------------------
# 회귀 분석
# 독립 변수: 1908년부터 지난 연수
# -------------------------
annual["지난연수"] = annual["연도"] - 기준시작연도

x = annual["지난연수"].to_numpy()
y = annual["평균기온"].to_numpy()

기울기, 절편 = np.polyfit(x, y, 1)

상관계수 = np.corrcoef(x, y)[0, 1]

시작연도 = int(annual["연도"].min())
끝연도 = int(annual["연도"].max())
사용개수 = len(annual)

# -------------------------
# 연도 선택 및 예측
# -------------------------
st.subheader("🔎 예측할 연도 선택")

선택연도 = st.slider(
    "연도",
    min_value=1900,
    max_value=2100,
    value=2026,
    step=1
)

예측지난연수 = 선택연도 - 기준시작연도
예상기온 = 기울기 * 예측지난연수 + 절편

st.metric(
    label=f"{선택연도}년 예상 연평균기온",
    value=f"{예상기온:.2f} °C"
)

if 선택연도 > 끝연도:
    st.info(
        "선택한 연도는 관측 데이터의 마지막 연도 이후입니다. "
        "회귀 직선을 연장한 추정값입니다."
    )
elif 선택연도 < 시작연도:
    st.info(
        "선택한 연도는 회귀 분석에 사용한 기간보다 이전입니다. "
        "회귀 직선을 과거로 연장한 추정값입니다."
    )

# -------------------------
# 분석 결과
# -------------------------
st.subheader("📊 회귀 분석 결과")

col1, col2, col3 = st.columns(3)

col1.metric("사용한 연도 개수", f"{사용개수}개")
col2.metric("시작 연도", f"{시작연도}년")
col3.metric("끝 연도", f"{끝연도}년")

st.write(f"**상관계수:** {상관계수:.4f}")
st.caption(
    "상관계수는 회귀 분석에 사용한 연도와 연평균기온 사이의 "
    "선형 상관관계를 나타냅니다."
)

st.write(
    f"**회귀식:** 평균기온 = "
    f"{기울기:.5f} × (연도 - {기준시작연도}) "
    f"+ {절편:.3f}"
)

# -------------------------
# 산점도와 회귀 직선
# -------------------------
st.subheader("📈 연도별 평균기온과 회귀 직선")

fig = go.Figure()

# 실제 관측값
fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["평균기온"],
        mode="markers",
        name="연평균기온",
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

# 1900~2100년 회귀선
선연도 = np.arange(1900, 2101)
선지난연수 = 선연도 - 기준시작연도
선기온 = 기울기 * 선지난연수 + 절편

fig.add_trace(
    go.Scatter(
        x=선연도,
        y=선기온,
        mode="lines",
        name="회귀 직선",
        line=dict(width=3, dash="solid")
    )
)

# 선택한 연도 표시
fig.add_trace(
    go.Scatter(
        x=[선택연도],
        y=[예상기온],
        mode="markers",
        name="선택 연도 예측",
        marker=dict(
            size=13,
            symbol="diamond",
            line=dict(width=2, color="black")
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
    margin=dict(l=20, r=20, t=50, b=20)
)

st.plotly_chart(fig, width="stretch")

# -------------------------
# 데이터 표
# -------------------------
with st.expander("연도별 분석 데이터 보기"):
    보여줄표 = annual[
        ["연도", "평균기온", "관측일수"]
    ].copy()

    보여줄표["평균기온"] = 보여줄표["평균기온"].round(2)

    st.dataframe(
        보여줄표,
        hide_index=True,
        width="stretch"
    )

st.caption(
    "※ 이 예측은 과거 연평균기온의 선형 추세를 연장한 값이며, "
    "실제 미래 기온이나 기후 예측 모델의 결과와는 다를 수 있습니다."
)
