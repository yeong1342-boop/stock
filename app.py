import os
from io import BytesIO, StringIO
from datetime import datetime

import streamlit as st
import requests
import pandas as pd
from bs4 import BeautifulSoup

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

st.set_page_config(page_title="KOSPI AI 요약", page_icon="📈", layout="wide")

URL = "https://finance.naver.com/sise/sise_index.naver?code=KOSPI"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def get_text(soup, selector):
    tag = soup.select_one(selector)
    if tag:
        return tag.get_text(" ", strip=True)
    return ""


def fetch_kospi_data():
    response = requests.get(URL, headers=HEADERS, timeout=10)
    if response.status_code != 200:
        raise Exception(f"페이지 요청 실패: {response.status_code}")

    response.encoding = "euc-kr"
    html = response.text
    soup = BeautifulSoup(html, "html.parser")

    kospi_info = {
        "수집일시": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "지수명": "KOSPI",
        "수집URL": URL,
        "현재지수": get_text(soup, "#now_value"),
        "전일대비": get_text(soup, "#change_value_and_rate"),
        "기준시각": get_text(soup, ".time"),
    }
    summary_df = pd.DataFrame([kospi_info])

    tables = pd.read_html(StringIO(html))
    cleaned_tables = []
    for table in tables:
        temp_df = table.copy()
        temp_df = temp_df.dropna(how="all")
        temp_df = temp_df.dropna(axis=1, how="all")
        temp_df.columns = [str(col).strip() for col in temp_df.columns]
        if len(temp_df) > 0 and len(temp_df.columns) > 1:
            cleaned_tables.append(temp_df)

    return summary_df, cleaned_tables


def df_to_text(df, max_rows=20):
    return df.head(max_rows).to_string(index=False)


def build_ai_input(summary_df, cleaned_tables):
    table_text_list = []
    for i, table_df in enumerate(cleaned_tables):
        table_text_list.append(f"\n[표 {i + 1}]\n{df_to_text(table_df)}\n")

    return f"""
아래는 네이버 금융 KOSPI 페이지에서 수집한 데이터입니다.

[주요 지수 정보]
{summary_df.to_string(index=False)}

[페이지 내 표 데이터]
{chr(10).join(table_text_list)}
"""


def run_gpt_summary(ai_input, model_name):
    model = ChatOpenAI(model=model_name)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
너는 증권사 마케팅 담당자가 보기 쉽게 시장 정보를 요약하는 애널리스트야.

아래 데이터를 바탕으로 다음 형식으로 요약해줘.

1. 한 줄 요약
2. 현재 KOSPI 흐름
3. 상승/하락 특징
4. 거래 관련 특징
5. 증권사 마케팅 관점에서 참고할 만한 포인트
6. 유의사항

주의사항:
- 데이터에 없는 내용은 추측하지 마.
- 투자 추천처럼 말하지 마.
- 매수/매도 권유 표현은 쓰지 마.
- 숫자는 원문 데이터 기준으로 언급해.
- 한국어로 간결하게 작성해.
"""
        ),
        ("user", "{input}")
    ])

    chain = prompt | model | StrOutputParser()
    return chain.invoke({"input": ai_input})


def build_excel(summary_df, cleaned_tables, gpt_summary):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="주요지수정보", index=False)
        for i, table_df in enumerate(cleaned_tables):
            sheet_name = f"표_{i + 1}"[:31]
            table_df.to_excel(writer, sheet_name=sheet_name, index=False)
        pd.DataFrame({"GPT 요약": [gpt_summary]}).to_excel(
            writer, sheet_name="GPT요약", index=False
        )
    buffer.seek(0)
    return buffer


# =========================
# Streamlit UI
# =========================

st.title("📈 네이버 KOSPI AI 요약")
st.caption("네이버 금융 KOSPI 페이지를 수집하고 GPT로 요약한 뒤 엑셀로 저장합니다.")

with st.sidebar:
    st.header("설정")
    model_name = st.text_input("OpenAI 모델명", value="gpt-4o-mini")
    has_key = bool(os.getenv("OPENAI_API_KEY"))
    if has_key:
        st.success("OPENAI_API_KEY 감지됨 (.env)")
    else:
        st.warning("OPENAI_API_KEY가 설정되어 있지 않습니다.")
    run_button = st.button("데이터 수집 + AI 요약 실행", type="primary", use_container_width=True)

if run_button:
    try:
        with st.spinner("네이버 금융에서 KOSPI 데이터 수집 중..."):
            summary_df, cleaned_tables = fetch_kospi_data()
    except Exception as e:
        st.error(f"데이터 수집 실패: {e}")
        st.stop()

    st.subheader("주요 지수 정보")
    st.dataframe(summary_df, use_container_width=True)

    st.subheader(f"페이지 내 표 데이터 ({len(cleaned_tables)}개)")
    for i, table_df in enumerate(cleaned_tables):
        with st.expander(f"표 {i + 1}"):
            st.dataframe(table_df, use_container_width=True)

    ai_input = build_ai_input(summary_df, cleaned_tables)

    try:
        with st.spinner("GPT 요약 생성 중..."):
            gpt_summary = run_gpt_summary(ai_input, model_name)
    except Exception as e:
        st.error(f"GPT 요약 실패: {e}")
        st.stop()

    st.subheader("GPT 요약")
    st.markdown(gpt_summary)

    excel_buffer = build_excel(summary_df, cleaned_tables, gpt_summary)
    st.download_button(
        label="엑셀 다운로드",
        data=excel_buffer,
        file_name="네이버_KOSPI_AI요약_결과.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
else:
    st.info("왼쪽 사이드바에서 실행 버튼을 눌러주세요.")
