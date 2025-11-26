import streamlit as st

st.set_page_config(
    page_title="PresentMate - 발표 피드백 도우미",
    page_icon="presen1.png",
    layout="wide",
)

st.image("presen1.png",width=300)
st.title("발표 피드백 도우미")
st.markdown(
    """
안녕하세요, **PresentMate** 홈 화면입니다.

왼쪽 사이드바의 **Pages** 영역에서

1. `파일 업로드 발표 분석`  
2. `라이브 발표 연습`  

중에서 모드를 선택해서 사용하세요 😊
"""
)
st.info("왼쪽 사이드바에서 페이지를 선택하면 기능별 화면으로 이동합니다.")

