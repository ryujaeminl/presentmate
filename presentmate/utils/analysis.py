import os
import tempfile
from datetime import timedelta

import librosa
import numpy as np
import whisper
import streamlit as st


# ---------------------------
# 공통: Whisper 모델 로딩
# ---------------------------
@st.cache_resource
def load_whisper_model(model_name: str = "base"):
    model = whisper.load_model(model_name)
    return model


# ---------------------------
# 공통: 업로드/녹음 파일 임시 저장
# ---------------------------
def save_bytes_to_temp(data: bytes, suffix: str = ".wav") -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        temp_path = tmp.name
    return temp_path


def save_uploaded_file_to_temp(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name
    return temp_path


# ---------------------------
# 공통: 오디오 길이 계산
# ---------------------------
def get_audio_duration(path: str) -> float:
    try:
        duration = librosa.get_duration(path=path)
    except Exception:
        y, sr = librosa.load(path, sr=None)
        duration = len(y) / sr
    return duration


# ---------------------------
# 공통: 말하기 속도 계산
# ---------------------------
def calc_speaking_speed(text: str, duration_sec: float):
    words = text.strip().split()
    word_count = len(words)

    if duration_sec <= 0:
        return word_count, 0.0, 0.0

    minutes = duration_sec / 60
    wpm = word_count / minutes
    wps = word_count / duration_sec
    return word_count, wpm, wps


def speaking_speed_comment(wpm: float) -> str:
    if wpm == 0:
        return "발표 길이나 단어 수 계산이 어려워 속도를 평가하지 못했어요."
    if wpm < 90:
        return "조금 느린 편이에요. 문장 사이 템포는 좋지만, 너무 천천히 느껴질 수 있어요."
    elif 90 <= wpm <= 140:
        return "적절한 속도예요. 대부분의 청중이 이해하기 편한 말하기 속도입니다."
    else:
        return "조금 빠른 편이에요. 중요한 부분에서 잠깐 멈추거나 포인트를 강조하면 더 좋겠어요."


# ---------------------------
# 공통: 군더더기 표현 카운트
# ---------------------------
def count_filler_words(text: str):
    text_lower = text.lower()
    filler_list = [
        "um", "uh", "uhm", "er", "ah",
        "like", "you know", "i mean",
        "well", "so", "actually", "basically",
        "그냥", "뭔가", "약간", "그러니까", "뭐랄까"
    ]

    counts = {}
    for f in filler_list:
        counts[f] = text_lower.count(f)

    counts = {k: v for k, v in counts.items() if v > 0}
    return counts


# ---------------------------
# 공통: 구조 피드백
# ---------------------------
def get_structure_suggestion(text: str):
    lower = text.lower()

    intro_keywords = ["good morning", "good afternoon", "hello", "today i will", "i'm going to talk"]
    body_keywords = ["first", "second", "third", "on the one hand", "on the other hand"]
    conclusion_keywords = ["in conclusion", "to sum up", "finally", "lastly"]

    intro_flag = any(k in lower for k in intro_keywords)
    body_flag = any(k in lower for k in body_keywords)
    concl_flag = any(k in lower for k in conclusion_keywords)

    messages = []

    if intro_flag:
        messages.append("• 서론(인사 및 발표 주제 소개)은 잘 들어가 있는 편이에요.")
    else:
        messages.append("• 서론에서 **인사 + 오늘 발표 주제**를 한 문장으로 더 분명히 말해주면 좋아요.")

    if body_flag:
        messages.append("• 본론에서 **First / Second / Finally** 같은 연결어를 사용해서 구조가 비교적 잘 보입니다.")
    else:
        messages.append("• 본론에 **First, Second, Finally** 같은 표현을 넣으면 청중이 구조를 잘 따라올 수 있어요.")

    if concl_flag:
        messages.append("• 결론 부분에서 발표를 어느 정도 정리해 주고 있습니다.")
    else:
        messages.append("• 마지막에 **In conclusion, To sum up** 같은 문장으로 전체 내용을 한번 정리해 주면 좋아요.")

    return "\n".join(messages)


# ---------------------------
# 공통: 타임라인 쪼개기
# ---------------------------
def make_timeline_chunks(text: str, duration_sec: float, chunk_count: int = 5):
    words = text.strip().split()
    if not words or duration_sec <= 0:
        return []

    n = len(words)
    chunk_size = max(1, n // chunk_count)
    chunks = []

    for i in range(0, n, chunk_size):
        part_words = words[i:i + chunk_size]
        start_ratio = i / n
        end_ratio = min((i + chunk_size) / n, 1.0)

        start_time = timedelta(seconds=int(duration_sec * start_ratio))
        end_time = timedelta(seconds=int(duration_sec * end_ratio))

        chunks.append(
            {
                "start": start_time,
                "end": end_time,
                "text": " ".join(part_words),
            }
        )
    return chunks

