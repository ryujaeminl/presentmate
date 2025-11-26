import os
import math
import tempfile
from typing import List, Tuple, Dict

import numpy as np
import streamlit as st

# ----------------- Optional deps -----------------
HAS_WEBRTC = True
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode
    import av

    RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
except Exception:
    HAS_WEBRTC = False
    RTC_CONFIGURATION = None

HAS_SF = True
try:
    import soundfile as sf
except Exception:
    HAS_SF = False

HAS_LIBROSA = True
try:
    import librosa
except Exception:
    HAS_LIBROSA = False

HAS_PLOT = True
try:
    import matplotlib.pyplot as plt
except Exception:
    HAS_PLOT = False

HAS_FW = True
try:
    from faster_whisper import WhisperModel
except Exception:
    HAS_FW = False

# ----------------- Constants -----------------
FILLERS_KO = ["음", "어", "그", "약간", "그러니까", "뭔가", "음...", "어..."]
FILLERS_EN = ["um", "uh", "like", "you know", "so", "well"]


# ----------------- Helper functions -----------------
def rms_db(y: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(y))) + 1e-12)
    return 20.0 * math.log10(rms)


def detect_pauses(
    y: np.ndarray, sr: int, top_db: int = 30, min_pause_ms: int = 250
) -> List[Tuple[float, float]]:
    if not HAS_LIBROSA or len(y) == 0 or sr <= 0:
        return []
    non_silent = librosa.effects.split(y, top_db=top_db)
    pauses: List[Tuple[float, float]] = []
    last_end = 0
    for start, end in non_silent:
        if start > last_end:
            pauses.append((last_end / sr, start / sr))
        last_end = end
    if last_end < len(y):
        pauses.append((last_end / sr, len(y) / sr))
    min_len = min_pause_ms / 1000.0
    return [(s, e) for s, e in pauses if (e - s) >= min_len]


def count_fillers(text: str) -> Dict[str, int]:
    if not text:
        return {}
    low = text.lower()
    tokens = low.replace("\n", " ").split()
    fillers = list(dict.fromkeys(FILLERS_KO + FILLERS_EN))
    counts: Dict[str, int] = {f: 0 for f in fillers}
    for t in tokens:
        for f in fillers:
            if f and f.lower() in t:
                counts[f] += 1
    return {k: v for k, v in counts.items() if v > 0}


def estimate_wpm_from_text(text: str, dur: float) -> float:
    if not text or dur <= 0:
        return 0.0
    words = text.split()
    return float(len(words) / (dur / 60.0))


def estimate_wpm_without_transcript(y: np.ndarray, sr: int) -> float:
    if not HAS_LIBROSA or len(y) == 0 or sr <= 0:
        return 0.0
    frame_len = int(0.03 * sr)
    hop = int(0.015 * sr)
    if frame_len <= 0 or hop <= 0 or len(y) <= frame_len:
        return 0.0
    frames = np.lib.stride_tricks.sliding_window_view(y, frame_len)[::hop]
    rms = np.sqrt((frames**2).mean(axis=1))
    thr = max(1e-6, float(np.median(rms) * 2.0))
    speech_mask = rms > thr
    voiced_dur = float(speech_mask.mean()) * (len(y) / sr)
    if voiced_dur <= 0:
        return 0.0
    zcr = ((np.diff(np.sign(y)) != 0).sum() / (len(y) / sr)) if len(y) > 1 else 0.0
    k = 0.02
    syll_per_sec = max(0.0, zcr * k)
    syllables = syll_per_sec * voiced_dur
    syl_per_word = 2.5
    words = syllables / syl_per_word
    return float(max(0.0, words / ((len(y) / sr) / 60.0)))


def speaking_speed_comment(wpm: float) -> str:
    if wpm <= 0:
        return "발표 시간이 너무 짧거나 텍스트가 인식되지 않았어요. 다시 한 번 시도해 보세요."
    if wpm < 90:
        return (
            f"현재 말하기 속도는 약 **{wpm:.1f} WPM** 으로 꽤 느린 편이에요. "
            "조금만 더 리듬감 있게 이어서 말하는 연습을 해보면 좋아요."
        )
    elif 90 <= wpm <= 150:
        return (
            f"현재 말하기 속도는 약 **{wpm:.1f} WPM** 으로 발표용으로 보기 좋은 속도예요 👏 "
            "중요한 부분에서 살짝 속도를 줄여 주면 더 좋습니다."
        )
    elif 150 < wpm <= 190:
        return (
            f"현재 말하기 속도는 약 **{wpm:.1f} WPM** 으로 조금 빠른 편이에요. "
            "문장 끝에서 잠깐 멈추는 연습을 해보면 좋아요."
        )
    else:
        return (
            f"현재 말하기 속도는 약 **{wpm:.1f} WPM** 으로 많이 빠른 편이에요 ⚠️ "
            "내용을 줄이거나, 한 문장씩 끊어 말하는 연습을 해보는 게 좋겠습니다."
        )


def _fmt_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m:02d}:{s:02d}"


def make_timeline_chunks(text: str, duration_sec: float):
    if not text or duration_sec <= 0:
        return []
    words = text.split()
    n_words = len(words)
    if n_words == 0:
        return []
    target_chunk_sec = 25.0
    n_chunks = max(1, int(round(duration_sec / target_chunk_sec)))
    n_chunks = min(n_chunks, 8)
    words_per_chunk = max(1, n_words // n_chunks)
    chunks = []
    for i in range(n_chunks):
        start_idx = i * words_per_chunk
        end_idx = (
            n_words if i == n_chunks - 1 else min(n_words, (i + 1) * words_per_chunk)
        )
        if start_idx >= n_words:
            break
        chunk_words = words[start_idx:end_idx]
        chunk_text = " ".join(chunk_words).strip()
        chunk_start_sec = duration_sec * (i / n_chunks)
        chunk_end_sec = duration_sec * ((i + 1) / n_chunks)
        chunks.append(
            {
                "start": _fmt_time(chunk_start_sec),
                "end": _fmt_time(chunk_end_sec),
                "text": chunk_text or "(이 구간에서는 인식된 텍스트가 거의 없습니다.)",
            }
        )
    return chunks


@st.cache_resource
def get_fw_model(model_name: str = "tiny"):
    if not HAS_FW:
        raise RuntimeError(
            "faster-whisper가 설치되어 있지 않습니다. "
            "`pip install faster-whisper` 후 다시 실행해 주세요."
        )
    return WhisperModel(model_name, device="cpu", compute_type="int8")


# ----------------- Streamlit UI -----------------
st.set_page_config(
    page_title="라이브 영상 발표 피드백",
    page_icon="presen1.png",
    layout="wide",
)

st.image("presen1.png", width=100)
st.title("라이브 발표 연습 (PresentMate)")
st.caption(
    "웹캠 + 마이크로 실시간 발표를 녹화하고, 말하는 동안 바로 전사/단어 분석을 확인해요."
)

# 초기 session_state 세팅
if "audio_chunks" not in st.session_state:
    st.session_state.audio_chunks = []
if "audio_sr" not in st.session_state:
    st.session_state.audio_sr = 16000
if "live_text" not in st.session_state:
    st.session_state.live_text = ""
if "live_fillers" not in st.session_state:
    st.session_state.live_fillers = {}
if "last_analyzed_len" not in st.session_state:
    st.session_state.last_analyzed_len = 0

col_main, col_side = st.columns([2, 1])

with col_side:
    st.header("⚙️ 설정")
    model_name = st.selectbox(
        "faster-whisper 모델",
        options=["tiny", "base", "small"],
        index=0,
    )
    lang_hint = st.selectbox("언어 힌트", ["자동", "ko", "en"], index=0)
    target_minutes = st.number_input(
        "목표 발표 시간 (분)",
        min_value=0.0,
        max_value=60.0,
        value=3.0,
        step=0.5,
    )
    top_db = st.slider("무음 감지 민감도 (dB)", 20, 60, 30, 2)
    vol_warn_db = st.slider("볼륨 경고 임계치 (dBFS)", -60, -10, -28, 1)
    auto_realtime = st.toggle("실시간 전사 / 단어 분석 켜기", value=True)
    min_rt_sec = st.slider("실시간 분석 주기 (초)", 3, 15, 6, 1)

    st.markdown("---")
    st.markdown("- 말하는 동안 오른쪽에 실시간 전사/군더더기 표현이 올라가요.")
    st.markdown("- 아래 버튼으로 전체 구간 기준 최종 분석도 할 수 있어요.")

with col_main:
    st.subheader("1️⃣ 웹캠 / 마이크 연결")

    if not HAS_WEBRTC or RTC_CONFIGURATION is None:
        st.error(
            "⚠️ `streamlit-webrtc` 또는 `av` 패키지가 설치되어 있지 않거나 초기화에 실패했습니다.\n\n"
            "`pip install streamlit-webrtc av` 후 다시 실행해 주세요."
        )
        st.stop()

    ctx = webrtc_streamer(
        key="live-av",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": True},
        async_processing=True,
    )

    st.markdown("웹캠이 보이고, 브라우저에서 카메라/마이크 허용을 눌러주세요.")

    # 오디오 프레임 수집
    if ctx.state.playing and ctx.audio_receiver:
        try:
            audio_frames = ctx.audio_receiver.get_frames(timeout=0.1)
        except Exception:
            audio_frames = []

        for frame in audio_frames:
            try:
                arr = frame.to_ndarray()
                if arr.ndim == 2:
                    audio_mono = arr.mean(axis=0).astype(np.float32)
                else:
                    audio_mono = arr.astype(np.float32)
                st.session_state.audio_chunks.append(audio_mono)
                st.session_state.audio_sr = frame.sample_rate
            except Exception:
                pass

    c1, c2 = st.columns(2)
    with c1:
        if st.button("♻️ 녹음 리셋"):
            st.session_state.audio_chunks = []
            st.session_state.audio_sr = 16000
            st.session_state.live_text = ""
            st.session_state.live_fillers = {}
            st.session_state.last_analyzed_len = 0
            st.success("녹음 버퍼를 초기화했습니다. 다시 발표를 시작해 주세요.")

    total_samples = (
        int(sum(len(ch) for ch in st.session_state.audio_chunks))
        if st.session_state.audio_chunks
        else 0
    )
    approx_dur = (
        total_samples / st.session_state.audio_sr
        if st.session_state.audio_sr > 0
        else 0.0
    )
    st.caption(f"현재까지 녹음된 분량 (추정): **{approx_dur:.1f}초**")

# ----------------- 실시간 분석 로직 -----------------
if auto_realtime and HAS_FW and st.session_state.audio_chunks:
    y_full = np.concatenate(st.session_state.audio_chunks).astype(np.float32)
    sr = int(st.session_state.audio_sr) if st.session_state.audio_sr > 0 else 16000
    new_len = len(y_full)
    # 마지막 분석 이후 min_rt_sec 초 이상 새로 쌓였을 때만 분석
    need_samples = int(sr * float(min_rt_sec))
    if new_len - st.session_state.last_analyzed_len >= need_samples:
        # 마지막 10초만 잘라서 분석 (너무 길어지면 느려지니까)
        window_sec = 10.0
        start = max(0, new_len - int(sr * window_sec))
        seg = y_full[start:new_len]

        fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        if HAS_SF:
            sf.write(wav_path, seg, sr, subtype="PCM_16")
        else:
            # librosa가 있다면 fallback
            if not HAS_LIBROSA:
                wav_path = None
            else:
                librosa.output.write_wav(wav_path, seg, sr)  # type: ignore

        if wav_path and os.path.exists(wav_path):
            try:
                with st.spinner("실시간 전사 업데이트 중..."):
                    model = get_fw_model(model_name)
                    language = None if lang_hint == "자동" else lang_hint
                    segments, info = model.transcribe(
                        wav_path,
                        language=language,
                        vad_filter=True,
                    )
                    texts = [seg.text.strip() for seg in segments if seg.text]
                    chunk_text = " ".join(texts).strip()
                    if chunk_text:
                        # 중복 붙는 거 방지하려고 약간 간격 두고 이어붙이기
                        if st.session_state.live_text:
                            st.session_state.live_text += " " + chunk_text
                        else:
                            st.session_state.live_text = chunk_text
                        st.session_state.live_fillers = count_fillers(
                            st.session_state.live_text
                        )
            except Exception as e:
                st.info(f"실시간 전사 중 오류 발생: {e}")
            finally:
                try:
                    os.remove(wav_path)
                except Exception:
                    pass

        st.session_state.last_analyzed_len = new_len

# ----------------- 실시간 전사 / 단어 분석 UI -----------------
st.markdown("---")
st.subheader("2️⃣ 실시간 전사 / 단어 분석")

col_t1, col_t2 = st.columns([2, 1])

with col_t1:
    st.markdown("### 📝 실시간 전사 텍스트")
    if st.session_state.live_text:
        st.text_area(
            "지금까지 인식된 발표 내용",
            value=st.session_state.live_text,
            height=220,
        )
    else:
        st.info("아직 인식된 텍스트가 없습니다. 마이크를 통해 말을 시작해 보세요.")

with col_t2:
    st.markdown("### 🗣️ 군더더기 표현 (실시간)")
    fillers = st.session_state.live_fillers or {}
    if fillers:
        sorted_fillers = sorted(fillers.items(), key=lambda x: x[1], reverse=True)
        for w, c in sorted_fillers:
            st.write(f"- `{w}` × {c}")
    else:
        st.write("현재까지 눈에 띄는 군더더기 표현은 거의 없습니다. 👍")

# ----------------- 최종 분석 버튼 -----------------
st.markdown("---")
st.subheader("3️⃣ 전체 구간 기준 최종 분석")

if st.button("🚀 지금까지 녹음된 음성 최종 분석", type="primary"):
    if not st.session_state.audio_chunks:
        st.warning("녹음된 오디오가 없습니다. 먼저 웹캠 위에서 발표를 해주세요.")
        st.stop()

    y = np.concatenate(st.session_state.audio_chunks).astype(np.float32)
    sr = int(st.session_state.audio_sr) if st.session_state.audio_sr > 0 else 16000
    duration_sec = len(y) / sr if len(y) > 0 and sr > 0 else 0.0
    if duration_sec < 1.0:
        st.warning("녹음 길이가 너무 짧습니다. 최소 1초 이상 말해 주세요.")
        st.stop()

    # 최종 분석용 wav 저장
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    if HAS_SF:
        sf.write(wav_path, y, sr, subtype="PCM_16")
    else:
        if not HAS_LIBROSA:
            st.error("soundfile/librosa가 없어 WAV 저장에 실패했습니다.")
            st.stop()
        librosa.output.write_wav(wav_path, y, sr)  # type: ignore

    final_text = st.session_state.live_text

    if not final_text and HAS_FW:
        # 실시간 전사가 비어 있을 경우 전체 다시 전사
        with st.spinner("전체 구간 전사 중..."):
            try:
                model = get_fw_model(model_name)
                language = None if lang_hint == "자동" else lang_hint
                segments, info = model.transcribe(
                    wav_path,
                    language=language,
                    vad_filter=True,
                )
                texts = [seg.text.strip() for seg in segments if seg.text]
                final_text = " ".join(texts).strip()
            except Exception as e:
                st.error(f"전사 중 오류 발생: {e}")

    vol_db_val = rms_db(y) if len(y) > 0 else -120.0
    pauses = detect_pauses(y, sr, top_db=top_db) if len(y) > 0 else []
    total_pause = float(sum(e - s for s, e in pauses))

    if final_text:
        wpm = estimate_wpm_from_text(final_text, duration_sec)
    else:
        wpm = estimate_wpm_without_transcript(y, sr)

    filler_counts = count_fillers(final_text) if final_text else {}
    timeline_chunks = (
        make_timeline_chunks(final_text, duration_sec) if final_text else []
    )

    try:
        os.remove(wav_path)
    except Exception:
        pass

    tab1, tab2, tab3 = st.tabs(["📊 요약 피드백", "🧩 타임라인", "📜 전체 전사"])

    with tab1:
        st.subheader("📊 발표 요약")
        c0, c1, c2, c3 = st.columns(4)
        with c0:
            st.metric("발표 길이", f"{duration_sec:.1f}s", f"{duration_sec/60:.2f}분")
        with c1:
            st.metric("말하기 속도", f"{wpm:.1f} WPM")
        with c2:
            st.metric("평균 볼륨", f"{vol_db_val:.1f} dBFS")
        with c3:
            st.metric("총 침묵", f"{total_pause:.1f}s")

        if target_minutes > 0:
            target_sec = target_minutes * 60
            diff = duration_sec - target_sec
            if abs(diff) <= 10:
                msg = "목표 시간과 거의 딱 맞게 연습했어요 👏"
            elif diff > 10:
                msg = f"목표보다 약 **{diff:.0f}초 길어요.** 조금 줄여보면 좋아요."
            else:
                msg = f"목표보다 약 **{abs(diff):.0f}초 짧아요.** 예시나 설명을 조금 더 넣어보세요."
            st.info(f"🎯 목표 시간 ({target_minutes}분) 비교: {msg}")

        st.markdown("### ⏱️ 말하기 속도 코멘트")
        st.write(speaking_speed_comment(wpm))

        st.markdown("### 🗣️ 군더더기 표현 (최종)")
        if filler_counts:
            sorted_fillers = sorted(
                filler_counts.items(), key=lambda x: x[1], reverse=True
            )
            s = ", ".join([f"`{w}` × {c}" for w, c in sorted_fillers])
            st.write(s)
        else:
            st.write("눈에 띄는 군더더기 표현은 거의 없어요. 👍")

        if vol_db_val < vol_warn_db:
            st.warning(
                f"평균 볼륨이 낮습니다: {vol_db_val:.1f} dBFS < 임계 {vol_warn_db} dBFS\n\n"
                "마이크와의 거리를 줄이거나, 목소리를 조금 더 크게 내보세요."
            )
        else:
            st.success("볼륨은 발표용으로 무난한 수준입니다.")

        if pauses:
            st.markdown("### 🤫 긴 침묵 구간")
            long_pauses = [(s, e) for s, e in pauses if (e - s) >= 3.0]
            if long_pauses:
                for s_p, e_p in long_pauses[:10]:
                    st.write(
                        f"- {s_p:.1f}s ~ {e_p:.1f}s ({(e_p - s_p):.1f}s): "
                        "이 구간에 예시를 추가하거나, 다음 내용을 미리 예고해 보세요."
                    )
            else:
                st.write("3초 이상 긴 침묵은 거의 없습니다.")

    with tab2:
        st.subheader("🧩 타임라인 기반 피드백")
        if not timeline_chunks:
            st.info("전사 텍스트가 충분하지 않아 타임라인을 만들기 어렵습니다.")
        else:
            for i, ch in enumerate(timeline_chunks, start=1):
                with st.expander(f"{i} 구간: {ch['start']} ~ {ch['end']}"):
                    st.write(ch["text"])

    with tab3:
        st.subheader("📜 전체 전사 텍스트")
        if final_text:
            st.write(final_text)
        else:
            st.info(
                "전사 텍스트가 비어 있습니다. 마이크 입력 상태와 faster-whisper 설치를 확인해 주세요."
            )
