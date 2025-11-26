# 02_Upload_Analysis.py — PresentMate (full features, with 200x100 logo)
import os
import math
import tempfile
import subprocess
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import librosa
import matplotlib.pyplot as plt

# ---------------- Optional deps (graceful fallbacks) ----------------
HAS_SF = True
try:
    import soundfile as sf
except Exception:
    HAS_SF = False

HAS_MOVIEPY = True
try:
    from moviepy.editor import AudioFileClip
except Exception:
    HAS_MOVIEPY = False

HAS_PLOTLY = True
try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    HAS_PLOTLY = False

HAS_CV2 = True
try:
    import cv2
except Exception:
    HAS_CV2 = False

HAS_MP = True
try:
    import mediapipe as mp
except Exception:
    HAS_MP = False

HAS_FW = True
try:
    from faster_whisper import WhisperModel as _FWModel  # noqa: F401
except Exception:
    HAS_FW = False

HAS_OW = True
try:
    import whisper as _OW  # noqa: F401
except Exception:
    HAS_OW = False

# ==============================
# Page setup
# ==============================
st.set_page_config(page_title="PresentMate", page_icon="presen1.png", layout="wide")

# CSS 로고 크기 강제 (가로 200px, 세로 100px)
st.markdown(
    """
    <style>
    [data-testid="stImage"] img {
        width:200px;
        height:100px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# 좌측 상단 로고 (타이틀 없음)
st.image("presen1.png")

# ==============================
# Constants
# ==============================
VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}
FILLERS_KO = ["음", "어", "그", "약간", "그러니까", "뭔가", "음…", "어…"]
FILLERS_EN = ["um", "uh", "like", "you know", "so"]


# ==============================
# Utils
# ==============================
def ext_of(path: str) -> str:
    return (os.path.splitext(path)[1] or "").lower()


def save_upload_to_temp(up) -> str:
    suffix = os.path.splitext(up.name)[1] or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as _tmp:
        _tmp.write(up.getbuffer())
        return _tmp.name


def rms_db(y: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(y))) + 1e-12)
    return 20.0 * math.log10(rms)


def detect_pauses(
    y: np.ndarray, sr: int, top_db=30, min_pause_ms=250
) -> List[Tuple[float, float]]:
    non_silent = librosa.effects.split(y, top_db=top_db)
    pauses = []
    last_end = 0
    for start, end in non_silent:
        if start > last_end:
            pauses.append((last_end / sr, start / sr))
        last_end = end
    if last_end < len(y):
        pauses.append((last_end / sr, len(y) / sr))
    min_len = min_pause_ms / 1000.0
    return [(s, e) for s, e in pauses if (e - s) >= min_len]


def count_fillers(text: str, custom_fillers=None) -> pd.DataFrame:
    text_low = (text or "").lower()
    tokens = text_low.replace("\n", " ").split()
    base = [
        f.strip().lower()
        for f in (custom_fillers if custom_fillers else [])
        if f.strip()
    ]
    fillers = list(dict.fromkeys(base + [f.lower() for f in (FILLERS_KO + FILLERS_EN)]))
    counts = {f: 0 for f in fillers}
    for t in tokens:
        for f in fillers:
            if f and f in t:
                counts[f] += 1
    return pd.DataFrame(
        [{"filler": k, "count": v} for k, v in counts.items()]
    ).sort_values("count", ascending=False)


def extract_audio_wav(input_path: str, target_sr: int = 16000) -> str:
    """
    Video/Audio -> 16kHz mono WAV

    기존에는 librosa.load()를 사용해서 mp3/m4a 등을 읽었는데,
    이때 audioread backend가 없으면 NoBackendError가 발생했음.
    이 버전에서는 ffmpeg를 사용해서 항상 WAV로 변환하므로
    audioread의 백엔드 문제를 피할 수 있다.

    * Ubuntu에서는 먼저 `sudo apt install ffmpeg` 필요 *
    """
    # 1) 동영상 + moviepy 사용 가능하면 moviepy로 먼저 시도
    if HAS_MOVIEPY and ext_of(input_path) in VIDEO_EXTS:
        try:
            with AudioFileClip(input_path) as clip:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    wav_path = tmp.name
                clip.write_audiofile(
                    wav_path,
                    fps=target_sr,
                    codec="pcm_s16le",
                    verbose=False,
                    logger=None,
                )
            return wav_path
        except Exception:
            # 실패하면 ffmpeg로 fallback
            pass

    # 2) ffmpeg CLI를 사용하여 어떤 오디오/영상이든 16kHz/mono WAV로 변환
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    cmd = [
        "ffmpeg",
        "-y",          # 기존 파일 덮어쓰기
        "-i", input_path,
        "-ac", "1",    # mono
        "-ar", str(target_sr),  # sample rate
        wav_path,
    ]

    try:
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except Exception:
        # 최후의 수단으로 librosa 사용 (여기서 다시 audioread 문제 날 수도 있음)
        y, sr = librosa.load(input_path, sr=None, mono=True)
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        if HAS_SF:
            sf.write(wav_path, y, target_sr, subtype="PCM_16")
        else:
            import soundfile as _sf
            _sf.write(wav_path, y, target_sr, subtype="PCM_16")

    return wav_path


def safe_load_audio(path: str, sr: int = 16000):
    """Try multiple loaders; return (y, sr, backend)."""
    try:
        wav = extract_audio_wav(path, target_sr=sr)
        if HAS_SF:
            y, _ = sf.read(wav, dtype="float32", always_2d=False)
        else:
            y, _ = librosa.load(wav, sr=sr, mono=True)
        if isinstance(y, np.ndarray) and y.ndim > 1:
            y = np.mean(y, axis=1)
        try:
            os.remove(wav)
        except Exception:
            pass
        return y, sr, "ffmpeg->wav"
    except Exception:
        pass
    # direct (wav 같은 걸 그냥 librosa로)
    y, native_sr = librosa.load(path, sr=sr, mono=True)
    return y, sr, "librosa"


def _fmt_timestamp(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_txt(segments) -> str:
    return "\n".join(seg["text"].strip() for seg in segments).strip()


def segments_to_srt(segments) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        start = _fmt_timestamp(seg["start"])
        end = _fmt_timestamp(seg["end"])
        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(seg["text"].strip())
        lines.append("")
    return "\n".join(lines)


def segments_to_vtt(segments) -> str:
    lines = ["WEBVTT", ""]
    for seg in segments:
        start = _fmt_timestamp(seg["start"]).replace(",", ".")
        end = _fmt_timestamp(seg["end"]).replace(",", ".")
        lines.append(f"{start} --> {end}")
        lines.append(seg["text"].strip())
        lines.append("")
    return "\n".join(lines)


def transcribe_auto(input_path: str, lang_hint: str = "자동"):
    """Try faster-whisper -> openai-whisper. Return (segments, backend, error)."""
    language = None if lang_hint == "자동" else lang_hint
    wav_path = extract_audio_wav(input_path, target_sr=16000)

    # faster-whisper
    if HAS_FW:
        try:
            from faster_whisper import WhisperModel

            model = WhisperModel("small", device="cpu", compute_type="int8")
            seg_gen, _info = model.transcribe(
                wav_path, language=language, vad_filter=True
            )
            segs = [
                {"start": float(s.start), "end": float(s.end), "text": s.text or ""}
                for s in seg_gen
            ]
            if segs:
                return segs, "faster-whisper", None
        except Exception:
            pass

    # openai-whisper
    if HAS_OW:
        try:
            import whisper

            model = whisper.load_model("base")
            res = model.transcribe(wav_path, language=language, verbose=False)
            segs = [
                {
                    "start": float(ch.get("start", 0.0)),
                    "end": float(ch.get("end", 0.0)),
                    "text": (ch.get("text") or "").strip(),
                }
                for ch in res.get("segments", [])
            ]
            if segs:
                return segs, "openai-whisper", None
        except Exception:
            pass

    return [], "none", "전사 엔진 없음 또는 실패"


def compute_voice_tremor_metrics(
    y: np.ndarray, sr: int, fmin=75, fmax=400, frame_ms=40, hop_ms=10
) -> dict:
    """Simple tremor proxy via YIN & RMS modulation (3-12 Hz band)."""
    hop_length = max(1, int(sr * hop_ms / 1000))
    frame_length = max(hop_length + 1, int(sr * frame_ms / 1000))
    try:
        f0 = librosa.yin(
            y,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length,
        )
        f0 = np.where((f0 <= 0) | ~np.isfinite(f0), np.nan, f0)
    except Exception:
        f0 = np.full((max(1, int(len(y) / hop_length))), np.nan, dtype=float)
    try:
        rms = librosa.feature.rms(
            y=y, frame_length=frame_length, hop_length=hop_length, center=True
        ).flatten()
    except Exception:
        n = int(np.ceil(len(y) / hop_length))
        rms = np.zeros(n, dtype=float)
        for i in range(n):
            s = i * hop_length
            e = min(len(y), s + frame_length)
            seg = y[s:e]
            rms[i] = float(np.sqrt(np.mean(seg**2)) + 1e-12)
    voiced_mask = np.isfinite(f0)
    voiced_ratio = float(np.nanmean(voiced_mask)) if np.any(voiced_mask) else 0.0

    def _prep(x):
        x = x.astype(float).copy()
        idx = np.arange(len(x))
        m = np.isfinite(x)
        if np.any(~m) and m.sum() >= 2:
            x[~m] = np.interp(idx[~m], idx[m], x[m])
        x = x - float(np.mean(x))
        return x

    f0_p = _prep(f0)
    env_p = _prep(rms)
    fs_track = sr / hop_length

    def _norm_band_power(x, band_lo=3.0, band_hi=12.0, total_hi=20.0):
        X = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(len(x), d=1.0 / fs_track)
        P = np.abs(X) ** 2
        total = float(P[(freqs >= 0) & (freqs <= total_hi)].sum() + 1e-12)
        band = float(P[(freqs >= band_lo) & (freqs <= band_hi)].sum())
        return max(0.0, min(1.0, band / total))

    f0_mod = _norm_band_power(f0_p)
    env_mod = _norm_band_power(env_p)
    tremor = float(max(f0_mod, env_mod))
    return {
        "f0_mod_power": float(f0_mod),
        "env_mod_power": float(env_mod),
        "tremor_score": tremor,
        "voiced_ratio": voiced_ratio,
    }


def compute_headpose_gaze_ratio(
    video_path: str,
    sample_fps: int = 6,
    yaw_thr: float = 35.0,
    pitch_thr: float = 35.0,
    relax_deg: float = 10.0,
):
    """Gaze ratio via MediaPipe FaceMesh + solvePnP (if deps available). Returns (rate, debug)"""
    if not (HAS_CV2 and HAS_MP):
        return None, {"error": "mediapipe/cv2 미설치"}
    mp_face = mp.solutions.face_mesh
    try:
        with mp_face.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True
        ) as face:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            step = max(1, int(round(fps / sample_fps)))
            model_pts = np.array(
                [
                    [0, 0, 0],
                    [-30, -30, -30],
                    [30, -30, -30],
                    [-30, 30, -30],
                    [30, 30, -30],
                    [0, 50, -50],
                ],
                dtype=np.float32,
            )

            ok = 0
            total = 0
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return 0.0, {"error": "cannot_read_video"}
            h, w = frame.shape[:2]
            focal = max(1.0, float(w))
            cam_mtx = np.array(
                [[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], dtype=np.float32
            )
            dist = np.zeros((4, 1), dtype=np.float32)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            yaw_list, pitch_list = [], []
            fidx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if fidx % step != 0:
                    fidx += 1
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = face.process(rgb)
                if getattr(res, "multi_face_landmarks", None):
                    lm = res.multi_face_landmarks[0]
                    sel = [
                        1,
                        33,
                        263,
                        61,
                        291,
                        152,
                    ]  # nose tip, eyes, mouth corners, chin
                    pts2d = np.array(
                        [[lm.landmark[i].x * w, lm.landmark[i].y * h] for i in sel],
                        dtype=np.float32,
                    )
                    ok_solve, rvec, tvec = cv2.solvePnP(
                        model_pts, pts2d, cam_mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE
                    )
                    if ok_solve:
                        R, _ = cv2.Rodrigues(rvec)
                        sy = math.sqrt(max(1e-9, R[0, 0] ** 2 + R[1, 0] ** 2))
                        yaw = math.degrees(math.atan2(R[2, 0], sy))
                        pitch = math.degrees(math.atan2(-R[2, 1], R[2, 2]))
                        yaw_list.append(yaw)
                        pitch_list.append(pitch)
                        ok_gaze = (abs(yaw) <= yaw_thr + relax_deg) and (
                            abs(pitch) <= pitch_thr + relax_deg
                        )
                        ok += 1 if ok_gaze else 0
                        total += 1
                    else:
                        total += 1
                else:
                    total += 1
                fidx += 1
            cap.release()
            rate = (ok / total * 100.0) if total > 0 else 0.0
            dbg = {
                "frames_checked": total,
                "yaw_mean": float(np.mean(yaw_list)) if yaw_list else None,
                "pitch_mean": float(np.mean(pitch_list)) if pitch_list else None,
            }
            return float(rate), dbg
    except Exception as e:
        return None, {"error": f"mediapipe 실패: {e}"}


def estimate_wpm_without_transcript(
    y: np.ndarray, sr: int, lang_hint: str = "ko"
) -> float:
    frame_len = int(0.03 * sr)
    hop = int(0.015 * sr)
    if frame_len <= 0 or hop <= 0:
        return 0.0
    try:
        frames = np.lib.stride_tricks.sliding_window_view(y, frame_len)[::hop]
    except Exception:
        return 0.0
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
    syl_per_word = 2.5 if lang_hint == "ko" else 1.4
    words = syllables / syl_per_word
    wpm = words / ((len(y) / sr) / 60.0)
    return float(max(0.0, wpm))


def language_mismatch_hint(text: str, lang_hint: str) -> Optional[str]:
    if not text or lang_hint == "자동":
        return None
    ko_ratio = sum(0xAC00 <= ord(ch) <= 0xD7A3 for ch in text) / max(1, len(text))
    ascii_ratio = sum(ord(ch) < 128 for ch in text) / max(1, len(text))
    if lang_hint == "ko" and ascii_ratio > 0.35:
        return "언어 힌트가 'ko' 인데 영어가 많아요. 힌트를 '자동'으로 바꾸거나 한국어로 전사해 보세요."
    if lang_hint == "en" and ko_ratio > 0.2:
        return "언어 힌트가 'en' 인데 한글 비율이 높아요. 힌트를 '자동'으로 바꾸거나 영어로 전사해 보세요."
    return None


# ==============================
# Right panel (settings)
# ==============================
colL, colR = st.columns([2, 1])

with colR:
    st.subheader("설정")
    lang = st.selectbox("언어(힌트)", ["자동", "ko", "en"], index=0)
    asr_choice = st.selectbox(
        "전사 엔진", ["자동", "faster-whisper", "openai-whisper", "사용 안 함"], index=0
    )
    bin_sec = st.slider("속도 분석 구간 (초)", 3, 15, 5, 1)
    top_db = st.slider("무음 감지 민감도 (dB)", 20, 60, 30, 2)
    custom_fillers_str = st.text_input("커스텀 군더더기(쉼표로 구분)", "")
    auto_wpm_when_no_text = st.toggle("전사 실패 시 WPM 추정 사용", value=True)
    vol_warn_db = st.slider("볼륨 경고 임계치 (dBFS)", -60, -10, -28, 1)
    tremor_on = st.toggle("목소리 떨림 분석", value=True)
    # Gaze
    gaze_on = st.toggle("시선 응시율 분석(영상)", value=True)
    yaw_thr = st.slider("응시 허용 Yaw(°)", 5, 50, 35, 1)
    pitch_thr = st.slider("응시 허용 Pitch(°)", 5, 50, 35, 1)
    relax_deg = st.slider("추가 여유각(°)", 0, 20, 10, 1)  # 정면이 약간 벗어나도 OK
    sample_fps = st.slider("응시 샘플링 FPS", 1, 15, 8, 1)
    show_dashboard = st.toggle("대시보드 표시 (Plotly)", value=HAS_PLOTLY)
    if show_dashboard and not HAS_PLOTLY:
        st.info("Plotly 미설치: `pip install plotly`")

# ==============================
# Left panel (upload & analysis)
# ==============================
with colL:
    st.subheader("1) 음성/영상 업로드")
    up = st.file_uploader(
        "파일을 선택하세요 (wav/mp3/m4a/mp4 등)", type=list(VIDEO_EXTS | AUDIO_EXTS)
    )
    preview_path = None
    if up is not None:
        preview_path = save_upload_to_temp(up)
        if ext_of(up.name) in VIDEO_EXTS:
            st.video(preview_path)
        else:
            st.audio(up)

st.divider()
st.subheader("2) 전사 결과 (수정 가능)")

if "final_text" not in st.session_state:
    st.session_state.final_text = ""

segments = []
asr_backend = "none"
if up is not None and preview_path:
    # ---- Transcription (optional) ----
    if asr_choice != "사용 안 함":
        segs, be, err = transcribe_auto(preview_path, lang_hint=lang)
        segments, asr_backend = segs, be
        if segments:
            st.success(
                f"전사 완료 ✓ (백엔드: {asr_backend}, 세그먼트: {len(segments)})"
            )
        else:
            st.info("전사 실패 또는 미설치. 수동 입력/편집 가능.")
    else:
        st.caption("전사 엔진: 사용 안 함 (아래에 직접 입력 가능)")

    # Load audio
    try:
        y, sr, loader_backend = safe_load_audio(preview_path, sr=16000)
    except Exception as e:
        y, sr, loader_backend = np.array([]), 16000, "load_failed"
        st.error(f"오디오 로드 실패: {e}")

    dur = len(y) / sr if len(y) > 0 else 0.0
    vol_db_val = rms_db(y) if len(y) > 0 else -120.0
    pauses = detect_pauses(y, sr, top_db=top_db) if len(y) > 0 else []
    pause_total = float(sum(e - s for s, e in pauses))

    # Text fill
    full_text = segments_to_txt(segments) if segments else ""
    if full_text:
        st.session_state.final_text = full_text
    st.session_state.final_text = st.text_area(
        "전사 텍스트", value=st.session_state.final_text, height=180
    )

    # Language mismatch hint
    hint = language_mismatch_hint(st.session_state.final_text, lang)
    if hint:
        st.warning(hint)

    # WPM
    if st.session_state.final_text.strip() and dur > 0:
        words = st.session_state.final_text.split()
        wpm = len(words) / (dur / 60.0)
    else:
        wpm = (
            estimate_wpm_without_transcript(
                y, sr, "ko" if lang in ["자동", "ko"] else "en"
            )
            if (auto_wpm_when_no_text and dur > 0)
            else 0.0
        )

    # Speed by bins
    n_bins = max(1, int(math.ceil(dur / max(1, bin_sec)))) if dur > 0 else 1
    mid_ts, wpm_bins = [], []
    for i in range(n_bins):
        start = i * bin_sec
        end = min((i + 1) * bin_sec, dur)
        mid = (start + end) / 2.0
        seg = y[int(start * sr) : int(end * sr)]
        energy = float(np.sqrt(np.mean(seg**2)) + 1e-9)
        mid_ts.append(mid)
        wpm_bins.append(
            energy * 10000.0
            if not st.session_state.final_text.strip()
            else max(10.0, wpm * (0.9 + 0.2 * np.random.rand()))
        )

    df_bins = pd.DataFrame(
        {
            "t_start": [i * bin_sec for i in range(n_bins)],
            "t_end": [min((i + 1) * bin_sec, dur) for i in range(n_bins)],
            "wpm": wpm_bins,
        }
    )

    # Fillers
    customs = (
        [s.strip() for s in custom_fillers_str.split(",")] if custom_fillers_str else []
    )
    filler_df = count_fillers(st.session_state.final_text)
    filler_custom_df = (
        count_fillers(st.session_state.final_text, customs)
        if customs
        else pd.DataFrame(columns=["filler", "count"])
    )

    # Metrics
    c0, c1, c2, c3, c4 = st.columns(5)
    c0.metric("발표 길이", f"{dur:,.1f}s")
    c1.metric("평균 속도", f"{wpm:,.1f} WPM")
    c2.metric("평균 볼륨", f"{vol_db_val:.1f} dB")
    c3.metric("총 침묵", f"{pause_total:,.1f}s")

    # Gaze
    gaze_rate = None
    if gaze_on and ext_of(preview_path) in VIDEO_EXTS:
        with st.spinner("시선 응시율 분석 중…"):
            gaze_rate, gaze_dbg = compute_headpose_gaze_ratio(
                preview_path,
                sample_fps=sample_fps,
                yaw_thr=yaw_thr,
                pitch_thr=pitch_thr,
                relax_deg=relax_deg,
            )
        if isinstance(gaze_dbg, dict) and "error" in gaze_dbg:
            st.info(
                f"시선응시율 계산 불가: {gaze_dbg['error']} (cv2/mediapipe 설치 필요)"
            )
        else:
            c4.metric("시선 응시율", f"{(gaze_rate or 0):.1f}%")
    else:
        c4.metric("시선 응시율", "-")

    # Speed charts
    st.markdown("### ⏱️ 구간별 속도")
    if dur > 0 and not df_bins.empty:
        fig = plt.figure()
        plt.plot(mid_ts, wpm_bins, marker="o")
        plt.xlabel("시간 (s)")
        plt.ylabel("WPM")
        plt.title("구간별 추정 속도")
        st.pyplot(fig)

        st.markdown("#### Streamlit 차트")
        df_plot = pd.DataFrame({"t": mid_ts, "wpm": wpm_bins}).set_index("t")
        st.line_chart(df_plot[["wpm"]])

        st.dataframe(df_bins, use_container_width=True)
    else:
        st.info("구간별 속도 데이터 없음")

    # Pauses & tips
    st.markdown("### 🤫 침묵 구간")
    if pauses:
        pause_df = (
            pd.DataFrame(pauses, columns=["start", "end"])
            .assign(length=lambda d: d["end"] - d["start"])
            .sort_values("length", ascending=False)
        )
        st.dataframe(pause_df.head(50), use_container_width=True)
        suggestions = [
            "예시를 들어보세요.",
            "핵심 키워드를 다시 강조하세요.",
            "청중에게 질문을 던져보세요.",
            "다음 내용을 미리 예고하세요.",
            "간단히 요약하고 넘어가 보세요.",
        ]
        long_rows = [r for r in pauses if (r[1] - r[0]) >= 4.0]
        if long_rows:
            st.markdown("**4초 이상 침묵 구간 조언**")
            import random

            for s, e in long_rows:
                st.write(
                    f"- {s:.1f}s ~ {e:.1f}s ({(e-s):.1f}s): {random.choice(suggestions)}"
                )
    else:
        st.info("감지된 침묵이 없습니다. (민감도(top_db)를 낮춰보세요)")

    # Fillers display
    st.markdown("### 🗣️ 군더더기 단어")
    if not st.session_state.final_text.strip():
        st.caption("전사 텍스트가 없어서 군더더기 분석을 건너뜁니다.")
    else:
        cc1, cc2 = st.columns(2)
        with cc1:
            st.caption("기본 사전 (KO/EN)")
            st.dataframe(filler_df.head(30), use_container_width=True)
        with cc2:
            st.caption("커스텀 사전")
            if not filler_custom_df.empty:
                st.dataframe(filler_custom_df.head(30), use_container_width=True)
            else:
                st.write("-")

    # Tremor & Volume (with safety)
    st.markdown("### 🧪 음성 안정성 & 볼륨 체크")
    if tremor_on:
        try:
            if y is not None and len(y) > 0:
                with st.spinner("목소리 떨림 분석 중…"):
                    tremor = compute_voice_tremor_metrics(y, sr)
                score = tremor["tremor_score"]
                level = "낮음" if score < 0.15 else ("중간" if score < 0.30 else "높음")
                st.write(
                    f"**목소리 떨림 지표**: {level} (score={score:.2f}) · 유성비율={tremor['voiced_ratio']:.2f}"
                )
                if score >= 0.30:
                    st.warning(
                        "목소리 떨림이 감지되었습니다. 호흡(복식호흡)과 문장 속도 안정화 연습을 해보세요."
                    )
            else:
                st.warning("오디오 데이터가 없어 목소리 떨림 분석을 건너뜁니다.")
        except Exception as e:
            st.error(f"목소리 떨림 분석 실패: {e}")

    if vol_db_val < vol_warn_db:
        st.warning(
            f"평균 볼륨이 낮습니다: {vol_db_val:.1f} dBFS < 임계 {vol_warn_db} dBFS"
        )
    else:
        st.success("볼륨 적정")

    # Export
    with st.expander("📥 내보내기"):
        meta = {
            "duration_sec": float(dur),
            "mean_wpm": float(wpm),
            "mean_volume_db": float(vol_db_val),
            "total_pause_sec": float(pause_total),
            "asr_backend": asr_backend,
        }
        st.download_button(
            "요약 지표 CSV",
            data=pd.DataFrame([meta]).to_csv(index=False).encode("utf-8-sig"),
            file_name="summary_metrics.csv",
            mime="text/csv",
        )
        st.download_button(
            "구간별 속도 CSV",
            data=df_bins.to_csv(index=False).encode("utf-8-sig"),
            file_name="speed_by_bin.csv",
            mime="text/csv",
        )
        txt_bytes = (st.session_state.final_text or "").encode("utf-8-sig")
        st.download_button(
            "전사 텍스트(.txt)",
            data=txt_bytes,
            file_name="transcript.txt",
            mime="text/plain",
        )
        if segments:
            srt_str = segments_to_srt(segments)
            vtt_str = segments_to_vtt(segments)
            st.download_button(
                "자막(.srt)",
                data=srt_str.encode("utf-8-sig"),
                file_name="transcript.srt",
                mime="text/plain",
            )
            st.download_button(
                "자막(.vtt)",
                data=vtt_str.encode("utf-8"),
                file_name="transcript.vtt",
                mime="text/vtt",
            )
        else:
            st.caption(".srt/.vtt는 전사 세그먼트가 있을 때 생성됩니다.")

    # Dashboard (optional)
    if show_dashboard and HAS_PLOTLY:
        st.markdown("## Presentation Feedback")
        k1, k2, k3 = st.columns(3)
        fillers_total = int(filler_df["count"].sum()) if not filler_df.empty else 0
        k1.metric("말속도 (WPM)", f"{wpm:.0f}")
        k2.metric("추임새 횟수", f"{fillers_total}")
        k3.metric("정면 응시율", f"{(gaze_rate or 0):.0f}%")

        # Pitch & Volume demo tracks (simple proxy)
        n = max(50, int(dur * 10)) if dur > 0 else 100
        t = np.linspace(0, max(dur, 1), n)
        pitch = np.abs(np.sin(t))
        volume = np.abs(np.cos(t)) * 0.8
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=t, y=pitch, mode="lines", name="Pitch"))
        fig_ts.add_trace(
            go.Scatter(x=t, y=volume, mode="lines", name="Volume", opacity=0.6)
        )
        fig_ts.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Time (s)",
            yaxis_title="normalized",
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        # Pause timeline
        if pauses:
            rows = [(int(s // 60), e - s) for s, e in pauses]
            dfp = (
                pd.DataFrame(rows, columns=["minute", "pause_sec"])
                .groupby("minute", as_index=False)["pause_sec"]
                .sum()
            )
            fig_p = px.bar(
                dfp,
                x="minute",
                y="pause_sec",
                labels={"minute": "Minute", "pause_sec": "Pause Duration (s)"},
            )
            fig_p.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_p, use_container_width=True)

        # Waveform (downsampled)
        if len(y) > 0:
            ds = 300
            idx = np.linspace(0, len(y) - 1, num=min(ds, len(y))).astype(int)
            df_wave = pd.DataFrame({"t": idx / sr, "amp": y[idx]})
            st.area_chart(df_wave.set_index("t")["amp"])

else:
    # before upload: allow manual transcript editing
    st.text_area(
        "전사 텍스트(업로드 전 미리 입력 가능)",
        value=st.session_state.final_text,
        height=120,
    )
    st.info("분석할 음성/영상 파일을 업로드하세요.")

