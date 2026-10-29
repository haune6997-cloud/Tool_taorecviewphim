"""Cấu hình, thể loại (Genre Bias), preset giọng đọc, bộ lọc từ ngữ."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

CONFIG_DIR = Path.home() / ".reviewcut"
CONFIG_FILE = CONFIG_DIR / "config.json"

# ---------------------------------------------------------------- thể loại
# w_*: trọng số chọn cảnh nòng cốt (Genre Bias). Tổng không cần bằng 1.
#   motion  : chuyển động khung hình (cảnh hành động / chiến đấu)
#   face    : khuôn mặt cận cảnh (nội tâm / đối thoại)
#   plot    : câu thoại "nút thắt" do Whisper bóc tách
#   audio   : đỉnh âm thanh (la hét, nổ, nhạc cao trào)
#   dark    : khung hình tối / u ám (kinh dị, giật gân)
GENRES: dict[str, dict] = {
    "KINH DỊ / GIẬT GÂN / TÂM LÝ": dict(
        w_motion=0.8, w_face=1.1, w_plot=1.4, w_audio=1.2, w_dark=0.9,
        rate=-4, pitch=-2, voice="hoaimy", grade="cold", zoom=True,
        style="U ám, rùng rợn, thì thào gài người, ngưng đọng hồi hộp. Câu ngắn, ngắt nhịp, tạo cảm giác sắp có chuyện.",
        music="tense dark ambient"),
    "HÀNH ĐỘNG / CHIẾN ĐẤU": dict(
        w_motion=1.6, w_face=0.5, w_plot=1.0, w_audio=1.2, w_dark=0.1,
        rate=8, pitch=0, voice="namtram", grade="teal_orange", zoom=True,
        style="Dồn dập, mạnh mẽ, nhịp nhanh, nhấn vào các pha va chạm và bước ngoặt.",
        music="epic action drums"),
    "TÌNH CẢM / TÂM LÝ XÃ HỘI": dict(
        w_motion=0.3, w_face=1.5, w_plot=1.3, w_audio=0.5, w_dark=0.0,
        rate=-6, pitch=0, voice="hoaimy", grade="warm", zoom=False,
        style="Nhẹ nhàng, giàu cảm xúc, nhấn vào mối quan hệ và lựa chọn của nhân vật.",
        music="soft piano emotional"),
    "HÀI HƯỚC / GIA ĐÌNH": dict(
        w_motion=0.8, w_face=1.0, w_plot=1.2, w_audio=0.8, w_dark=0.0,
        rate=10, pitch=4, voice="hoatngon", grade="warm", zoom=False,
        style="Tinh nghịch, dí dỏm, nhịp nhanh, chêm câu đùa vừa phải.",
        music="light playful"),
    "KHOA HỌC VIỄN TƯỞNG / SIÊU ANH HÙNG": dict(
        w_motion=1.3, w_face=0.7, w_plot=1.3, w_audio=1.0, w_dark=0.3,
        rate=4, pitch=0, voice="namminh", grade="teal_orange", zoom=True,
        style="Hoành tráng, giải thích nhanh thiết lập thế giới, nhấn mạnh bước ngoặt công nghệ / sức mạnh.",
        music="cinematic sci-fi"),
    "TRINH THÁM / HÌNH SỰ": dict(
        w_motion=0.6, w_face=1.2, w_plot=1.8, w_audio=0.6, w_dark=0.4,
        rate=0, pitch=-1, voice="namminh", grade="cold", zoom=False,
        style="Lạnh lùng, logic, dẫn dắt từng manh mối; câu thoại nút thắt phải được giữ lại.",
        music="noir suspense"),
    "CỔ TRANG / KIẾM HIỆP": dict(
        w_motion=1.2, w_face=0.9, w_plot=1.2, w_audio=0.8, w_dark=0.1,
        rate=0, pitch=0, voice="namtram", grade="warm", zoom=True,
        style="Trang trọng, kể chuyện như thuyết thư, nhấn ân oán và các màn giao đấu.",
        music="traditional epic"),
}
DEFAULT_GENRE = "KINH DỊ / GIẬT GÂN / TÂM LÝ"

# ---------------------------------------------------------------- giọng đọc (edge-tts)
VOICES: dict[str, dict] = {
    "hoaimy":   dict(label="Hoài My – Nữ chuẩn",            edge="vi-VN-HoaiMyNeural",  rate=0,  pitch=0),
    "namminh":  dict(label="Nam Minh – Nam chuẩn",          edge="vi-VN-NamMinhNeural", rate=0,  pitch=0),
    "hoatngon": dict(label="Cô gái hoạt ngôn (review nhanh)", edge="vi-VN-HoaiMyNeural",  rate=14, pitch=4),
    "tuoitan":  dict(label="Nữ tươi tắn (nhẹ, cao)",        edge="vi-VN-HoaiMyNeural",  rate=6,  pitch=8),
    "namtram":  dict(label="Nam trầm (hành động)",          edge="vi-VN-NamMinhNeural", rate=6,  pitch=-4),
}

ASPECTS = {
    "16:9 (YouTube 1920x1080)": "16:9",
    "Cinemascope 2.39:1 (có viền đen)": "2.39",
    "9:16 dọc – nền mờ (TikTok/Shorts)": "9:16blur",
    "9:16 dọc – cắt giữa": "9:16crop",
    "1:1 vuông": "1:1",
}
GRADES = {"Không chỉnh màu": "none", "Cinema Teal & Orange": "teal_orange",
          "Lạnh – u ám": "cold", "Ấm – hoài niệm": "warm"}

# "Khiên vàng": thay từ ngữ bạo lực / nhạy cảm trong LỜI BÌNH để tránh bị hạn chế quảng cáo.
SOFTEN = {
    "giết chết": "hạ gục", "giết hại": "hạ gục", "giết": "hạ gục", "sát hại": "hạ gục",
    "tự tử": "tự kết thúc", "máu me": "vết thương", "đẫm máu": "khốc liệt", "máu": "vết thương",
    "xác chết": "thi thể", "chém": "tấn công", "đâm": "tấn công", "bắn chết": "bắn hạ",
    "tra tấn": "hành hạ", "hãm hiếp": "xâm hại", "cưỡng bức": "xâm hại",
    "khỏa thân": "không mảnh vải", "ma túy": "chất cấm", "chết": "ra đi",
}


@dataclass
class Settings:
    # đầu vào
    video: str = ""
    workdir: str = ""
    srt_path: str = ""                 # (tuỳ chọn) phụ đề sẵn có – bỏ qua Whisper
    title: str = ""                    # tên phim (giúp AI viết kịch bản)
    # API
    gemini_key: str = ""
    gemini_model: str = "gemini-1.5-flash"
    # Whisper
    whisper_model: str = "small"       # tiny/base/small/medium/large-v3
    whisper_lang: str = "auto"
    whisper_device: str = "auto"
    # cắt cảnh
    extraction_mode: str = "highlight"  # highlight | all
    target_minutes: float = 10.0
    micro_cut: bool = True
    shot_len: float = 3.2
    shot_min: float = 2.0
    shot_max: float = 3.5
    skip_head_sec: float = 0.0
    skip_tail_sec: float = 0.0
    auto_skip_credits: bool = True
    highlight_priority: bool = True
    block_sec: float = 12.0            # độ dài 1 đoạn lời bình
    outro_montage: bool = True
    outro_shots: int = 4
    # thể loại + giọng
    genre: str = DEFAULT_GENRE
    voice: str = "auto"                # auto = theo thể loại
    speed: float = 1.0                 # tốc độ đọc nhân thêm
    safety_filter: bool = True
    # hình ảnh
    aspect: str = "16:9"
    grade: str = "auto"                # auto = theo thể loại
    zoom: bool = True
    speed_ramp: float = 1.08           # 1.0 = tắt
    crop_bottom: bool = True           # cắt dải phụ đề cứng dưới đáy
    vignette: bool = False
    grain: bool = False
    # âm thanh / phụ đề
    music_path: str = ""               # file hoặc thư mục
    music_volume: float = 0.18
    orig_volume: float = 0.22
    subtitles: bool = True
    sub_font: str = "Arial"
    sub_size: float = 0.052            # tỉ lệ theo chiều cao khung
    encoder: str = "libx264"           # libx264 | h264_nvenc
    fps: int = 30

    def genre_cfg(self) -> dict:
        return GENRES.get(self.genre, GENRES[DEFAULT_GENRE])

    def resolved_voice(self) -> str:
        return self.genre_cfg()["voice"] if self.voice == "auto" else self.voice

    def resolved_grade(self) -> str:
        return self.genre_cfg()["grade"] if self.grade == "auto" else self.grade

    def target_sec(self) -> float:
        return max(30.0, self.target_minutes * 60.0)


def load_settings() -> Settings:
    try:
        raw = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return Settings()
    names = {f.name for f in fields(Settings)}
    return Settings(**{k: v for k, v in raw.items() if k in names})


def save_settings(s: Settings, keep_keys: bool = True) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    d = asdict(s)
    if not keep_keys:
        d["gemini_key"] = ""
    CONFIG_FILE.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
