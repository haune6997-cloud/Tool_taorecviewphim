"""Module Semantic Matcher: Khớp các shot video với từng câu kịch bản theo ngữ nghĩa,
đặc trưng thị giác (khuôn mặt/chuyển động/bối cảnh) và ràng buộc tiến trình thời gian.
"""
from __future__ import annotations

import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from reviewcut.select import Block

# Tự động ưu tiên lưu cache model sang ổ D nếu có thư mục để tránh đầy ổ C
if os.path.exists("D:\\") and "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = r"D:\huggingface_cache"

_embedder = None

# Từ khóa nhận diện ý đồ thị giác trong câu kịch bản
_ACTION_WORDS = re.compile(
    r"\b(lao vào|tấn công|đòn|nổ|va chạm|chiến đấu|đánh|bắn|chạy|rượt|đuổi|phá hủy|xé toạc|sấm sét|tốc độ|tung chiêu|hạ gục|vung)\b",
    re.I,
)
_FACE_WORDS = re.compile(
    r"\b(bàng hoàng|ngơ ngác|ánh mắt|khuôn mặt|gương mặt|biểu cảm|nói|thốt lên|tuyên bố|hét|khóc|cười|sợ hãi|run rẩy|nhìn|trầm ngâm|suy tư)\b",
    re.I,
)
_DARK_WORDS = re.compile(
    r"\b(bóng tối|đêm|u ám|rùng rợn|bí ẩn|lạnh lẽo|hiểm địa|cạm bẫy|nguy hiểm|kinh hoàng)\b",
    re.I,
)


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return _embedder


def clean_text_for_embed(text: str) -> str:
    """Loại bỏ ký tự lạ và chuẩn hóa văn bản trước khi vector hóa."""
    t = re.sub(
        r"[^\w\sàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđĐ]",
        " ",
        text,
        flags=re.I,
    )
    return re.sub(r"\s+", " ", t).strip()


def _visual_bonus(sentence: str, shot) -> float:
    """Chấm điểm thưởng khi cảm xúc/hành động trong câu khớp với đặc trưng vật lý của khung hình."""
    bonus = 0.0
    if _ACTION_WORDS.search(sentence):
        bonus += 0.18 * getattr(shot, "motion", 0.0)
        if getattr(shot, "kind", "") == "hành động":
            bonus += 0.08
    if _FACE_WORDS.search(sentence):
        bonus += 0.20 * getattr(shot, "face", 0.0)
        if "nội tâm" in getattr(shot, "kind", "") or "đối thoại" in getattr(shot, "kind", ""):
            bonus += 0.08
    if _DARK_WORDS.search(sentence):
        if "tối" in getattr(shot, "kind", ""):
            bonus += 0.12
    if getattr(shot, "must", False):
        bonus += 0.06
    return bonus


def match_story_to_shots(script_sentences: list[str], candidate_shots: list, st) -> list[Block]:
    """Ghép nối từng câu kịch bản với các shot hình phù hợp nhất theo ngữ nghĩa + thị giác + thời gian."""
    main_shots = [s for s in candidate_shots if getattr(s, "role", "main") == "main"]
    outro_shots = [s for s in candidate_shots if getattr(s, "role", "main") == "outro"]

    if not main_shots or not script_sentences:
        return []

    model = get_embedder()
    film_len = max((s.end for s in main_shots), default=1.0)

    # 1. Chuẩn bị văn bản đại diện cho từng shot (Caption Vision AI + Nhãn + Thoại Whisper)
    shot_texts = []
    for s in main_shots:
        diag = " ".join(d.get("text", "") for d in getattr(s, "dialogue", []))
        cap = getattr(s, "caption", "")
        tags = " ".join(getattr(s, "tags", []) or [])
        kind = getattr(s, "kind", "")
        shot_texts.append(clean_text_for_embed(f"{cap} {kind} {tags} {diag}"))

    shot_embeddings = model.encode(shot_texts, normalize_embeddings=True, show_progress_bar=False)
    sentence_embeddings = model.encode(
        [clean_text_for_embed(s) for s in script_sentences],
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # Ma trận tương đồng ngữ nghĩa: [Số câu, Số shot]
    sim_matrix = np.dot(sentence_embeddings, shot_embeddings.T)

    from reviewcut.script import soften, words_per_sec
    wps = words_per_sec(st)

    used_shot_indices = set()
    blocks: list[Block] = []
    num_sentences = len(script_sentences)

    for idx, raw_sentence in enumerate(script_sentences):
        sentence = soften(raw_sentence) if getattr(st, "safety_filter", True) else raw_sentence
        needed_duration = max(2.5, len(sentence.split()) / wps)

        # Ràng buộc cửa sổ thời gian (Chronological Windowing) giữ mạch phim tuần tự
        frac = idx / max(num_sentences - 1, 1)
        center_time = frac * film_len
        window_size = film_len * 0.28
        min_time = max(0.0, center_time - window_size * 0.6)
        max_time = min(film_len, center_time + window_size * 1.2)

        valid_indices = [
            i for i, s in enumerate(main_shots)
            if i not in used_shot_indices and min_time <= s.start <= max_time
        ]
        if not valid_indices:
            valid_indices = [i for i in range(len(main_shots)) if i not in used_shot_indices]
        if not valid_indices:
            valid_indices = list(range(len(main_shots)))

        # Kết hợp điểm Semantic Vector + Điểm thưởng khớp hình ảnh (Action/Face/Dark) + Điểm chất lượng gốc
        scores = []
        for i in valid_indices:
            s = main_shots[i]
            sem_score = float(sim_matrix[idx][i])
            vis_score = _visual_bonus(sentence, s)
            base_score = 0.08 * getattr(s, "score", 0.0)
            # Phạt nhẹ nếu shot nằm quá xa tâm thời gian dự kiến của câu
            dist_penalty = 0.10 * abs(s.start - center_time) / max(film_len, 1.0)
            total_score = sem_score + vis_score + base_score - dist_penalty
            scores.append((total_score, i))

        scores.sort(key=lambda x: x[0], reverse=True)

        chosen_shots = []
        accumulated_dur = 0.0

        for _, shot_idx in scores:
            s = main_shots[shot_idx]
            chosen_shots.append(s)
            used_shot_indices.add(shot_idx)
            accumulated_dur += s.dur
            if accumulated_dur >= needed_duration:
                break

        # Sắp xếp các shot trong cùng một câu theo đúng thứ tự thời gian xuất hiện
        chosen_shots.sort(key=lambda x: x.start)

        dlg = []
        for s in chosen_shots:
            dlg.extend(getattr(s, "dialogue", []))

        block = Block(
            id=idx,
            shots=chosen_shots,
            text=sentence,
            role="main",
            dialogue=dlg,
        )
        blocks.append(block)

    # Bảo toàn khối Outro Montage nếu người dùng bật
    if getattr(st, "outro_montage", False) and outro_shots:
        blocks.append(Block(id=len(blocks), shots=outro_shots, text="", role="outro"))

    return blocks