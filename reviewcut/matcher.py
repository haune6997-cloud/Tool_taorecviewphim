"""Module Semantic Matcher: Khớp các shot video với từng câu kịch bản theo ngữ nghĩa
và ràng buộc tiến trình thời gian (Chronologically-Constrained Semantic Alignment).
"""
from __future__ import annotations

import re
import numpy as np
from sentence_transformers import SentenceTransformer
from reviewcut.select import Block

# Tải model nhúng đa ngữ siêu nhẹ
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return _embedder


def clean_text_for_embed(text: str) -> str:
    """Loại bỏ ký tự lạ và chuẩn hóa văn bản trước khi vector hóa."""
    t = re.sub(r"[^\w\sàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđĐ]", " ", text, flags=re.I)
    return re.sub(r"\s+", " ", t).strip()


def match_story_to_shots(script_sentences: list[str], candidate_shots: list, st) -> list[Block]:
    """Ghép nối từng câu kịch bản với các shot hình phù hợp nhất theo ngữ nghĩa và thời gian."""
    if not candidate_shots or not script_sentences:
        return []

    model = get_embedder()
    film_len = max((s.end for s in candidate_shots), default=1.0)

    # 1. Chuẩn bị dữ liệu văn bản đại diện cho từng shot (Caption + Thoại Whisper)
    shot_texts = []
    for s in candidate_shots:
        diag = " ".join(d.get("text", "") for d in getattr(s, "dialogue", []))
        cap = getattr(s, "caption", "")
        shot_texts.append(clean_text_for_embed(f"{cap} {diag}"))

    shot_embeddings = model.encode(shot_texts, normalize_embeddings=True, show_progress_bar=False)
    sentence_embeddings = model.encode([clean_text_for_embed(s) for s in script_sentences], normalize_embeddings=True, show_progress_bar=False)

    # Ma trận tương đồng ngữ nghĩa: [Số câu, Số shot]
    sim_matrix = np.dot(sentence_embeddings, shot_embeddings.T)

    from reviewcut.script import words_per_sec
    wps = words_per_sec(st)

    used_shot_indices = set()
    blocks = []
    num_sentences = len(script_sentences)

    for idx, sentence in enumerate(script_sentences):
        # Tính thời lượng tối thiểu cần có cho câu đọc này
        needed_duration = max(2.5, len(sentence.split()) / wps)

        # Ràng buộc cửa sổ thời gian (Chronological Windowing):
        # Câu ở vị trí thứ k chỉ tìm trong khoảng [k - window, k + window] của phim gốc để tránh nhảy cảnh lộn xộn
        frac = idx / max(num_sentences, 1)
        center_time = frac * film_len
        window_size = film_len * 0.35  # Cho phép tìm trong biên độ 35% thời lượng phim
        min_time = max(0.0, center_time - window_size * 0.7)
        max_time = min(film_len, center_time + window_size * 1.3)

        # Lọc danh sách shot nằm trong cửa sổ thời gian
        valid_indices = [
            i for i, s in enumerate(candidate_shots)
            if i not in used_shot_indices and min_time <= s.start <= max_time
        ]

        if not valid_indices:
            # Nếu hết cảnh trong cửa sổ, mở rộng tìm các cảnh chưa dùng bất kỳ
            valid_indices = [i for i in range(len(candidate_shots)) if i not in used_shot_indices]

        if not valid_indices:
            # Nếu kho cảnh đã dùng hết sạch, cho phép dùng lại các shot đẹp nhất
            valid_indices = list(range(len(candidate_shots)))

        # Sắp xếp các shot theo điểm tương đồng ngữ nghĩa giảm dần
        scores = [(sim_matrix[idx][i], i) for i in valid_indices]
        scores.sort(key=lambda x: x[0], reverse=True)

        chosen_shots = []
        accumulated_dur = 0.0

        for _, shot_idx in scores:
            s = candidate_shots[shot_idx]
            chosen_shots.append(s)
            used_shot_indices.add(shot_idx)
            accumulated_dur += s.dur
            if accumulated_dur >= needed_duration:
                break

        # Sắp xếp các shot trong cùng một câu theo đúng thứ tự thời gian xuất hiện
        chosen_shots.sort(key=lambda x: x.start)

        block = Block(
            id=idx,
            shots=chosen_shots,
            dur=accumulated_dur,
            role="main"
        )
        block.text = sentence
        blocks.append(block)

    return blocks