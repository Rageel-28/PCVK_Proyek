"""Pipeline OCR klasik lengkap: gambar struk -> teks.

Alur:
    grayscale -> denoise -> binarisasi -> deskew
    -> segmentasi baris -> segmentasi karakter
    -> normalisasi 32x32 -> HOG -> Linear SVM
    -> rekonstruksi baris (spasi dari celah)
    -> koreksi leksikon

Tidak memakai library OCR siap pakai dan tidak memakai deep learning.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

import config as C
import deskew as DK
import features as F
import paper as PPR
import preprocess as PP
import resize_cubic as RC
import segment as SG
from lexicon import Lexicon
from render_synthetic import normalize_char


# ---------------------------------------------------------------------------
# Hasil
# ---------------------------------------------------------------------------
@dataclass
class CharPrediction:
    char: str
    confidence: float
    bbox: tuple[int, int, int, int]
    margin: float = 0.0


@dataclass
class LinePrediction:
    text: str
    confidence: float
    chars: list[CharPrediction] = field(default_factory=list)
    box: tuple[int, int] = (0, 0)


@dataclass
class OcrResult:
    text: str
    lines: list[LinePrediction]
    raw_text: str
    mean_confidence: float
    low_confidence_chars: int
    skew_angle: float
    lexicon_fixes: int

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "raw_text": self.raw_text,
            "mean_confidence": round(self.mean_confidence, 4),
            "low_confidence_chars": self.low_confidence_chars,
            "skew_angle": round(self.skew_angle, 2),
            "lexicon_fixes": self.lexicon_fixes,
            "n_lines": len(self.lines),
        }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class CharClassifier:
    """Linear SVM yang dieksekusi manual: argmax(coef @ x + intercept)."""

    def __init__(self, model_path: Path | None = None) -> None:
        path = model_path or (C.MODEL_DIR / "char_svm.json")
        if not path.exists():
            raise FileNotFoundError(
                f"{path} belum ada. Jalankan: python train_char.py"
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        svm = data["svm"]
        self.coef = np.asarray(svm["coef"], dtype=np.float32)
        self.intercept = np.asarray(svm["intercept"], dtype=np.float32)
        self.classes = [int(c) for c in svm["classes"]]
        self.charset = data.get("charset", C.CHARSET)
        self.threshold = float(data.get("confidence_threshold", C.CONFIDENCE_THRESHOLD))

    def predict(self, hog_vector: np.ndarray) -> tuple[str, float, float]:
        """Kembalikan (karakter, keyakinan, margin)."""
        return self.predict_restricted(hog_vector, None)

    def predict_restricted(
        self,
        hog_vector: np.ndarray,
        allowed: frozenset[str] | None = None,
    ) -> tuple[str, float, float]:
        """Prediksi dengan kandidat dibatasi ke himpunan karakter tertentu.

        Dipakai saat konteks sudah pasti: kolom harga hanya berisi angka
        dan pemisah, sehingga karakter seperti ``S``/``G``/``B`` tidak
        mungkin benar walaupun skor linearnya tinggi. Ini menyerang
        kesalahan yang paling merusak struk (nominal uang) tanpa
        melatih ulang model.
        """
        scores = self.coef @ hog_vector + self.intercept

        if allowed is not None:
            mask = np.full(len(self.classes), -np.inf, dtype=np.float32)
            for i, cls in enumerate(self.classes):
                if cls < len(self.charset) and self.charset[cls] in allowed:
                    mask[i] = 0.0
            if np.isfinite(mask).any():
                scores = scores + mask

        order = np.argsort(scores)[::-1]
        best = int(order[0])

        # Softmax atas skor linear sebagai keyakinan.
        finite = scores[np.isfinite(scores)]
        z = scores - finite.max()
        exp = np.where(np.isfinite(z), np.exp(z.astype(np.float64)), 0.0)
        total = exp.sum()
        confidence = float(exp[best] / total) if total > 0 else 0.0

        margin = float(scores[order[0]] - scores[order[1]]) if len(order) > 1 else float("inf")
        if not np.isfinite(margin):
            margin = 0.0
        idx = self.classes[best]
        char = self.charset[idx] if idx < len(self.charset) else "?"
        return char.upper(), confidence, margin


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class ClassicalOCR:
    def __init__(
        self,
        method: str = "sauvola",
        use_deskew: bool = True,
        use_lexicon: bool = False,
        use_paper_crop: bool = True,
        use_context: bool = True,
        model_path: Path | None = None,
    ) -> None:
        """Pipeline OCR klasik.

        ``use_lexicon`` default **False**. Koreksi kamus dirancang untuk
        data sintetis yang kosakatanya berasal dari katalog yang sama,
        tetapi pada 28 baris struk asli ia menurunkan exact match dari
        67,9% ke 60,7% (CER 7,46% -> 7,55%): nama produk yang benar
        seperti ``GAM`` dan ``BACTERIAL`` justru "dikoreksi" menjadi
        ``GA`` dan ``BATERAI``. Aktifkan hanya bila kamusnya memang
        mencakup kosakata struk yang sedang dibaca.
        """
        self.method = method
        self.use_deskew = use_deskew
        self.use_lexicon = use_lexicon
        self.use_paper_crop = use_paper_crop
        self.use_context = use_context
        self.classifier = CharClassifier(model_path)
        self.lexicon = Lexicon.from_dataset() if use_lexicon else None
        if self.lexicon:
            print(f"  leksikon: {len(self.lexicon.words)} kata")

    # -- tahap 1: citra -> mask tinta -------------------------------------
    def _binarize_once(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gray = PP.to_gray(image)
        gray = PP.denoise(gray, 3)
        ink = PP.binarize(gray, self.method)
        ink = PP.remove_small_components(ink, C.CHAR_MIN_INK)
        return gray, ink

    def _adaptive_scale(self, ink: np.ndarray) -> float:
        """Perbesar citra bila tinggi baris terlalu kecil untuk segmentasi."""
        lines = SG.segment_lines(ink)
        if not lines:
            return 1.0
        heights = [b - a + 1 for a, b in lines]
        med = float(np.median(heights))
        if med <= 0:
            return 1.0
        factor = C.TARGET_LINE_HEIGHT / med
        factor = max(1.0, min(C.MAX_UPSCALE, factor))
        # Bulatkan ke 0.5 supaya deterministik dan hemat.
        return round(factor * 2) / 2

    def binarize(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        gray = PP.to_gray(image)

        # Foto struk sering berlatar gelap; potong dulu area kertasnya.
        if self.use_paper_crop:
            gray, _box = PPR.crop_paper(gray, enabled=True)

        gray, ink = self._binarize_once(gray)

        factor = self._adaptive_scale(ink)
        if factor > 1.0:
            # Pembulatan mengikuti Dart (setengah ke genap) dan interpolasi
            # memakai modul yang sama dengan port Dart, supaya hasilnya
            # identik bit-per-bit.
            nw = max(1, int(round(gray.shape[1] * factor)))
            nh = max(1, int(round(gray.shape[0] * factor)))
            gray = RC.resize_bicubic(gray, nw, nh)
            gray, ink = self._binarize_once(gray)

        angle = 0.0
        if self.use_deskew:
            gray, ink, angle = DK.auto_deskew_pair(gray, ink)
        return gray, ink, angle

    # -- tahap 2: segmentasi + klasifikasi --------------------------------
    def read_line(
        self, gray: np.ndarray, ink: np.ndarray, line_box: tuple[int, int]
    ) -> LinePrediction:
        boxes = SG.segment_characters(ink, line_box)
        if not boxes:
            return LinePrediction(text="", confidence=0.0, box=line_box)

        vecs: list[np.ndarray] = []
        preds: list[CharPrediction] = []
        for bb in boxes:
            patch = normalize_char(gray, bb)
            vec = F.hog(patch)
            char, conf, margin = self.classifier.predict(vec)
            vecs.append(vec)
            preds.append(CharPrediction(char, conf, bb, margin))

        if self.use_context:
            preds = self._apply_context(preds, vecs, line_box)

        text = self._join_with_spaces(preds)
        conf = float(np.mean([p.confidence for p in preds])) if preds else 0.0
        return LinePrediction(text=text, confidence=conf, chars=preds, box=line_box)

    # -- konteks per kata -------------------------------------------------
    @staticmethod
    def _space_threshold(preds: list[CharPrediction]) -> float:
        """Ambang celah (piksel) yang dianggap batas kata.

        Versi lama memakai kelipatan median celah. Itu gagal pada struk
        karena tanda baca punya celah lebar di kedua sisinya: pada
        ``121,500`` celah di sekitar koma (12 px) lebih besar daripada
        median celah (5 px), sehingga spasi disisipkan di tengah angka.

        Lebar karakter jauh lebih stabil: dalam satu baris, celah di
        dalam kata selalu < 1x lebar karakter, sedangkan celah antar kata
        selalu > 1,5x. Ambang 0,9x lebar karakter memisahkan keduanya.
        """
        if not preds:
            return 2.0
        widths = [p.bbox[2] - p.bbox[0] + 1 for p in preds]
        median_w = float(np.median(widths))
        return max(2.0, median_w * C.SPACE_WIDTH_RATIO)

    def _word_groups(self, preds: list[CharPrediction]) -> list[list[int]]:
        """Kelompokkan indeks karakter menjadi kata memakai celah yang sama
        dengan :meth:`_join_with_spaces`, supaya batas kata konsisten."""
        if not preds:
            return []
        if len(preds) == 1:
            return [[0]]

        threshold = self._space_threshold(preds)
        groups: list[list[int]] = [[0]]
        for i in range(1, len(preds)):
            gap = preds[i].bbox[0] - preds[i - 1].bbox[2]
            glued = self._glued_by_separator(preds[i - 1].char, preds[i].char)
            if gap > threshold and not glued:
                groups.append([i])
            else:
                groups[-1].append(i)
        return groups

    def _apply_context(
        self,
        preds: list[CharPrediction],
        vecs: list[np.ndarray],
        line_box: tuple[int, int],
    ) -> list[CharPrediction]:
        """Prediksi ulang tiap kata dengan kandidat yang sesuai konteks.

        Struk thermal sangat sering menukar angka dengan huruf yang
        bentuknya mirip: ``0`` dengan ``O``/``D``, ``1`` dengan ``I``/``L``,
        ``5`` dengan ``S``, ``8`` dengan ``B``, ``2`` dengan ``Z``, dan
        koma dengan titik atau ``S``. Model karakter tidak bisa memutuskan
        sendiri karena bentuknya memang nyaris sama pada 32x32 piksel.

        Konteks yang dipakai:

        - **Ukuran glyph.** Tanda baca kecil (koma, titik, hubung) hanya
          menempati sebagian kecil tinggi baris. Pada 3-4 piksel, HOG
          praktis tidak informatif dan prediksi bisa berubah hanya karena
          satu baris piksel berbeda. Glyph seperti ini dibatasi ke
          ``.,-`` sehingga hasilnya deterministik.
        - **Isi kata.** Kata yang mayoritasnya angka dibatasi ke angka dan
          pemisah (``20,700`` tidak mungkin menjadi ``20. 70G``); kata yang
          mayoritasnya huruf dibatasi ke huruf.
        """
        out = list(preds)
        if not out:
            return out

        line_h = line_box[1] - line_box[0] + 1
        widths = [p.bbox[2] - p.bbox[0] + 1 for p in out]
        median_w = float(np.median(widths)) if widths else 1.0

        # 1. Glyph sangat kecil -> tanda baca.
        for i, p in enumerate(out):
            h = p.bbox[3] - p.bbox[1] + 1
            w = p.bbox[2] - p.bbox[0] + 1
            if line_h > 0 and h < line_h * C.TINY_GLYPH_HEIGHT_RATIO and w < median_w * C.TINY_GLYPH_WIDTH_RATIO:
                char, conf, margin = self.classifier.predict_restricted(
                    vecs[i], C.PUNCT_CHARSET
                )
                out[i] = CharPrediction(char, conf, p.bbox, margin)

        # 2. Konteks kata.
        for group in self._word_groups(out):
            if len(group) < 2:
                continue
            chars = [out[i].char for i in group]
            allowed = C.charset_for_token(chars)
            if allowed is None:
                continue

            for i in group:
                char, conf, margin = self.classifier.predict_restricted(vecs[i], allowed)
                out[i] = CharPrediction(char, conf, out[i].bbox, margin)
        return out

    @staticmethod
    def _join_with_spaces(preds: list[CharPrediction]) -> str:
        """Sisipkan spasi berdasarkan lebar karakter pada baris itu.

        Celah di dalam kata dan celah antar kata saling tumpang tindih bila
        diukur dengan median celah, sehingga ambangnya memakai lebar
        karakter (lihat :meth:`_space_threshold`).
        """
        if not preds:
            return ""
        if len(preds) == 1:
            return preds[0].char

        threshold = ClassicalOCR._space_threshold(preds)

        out: list[str] = [preds[0].char]
        for prev, cur in zip(preds, preds[1:]):
            gap = cur.bbox[0] - prev.bbox[2]
            if gap > threshold and not ClassicalOCR._glued_by_separator(prev.char, cur.char):
                out.append(" ")
            out.append(cur.char)
        return "".join(out)

    @staticmethod
    def _glued_by_separator(left: str, right: str) -> bool:
        """True bila dua karakter tidak boleh dipisah spasi.

        Pemisah ribuan pada struk selalu menempel: ``113,100`` tidak pernah
        ditulis ``113, 100``. Celah di sekitar koma/titik bisa melebihi
        ambang karena glyph koma sempit, jadi aturan ini mencegah spasi
        palsu di tengah nominal uang.
        """
        separators = ",."
        if left in separators and (right.isdigit() or right in separators):
            return True
        if right in separators and left.isdigit():
            return True
        return False

    # -- tahap 3: publik --------------------------------------------------
    def run(self, image: np.ndarray, apply_lexicon: bool | None = None) -> OcrResult:
        gray, ink, angle = self.binarize(image)
        line_boxes = SG.segment_lines(ink)

        lines: list[LinePrediction] = []
        for box in line_boxes:
            lp = self.read_line(gray, ink, box)
            if lp.text.strip():
                lines.append(lp)

        raw_text = "\n".join(lp.text for lp in lines)
        text = raw_text
        fixes = 0

        do_lex = self.use_lexicon if apply_lexicon is None else apply_lexicon
        if do_lex and self.lexicon is not None:
            corrected_lines = []
            for line in text.split("\n"):
                new_line, n = self.lexicon.correct_text(line)
                fixes += n
                corrected_lines.append(new_line)
            text = "\n".join(corrected_lines)

        all_chars = [c for lp in lines for c in lp.chars]
        mean_conf = float(np.mean([c.confidence for c in all_chars])) if all_chars else 0.0
        low = sum(1 for c in all_chars if c.confidence < self.classifier.threshold)

        return OcrResult(
            text=text,
            lines=lines,
            raw_text=raw_text,
            mean_confidence=mean_conf,
            low_confidence_chars=low,
            skew_angle=angle,
            lexicon_fixes=fixes,
        )

    def run_file(self, path: str | Path) -> OcrResult:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Gambar tidak terbaca: {path}")
        return self.run(img)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="OCR klasik untuk struk belanja")
    ap.add_argument("image", nargs="?", help="path gambar struk")
    ap.add_argument("--method", default="sauvola", choices=["sauvola", "otsu", "adaptive"])
    ap.add_argument("--no-deskew", action="store_true")
    ap.add_argument("--no-lexicon", action="store_true")
    ap.add_argument("--no-paper-crop", action="store_true")
    ap.add_argument("--json", action="store_true", help="cetak hasil sebagai JSON")
    args = ap.parse_args()

    if not args.image:
        ap.error("path gambar wajib diisi")

    print("Memuat model...")
    ocr = ClassicalOCR(
        method=args.method,
        use_deskew=not args.no_deskew,
        use_lexicon=not args.no_lexicon,
        use_paper_crop=not args.no_paper_crop,
    )

    result = ocr.run_file(args.image)

    if args.json:
        print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False))
        print(json.dumps({"lines": [{"text": l.text, "conf": round(l.confidence, 3)} for l in result.lines]}, indent=2, ensure_ascii=False))
    else:
        print("\n=== HASIL OCR ===")
        print(result.text)
        print("\n--- ringkasan ---")
        for k, v in result.as_dict().items():
            if k != "text":
                print(f"  {k:22}: {v}")


if __name__ == "__main__":
    main()
