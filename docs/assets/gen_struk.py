import os, sys, json
sys.path.insert(0, os.path.expanduser('~/development/htdocs/PCVK_Proyek/ML'))
os.chdir(os.path.expanduser('~/development/htdocs/PCVK_Proyek/ML'))
import numpy as np, cv2
import ocr.config as C
from ocr.preprocess import prepare, estimate_skew, binarize, to_gray, _paper_mask
from ocr.lines import detect_lines
from ocr.rules import filter_bands, classify_band
from ocr.words import segment_words
from ocr.chars import segment_glyphs
from ocr.features import extract_features_batch
from ocr.model import load_model
from ocr.lm import CharLM
from ocr.lexicon import Lexicon
from ocr.templates import Templates
from ocr.pipeline import recognize_page, _line_metrics, _word_probs
from PIL import Image

OUT = os.path.expanduser('~/development/htdocs/PCVK_Proyek/docs/assets/struk')
os.makedirs(OUT, exist_ok=True)

model = load_model()
lm = CharLM.load(C.MODEL_DIR / 'lm.json')
lex = Lexicon.load(C.MODEL_DIR / 'lexicon.json')
tpl = Templates.load(C.MODEL_DIR / 'templates.json')

def save(a, name):
    Image.fromarray(a).save(os.path.join(OUT, name))

report = {}
for name, key in [('indomaret_01.png', 'indomaret'), ('alfamart_01.jpeg', 'alfamart')]:
    src = os.path.expanduser('~/development/htdocs/PCVK_Proyek/testdata/real/' + name)
    img = cv2.imread(src)
    gray = to_gray(img)
    save(gray, f'{key}_1gray.png')
    paper = _paper_mask(gray)
    if paper is not None:
        save((paper * 255).astype(np.uint8), f'{key}_2paper.png')
    skew = estimate_skew(binarize(gray))
    prep = prepare(img)
    save(prep.binary, f'{key}_3bin.png')
    bands = detect_lines(prep.binary)
    kinds = [(classify_band(prep.binary, b), b) for b in bands]
    kept = filter_bands(prep.binary, bands)
    vis = cv2.cvtColor(prep.binary, cv2.COLOR_GRAY2BGR)
    color = {'text': (0, 180, 0), 'rule': (0, 140, 255), 'barcode': (255, 120, 0), 'graphic': (180, 0, 200)}
    for k, b in kinds:
        cv2.rectangle(vis, (b.left, b.top), (b.right, b.bottom), color.get(k, (0, 0, 255)), 3)
    save(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), f'{key}_4bands.png')
    res = recognize_page(img, model, lm, lex, tpl, prepared=prep)
    tot = [l for l in res.lines if l.text.upper().startswith(('TOTAL', 'TOTAI'))][0]
    band = [b for b in kept if abs(b.top - tot.top) < 5][0]
    x0, x1 = max(0, band.left - 20), min(prep.binary.shape[1], band.right + 20)
    y0, y1 = max(0, band.top - 14), min(prep.binary.shape[0], band.bottom + 14)
    crop = prep.binary[y0:y1, x0:x1]
    save(crop, f'{key}_5band.png')
    words = segment_words(crop, y0, x0)
    visw = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    for w in words:
        cv2.rectangle(visw, (w.left - x0, w.top - y0), (w.right - x0, w.bottom - y0), (0, 200, 0), 2)
    save(cv2.cvtColor(visw, cv2.COLOR_BGR2RGB), f'{key}_6words.png')
    wtarget = words[-1]
    wb = crop[wtarget.top - y0:wtarget.bottom - y0, wtarget.left - x0:wtarget.right - x0]
    H, W = wb.shape
    big = cv2.resize(wb, (W * 6, H * 6), interpolation=cv2.INTER_NEAREST)
    save(big, f'{key}_7word.png')
    gl = segment_glyphs(wb, wtarget.top, wtarget.left)
    visg = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
    for g in gl:
        cv2.rectangle(visg, ((g.left - wtarget.left) * 6, (g.top - wtarget.top) * 6),
                      ((g.right - wtarget.left) * 6, (g.bottom - wtarget.top) * 6), (0, 120, 255), 2)
    save(cv2.cvtColor(visg, cv2.COLOR_BGR2RGB), f'{key}_8glyphs.png')
    norm = np.full((32 + 12, (32 + 10) * len(gl), 3), 255, np.uint8)
    for i, g in enumerate(gl):
        c = wb[g.top - wtarget.top:g.bottom - wtarget.top, g.left - wtarget.left:g.right - wtarget.left]
        norm[6:38, i * 42:i * 42 + 32] = cv2.cvtColor(cv2.resize(c, (32, 32), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    save(cv2.cvtColor(norm, cv2.COLOR_BGR2RGB), f'{key}_9norm.png')
    rw, mh = _line_metrics(crop, y0, x0)
    p = _word_probs(crop, y0, x0, wtarget, model, rw, mh)
    per_glyph = []
    for i, row in enumerate(p):
        top = np.argsort(row)[::-1][:3]
        per_glyph.append([(C.IDX_TO_CHAR[t], round(float(row[t]), 3)) for t in top])
    report[key] = {
        'size': list(img.shape[:2]),
        'skew': round(float(skew), 2),
        'scale': round(float(prep.scale), 2),
        'bands': len(bands),
        'kinds': {k: sum(1 for kk, _ in kinds if kk == k) for k in set(k for k, _ in kinds)},
        'kept': len(kept),
        'total_band': [band.top, band.bottom, band.left, band.right],
        'words': len(words),
        'word_text_argmax': ''.join(C.IDX_TO_CHAR[i] for i in p.argmax(1)),
        'glyphs': len(gl),
        'per_glyph': per_glyph,
        'ocr_lines': [l.text for l in res.lines],
    }
print(json.dumps(report, indent=1, ensure_ascii=False))
with open(os.path.join(OUT, 'report.json'), 'w') as f:
    json.dump(report, f, indent=1, ensure_ascii=False)
