// Generator Dokumentasi_SmartReceipt_Analyzer.docx dari docs/*.md
// Jalankan: NODE_PATH=<dir dengan node_modules docx> node build_docx.js
// Mendukung: heading, paragraf, tabel (\| escape), code block, quote,
// bullet/ordered list (restart per grup), gambar markdown, hr, TOC, cover.
const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow,
  TableCell, WidthType, ShadingType, AlignmentType, BorderStyle,
  LevelFormat, PageBreak, TableOfContents, Header, Footer, PageNumber,
} = require('docx');

const ROOT = '/Users/aldisuryas/development/htdocs/PCVK_Proyek';
const DOCS = path.join(ROOT, 'docs');
const OUT = path.join(ROOT, 'Dokumentasi_SmartReceipt_Analyzer.docx');

const FILES = [
  '01-ringkasan-proyek.md', '02-arsitektur-aplikasi.md', '03-pipeline-ocr.md',
  '04-parser-struk.md', '05-klasifikasi-kategori.md', '06-evaluasi-pengujian.md',
  '07-build-rilis.md', '08-layar-ui.md', '09-analitik-prediksi.md',
  '10-praproses-aplikasi.md', '11-metodologi-pelatihan.md', '12-keputusan-desain.md',
  '13-glosarium-peta-kode.md', '14-contoh-struk.md',
];

const MONO = 'Consolas', BODY = 'Calibri';
const ACCENT = '1F3864', H2C = '2E5496', H3C = '44546A';
const TOTAL_W = 9638;

const dims = JSON.parse(fs.readFileSync(path.join(DOCS, 'assets/struk/dims.json'), 'utf8'));

// ---------- inline ----------
function inline(text, base = {}) {
  const runs = [];
  const re = /(`[^`]+`|\*\*[^*]+\*\*)/g;
  let last = 0, m;
  const t = text.replace(/\\\|/g, '\u0001'); // lindungi pipe escape
  while ((m = re.exec(t)) !== null) {
    if (m.index > last) runs.push(new TextRun({ text: t.slice(last, m.index), ...base }));
    const tok = m[0];
    if (tok.startsWith('`')) {
      runs.push(new TextRun({ text: tok.slice(1, -1).replace(/\u0001/g, '|'), font: MONO, size: (base.size ?? 22) - 2, color: 'A31515' }));
    } else {
      runs.push(new TextRun({ text: tok.slice(2, -2), bold: true, ...base }));
    }
    last = m.index + tok.length;
  }
  if (last < t.length) runs.push(new TextRun({ text: t.slice(last), ...base }));
  if (!runs.length) runs.push(new TextRun({ text: '', ...base }));
  return runs;
}
const plain = (s) => s.replace(/`([^`]+)`/g, '$1').replace(/\*\*([^*]+)\*\*/g, '$1').replace(/\\\|/g, '|');

// split baris tabel hanya pada pipe yang tidak di-escape
function splitRow(r) {
  const cells = []; let cur = '';
  for (let i = 0; i < r.length; i++) {
    if (r[i] === '\\' && r[i + 1] === '|') { cur += '\u0001'; i++; continue; }
    if (r[i] === '|') { cells.push(cur); cur = ''; continue; }
    cur += r[i];
  }
  cells.push(cur);
  let out = cells.map((c) => c.trim().replace(/\u0001/g, '|'));
  if (r.trim().startsWith('|') && out.length && out[0] === '') out = out.slice(1);
  if (r.trim().endsWith('|') && out.length && out[out.length - 1] === '') out = out.slice(0, -1);
  return out;
}

// ---------- tabel ----------
function buildTable(rows) {
  const header = rows[0];
  const n = header.length;
  const body = rows.slice(1).map((r) => { const c = r.slice(0, n); while (c.length < n) c.push(''); return c; });
  let widths = new Array(n).fill(Math.floor(TOTAL_W / n));
  widths[0] += TOTAL_W - widths.reduce((a, b) => a + b, 0);
  const cell = (t, hdr, w) => new TableCell({
    width: { size: w, type: WidthType.DXA },
    shading: hdr ? { type: ShadingType.CLEAR, fill: 'DEEAF6', color: 'auto' } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ spacing: { before: 0, after: 0 }, children: inline(t, { size: 19, bold: hdr ? true : undefined }) })],
  });
  return new Table({
    width: { size: TOTAL_W, type: WidthType.DXA },
    columnWidths: widths,
    borders: {
      top: { style: BorderStyle.SINGLE, size: 4, color: 'BFBFBF' },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: 'BFBFBF' },
      left: { style: BorderStyle.SINGLE, size: 4, color: 'BFBFBF' },
      right: { style: BorderStyle.SINGLE, size: 4, color: 'BFBFBF' },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: 'D9D9D9' },
      insideVertical: { style: BorderStyle.SINGLE, size: 2, color: 'D9D9D9' },
    },
    rows: [
      new TableRow({ tableHeader: true, cantSplit: true, children: header.map((h, i) => cell(h, true, widths[i])) }),
      ...body.map((r) => new TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, false, widths[i])) })),
    ],
  });
}

// ---------- code block ----------
function buildCode(lines) {
  return new Table({
    width: { size: TOTAL_W, type: WidthType.DXA },
    columnWidths: [TOTAL_W],
    borders: {
      top: { style: BorderStyle.SINGLE, size: 4, color: 'D9D9D9' },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: 'D9D9D9' },
      left: { style: BorderStyle.SINGLE, size: 4, color: 'D9D9D9' },
      right: { style: BorderStyle.SINGLE, size: 4, color: 'D9D9D9' },
      insideHorizontal: { style: BorderStyle.NONE }, insideVertical: { style: BorderStyle.NONE },
    },
    rows: [new TableRow({ children: [new TableCell({
      width: { size: TOTAL_W, type: WidthType.DXA },
      shading: { type: ShadingType.CLEAR, fill: 'F2F2F2', color: 'auto' },
      margins: { top: 100, bottom: 100, left: 140, right: 140 },
      children: lines.map((l) => new Paragraph({
        spacing: { before: 0, after: 0, line: 240 },
        children: [new TextRun({ text: l === '' ? ' ' : l, font: MONO, size: 17, color: '1F1F1F' })],
      })),
    })] })],
  });
}

// ---------- gambar ----------
const DPI = 96, CAP_H = 6.3;
function imgWidthIn(file, w, h) {
  const aspect = h / w;
  if (/_asli|_1gray/.test(file)) return aspect > 1.6 ? 2.1 : 2.7;
  if (/_3bin|_4bands/.test(file)) return aspect > 1.2 ? 3.9 : 6.2;
  if (/_5band|_6words|_line|_words/.test(file)) return 6.2;
  if (/_7word|_8glyphs|glyphs/.test(file)) return 4.6;
  if (/_9norm|_norm/.test(file)) return 4.0;
  return 4.5;
}
function buildImage(file, alt) {
  const p = path.join(DOCS, file);
  const ext = path.extname(file).slice(1).toLowerCase();
  const type = ext === 'jpg' || ext === 'jpeg' ? 'jpg' : 'png';
  const [w, h] = dims[path.basename(file)] || [800, 600];
  let wi = imgWidthIn(file, w, h), hi = wi * h / w;
  if (hi > CAP_H) { hi = CAP_H; wi = hi * w / h; }
  if (hi < 0.35) { hi = 0.35; wi = hi * w / h; }
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 120 },
    children: [new (require('docx').ImageRun)({
      type, data: fs.readFileSync(p),
      transformation: { width: Math.round(wi * DPI), height: Math.round(hi * DPI) },
      altText: { title: plain(alt).slice(0, 60), description: plain(alt).slice(0, 200), name: plain(alt).slice(0, 60) },
    })],
  });
}

// ---------- parse markdown ----------
function parse(md, file) {
  const blocks = [];
  const lines = md.split(/\r?\n/);
  let i = 0, firstH1 = true;
  const IMG = /^!\[([^\]]*)\]\(([^)]+)\)\s*$/;
  while (i < lines.length) {
    const line = lines[i];
    if (/^```/.test(line.trim())) {
      const buf = []; i++;
      while (i < lines.length && !/^```/.test(lines[i].trim())) { buf.push(lines[i]); i++; }
      i++; blocks.push({ k: 'code', lines: buf }); continue;
    }
    if (/^---+\s*$/.test(line) || /^\*\*\*+\s*$/.test(line)) { blocks.push({ k: 'hr' }); i++; continue; }
    const im = IMG.exec(line.trim());
    if (im) { blocks.push({ k: 'img', alt: im[1], file: im[2] }); i++; continue; }
    const h = /^(#{1,6})\s+(.*)$/.exec(line);
    if (h) {
      if (h[1].length === 1 && firstH1) { blocks.push({ k: 'chapter', text: h[2].trim(), file }); firstH1 = false; }
      else blocks.push({ k: 'heading', level: h[1].length === 1 ? 2 : h[1].length, text: h[2].trim() });
      i++; continue;
    }
    if (/^\s*\|/.test(line) && i + 1 < lines.length && /^\s*\|[\s:|-]+\|\s*$/.test(lines[i + 1])) {
      const rows = [splitRow(line)];
      i += 2;
      while (i < lines.length && /^\s*\|/.test(lines[i])) { rows.push(splitRow(lines[i])); i++; }
      blocks.push({ k: 'table', rows }); continue;
    }
    if (/^>\s?/.test(line)) {
      const buf = [];
      while (i < lines.length && /^>\s?/.test(lines[i])) { buf.push(lines[i].replace(/^>\s?/, '')); i++; }
      blocks.push({ k: 'quote', lines: buf }); continue;
    }
    const lb = /^(\s*)[-*+]\s+(.*)$/.exec(line);
    const lo = /^(\s*)(\d+)\.\s+(.*)$/.exec(line);
    if (lb || lo) {
      const items = [];
      while (i < lines.length) {
        const b = /^(\s*)[-*+]\s+(.*)$/.exec(lines[i]);
        const o = /^(\s*)(\d+)\.\s+(.*)$/.exec(lines[i]);
        if (b) items.push({ t: 'b', lvl: Math.min(2, Math.floor(b[1].length / 2)), x: b[2] });
        else if (o) items.push({ t: 'o', lvl: Math.min(2, Math.floor(o[1].length / 2)), x: o[3] });
        else if (/^\s{2,}\S/.test(lines[i]) && items.length) items[items.length - 1].x += ' ' + lines[i].trim();
        else break;
        i++;
      }
      blocks.push({ k: 'list', items }); continue;
    }
    if (line.trim() === '') { i++; continue; }
    const buf = [line.trim()]; i++;
    while (i < lines.length) {
      const l = lines[i];
      if (l.trim() === '' || /^(#{1,6}\s|```|>\s?|\s*[-*+]\s|\s*\d+\.\s|\s*\||!\[)/.test(l) || /^---+\s*$/.test(l)) break;
      buf.push(l.trim()); i++;
    }
    blocks.push({ k: 'para', text: buf.join(' ') });
  }
  assignRefs(blocks);
  return blocks;
}

// ---------- render ----------
const numberingConfig = [];
let listGroup = 0;
function newListRef(ordered) {
  const name = ordered ? `ordered-${listGroup++}` : 'bullet';
  if (ordered) numberingConfig.push({
    reference: name,
    levels: [
      { level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
      { level: 1, format: LevelFormat.LOWER_LETTER, text: '%2.', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
      { level: 2, format: LevelFormat.LOWER_ROMAN, text: '%3.', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1440, hanging: 360 } } } },
    ],
  });
  return name;
}
// pre-pass: assign references
function assignRefs(blocks) {
  let cur = null;
  for (const b of blocks) {
    if (b.k === 'list') {
      const hasO = b.items.some((it) => it.t === 'o');
      if (!cur || hasO || cur.ordered !== hasO) { cur = { ref: newListRef(hasO), ordered: hasO }; b.ref = cur.ref; }
      else b.ref = cur.ref;
    } else cur = null;
  }
}
assignRefs.all = () => {};

function render(blocks, out) {
  for (const b of blocks) {
    if (b.k === 'chapter') {
      out.push(new Paragraph({
        heading: HeadingLevel.HEADING_1, pageBreakBefore: true,
        border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: '8EAADB', space: 6 } },
        spacing: { before: 0, after: 260 },
        children: [new TextRun({ text: plain(b.text), bold: true, color: ACCENT, size: 38 })],
      }));
    } else if (b.k === 'heading') {
      const map = { 2: [HeadingLevel.HEADING_2, H2C, 26, 280, 120], 3: [HeadingLevel.HEADING_3, H3C, 23, 220, 100] };
      const [hl, col, sz, bef, aft] = map[b.level] || [HeadingLevel.HEADING_3, H3C, 23, 220, 100];
      out.push(new Paragraph({
        heading: hl, spacing: { before: bef, after: aft },
        children: [new TextRun({ text: plain(b.text), bold: true, color: col, size: sz })],
      }));
    } else if (b.k === 'para') {
      out.push(new Paragraph({ spacing: { before: 0, after: 140, line: 290 }, alignment: AlignmentType.JUSTIFIED, children: inline(b.text) }));
    } else if (b.k === 'table') {
      out.push(buildTable(b.rows));
      out.push(new Paragraph({ spacing: { before: 0, after: 120 }, children: [new TextRun({ text: '' })] }));
    } else if (b.k === 'code') {
      out.push(buildCode(b.lines));
      out.push(new Paragraph({ spacing: { before: 0, after: 120 }, children: [new TextRun({ text: '' })] }));
    } else if (b.k === 'quote') {
      for (const q of b.lines) out.push(new Paragraph({
        spacing: { before: 40, after: 40, line: 280 }, indent: { left: 360 },
        border: { left: { style: BorderStyle.SINGLE, size: 12, color: '8EAADB', space: 8 } },
        children: inline(q, { size: 21, color: '2F4F6F' }),
      }));
      out.push(new Paragraph({ spacing: { before: 0, after: 80 }, children: [new TextRun({ text: '' })] }));
    } else if (b.k === 'img') {
      try { out.push(buildImage(b.file, b.alt)); } catch (e) { console.error('IMG fail', b.file, e.message); }
    } else if (b.k === 'list') {
      for (const it of b.items) out.push(new Paragraph({
        numbering: { reference: it.t === 'o' ? b.ref : 'bullet', level: it.lvl },
        spacing: { before: 20, after: 60, line: 280 },
        children: inline(it.x),
      }));
    } else if (b.k === 'hr') {
      out.push(new Paragraph({
        spacing: { before: 200, after: 200 },
        border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: 'BFBFBF', space: 1 } },
        children: [new TextRun({ text: '' })],
      }));
    }
  }
}

// ---------- kumpulkan ----------
numberingConfig.push({
  reference: 'bullet',
  levels: [
    { level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
    { level: 1, format: LevelFormat.BULLET, text: '◦', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
    { level: 2, format: LevelFormat.BULLET, text: '▪', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1440, hanging: 360 } } } },
  ],
});

const children = [];
children.push(new Paragraph({ spacing: { before: 2400 }, alignment: AlignmentType.CENTER, children: [new TextRun({ text: 'SmartReceipt Analyzer', bold: true, size: 56, color: ACCENT })] }));
children.push(new Paragraph({ spacing: { before: 120 }, alignment: AlignmentType.CENTER, children: [new TextRun({ text: 'Dokumentasi Proyek', size: 32, color: '404040' })] }));
children.push(new Paragraph({ spacing: { before: 60, after: 800 }, alignment: AlignmentType.CENTER, children: [new TextRun({ text: 'Pengolahan Citra (PCVK)', size: 24, color: '595959' })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: '8EAADB', space: 6 } }, children: [new TextRun({ text: '' })] }));
children.push(new Paragraph({ spacing: { before: 800 }, alignment: AlignmentType.CENTER, children: [new TextRun({ text: 'Pengenalan teks struk belanja (OCR) yang dibangun dari nol —', size: 22, italics: true, color: '595959' })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: 'tanpa library OCR siap pakai dan tanpa deep learning.', size: 22, italics: true, color: '595959' })] }));
children.push(new Paragraph({
  pageBreakBefore: true, spacing: { before: 0, after: 240 },
  children: [new TextRun({ text: 'Daftar Isi', bold: true, size: 32, color: ACCENT })],
}));
children.push(new TableOfContents('Daftar Isi', { hyperlink: true, headingStyleRange: '1-3' }));

for (const f of FILES) {
  const md = fs.readFileSync(path.join(DOCS, f), 'utf8');
  render(parse(md, f), children);
}

const doc = new Document({
  numbering: { config: numberingConfig },
  styles: { default: {
    document: { run: { font: BODY, size: 22, color: '1F1F1F' }, paragraph: { spacing: { line: 290 } } },
    heading1: { run: { font: BODY, size: 32, bold: true, color: ACCENT }, paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0 } },
    heading2: { run: { font: BODY, size: 26, bold: true, color: H2C }, paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 1 } },
    heading3: { run: { font: BODY, size: 23, bold: true, color: H3C }, paragraph: { spacing: { before: 220, after: 100 }, outlineLevel: 2 } },
  } },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, right: 1134, bottom: 1134, left: 1134, header: 708, footer: 708 } } },
    headers: { default: new Header({ children: [new Paragraph({
      alignment: AlignmentType.RIGHT,
      border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: 'D9D9D9', space: 4 } },
      children: [new TextRun({ text: 'SmartReceipt Analyzer — Dokumentasi Proyek PCVK', size: 16, color: '808080' })],
    })] }) },
    footers: { default: new Footer({ children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [
        new TextRun({ text: 'Halaman ', size: 16, color: '808080' }),
        new TextRun({ children: [PageNumber.CURRENT], size: 16, color: '808080' }),
        new TextRun({ text: ' dari ', size: 16, color: '808080' }),
        new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 16, color: '808080' }),
      ],
    })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log('Wrote', OUT, (buf.length / 1024).toFixed(0) + ' KB'); });
