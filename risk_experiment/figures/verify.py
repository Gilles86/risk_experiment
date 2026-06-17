"""Verify exported figure panels against the Nature Communications artwork
rules + the editor's instructions (brief section 6).

For each PDF it reports:
  * page width x height in mm, and whether it fits a Nature column width
    (88 / 120 / 180 mm, + small tolerance) and within A4 height (<= 230 mm);
  * the minimum rendered font size in pt (must be >= 6 pt; we want >= 7 pt);
  * fonts embedded as editable text, not Type 3 (paths);
  * colour space is RGB, not CMYK.

Designing at the true figsize and never rescaling on export means the rcParams
point sizes equal the rendered pt -- this is the belt-and-suspenders check.

Usage
-----
    python -m risk_experiment.figures.verify              # all panels in repo
    python -m risk_experiment.figures.verify a.pdf b.pdf  # specific files
"""
import glob
import os.path as op
import subprocess
import sys

from risk_experiment.figures import style

PT_PER_MM = 1 / 0.352778
NATURE_WIDTHS_MM = [88, 120, 180]
WIDTH_TOL_MM = 4
MAX_HEIGHT_MM = 230
MIN_FONT_PT = 6.0
WANT_FONT_PT = 7.0


def page_size_mm(pdf):
    out = subprocess.run(['pdfinfo', pdf], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith('Page size:'):
            # "Page size:      340.16 x 172.8 pts"
            parts = line.split(':')[1].strip().split()
            w_pt, h_pt = float(parts[0]), float(parts[2])
            return w_pt * 0.352778, h_pt * 0.352778
    return None, None


def min_font_pt(pdf):
    import math

    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTChar, LTTextContainer

    sizes = []

    def char_size(ch):
        # LTChar.size reports the glyph height, which equals the font size for
        # upright text but reports the wrong (cross) dimension for rotated text
        # such as a 90deg-rotated y-axis label -- there the true font size shows
        # up as the glyph *width* instead. Detect rotation from the text matrix
        # (|b| > |a| => rotated towards vertical) and use width in that case.
        try:
            a, b = ch.matrix[0], ch.matrix[1]
            if abs(b) > abs(a):
                return ch.width
        except (AttributeError, TypeError, IndexError):
            pass
        return ch.size

    def walk(obj):
        if isinstance(obj, LTChar):
            if obj.get_text().strip():  # ignore spaces
                sizes.append(char_size(obj))
        if isinstance(obj, (LTTextContainer,)) or hasattr(obj, '__iter__'):
            try:
                for child in obj:
                    walk(child)
            except TypeError:
                pass

    for page in extract_pages(pdf):
        walk(page)
    return min(sizes) if sizes else None


def fonts_info(pdf):
    out = subprocess.run(['pdffonts', pdf], capture_output=True, text=True).stdout
    lines = out.splitlines()[2:]
    type3 = any(' Type 3' in ln for ln in lines)
    embedded = all(
        (ln.split()[-4] == 'yes') for ln in lines if len(ln.split()) >= 4
    ) if lines else False
    names = [ln.split()[0] for ln in lines if ln.strip()]
    return embedded, type3, names


def is_rgb(pdf):
    """No DeviceCMYK / Separation colour operators in the content stream."""
    with open(pdf, 'rb') as fh:
        raw = fh.read()
    return b'/DeviceCMYK' not in raw and b'/Separation' not in raw


def check(pdf):
    w, h = page_size_mm(pdf)
    width_ok = w is not None and any(
        abs(w - nw) <= WIDTH_TOL_MM or w <= nw + WIDTH_TOL_MM
        for nw in NATURE_WIDTHS_MM) and w <= 180 + WIDTH_TOL_MM
    height_ok = h is not None and h <= MAX_HEIGHT_MM
    mf = min_font_pt(pdf)
    font_ok = mf is not None and mf >= MIN_FONT_PT
    embedded, type3, _ = fonts_info(pdf)
    rgb = is_rgb(pdf)
    return dict(pdf=op.basename(pdf), w=w, h=h, width_ok=width_ok,
               height_ok=height_ok, min_font=mf, font_ok=font_ok,
               embedded=embedded, type3=type3, rgb=rgb)


def main(paths):
    if not paths:
        paths = sorted(glob.glob(op.join(style.figures_dir(), '*.pdf')))
    if not paths:
        print('No PDFs found.')
        return 1

    rows = [check(p) for p in paths]
    hdr = f'{"panel":42} {"w×h (mm)":15} {"minpt":6} {"emb":4} {"vec":4} {"rgb":4} {"ok":3}'
    print(hdr)
    print('-' * len(hdr))
    all_ok = True
    for r in rows:
        ok = (r['width_ok'] and r['height_ok'] and r['font_ok']
              and r['embedded'] and not r['type3'] and r['rgb'])
        all_ok &= ok
        wh = f"{r['w']:.0f}×{r['h']:.0f}" if r['w'] else '??'
        mf = f"{r['min_font']:.1f}" if r['min_font'] else '??'
        flag = '' if r['font_ok'] else '  <-- FONT < 6pt'
        if r['min_font'] and r['min_font'] < WANT_FONT_PT and r['font_ok']:
            flag = '  (<7pt, ok but tight)'
        print(f"{r['pdf']:42} {wh:15} {mf:6} "
              f"{'y' if r['embedded'] else 'N':4} "
              f"{'y' if not r['type3'] else 'N':4} "
              f"{'y' if r['rgb'] else 'N':4} "
              f"{'OK' if ok else 'FAIL':3}{flag}")
    print('-' * len(hdr))
    print('ALL PASS' if all_ok else 'SOME PANELS FAIL -- see above')
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
