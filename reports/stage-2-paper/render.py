"""Render the editable stage II article with vector figures and a page mark."""

import argparse
from html import escape
from io import BytesIO
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import resvg_py
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, KeepTogether, PageTemplate, Paragraph, Preformatted,
    Spacer, Table, TableStyle,
)
from svglib.svglib import svg2rlg


HERE = Path(__file__).resolve().parent
VISUAL = HERE.parents[1] / 'references' / 'visual_identity' / 'assets'
PAGE_W, PAGE_H = A4
MARGIN = 51
CONTENT_W = PAGE_W - 2*MARGIN
PALETTE = json.loads((VISUAL / 'palette-tokens.json').read_bytes())[0]['colors']
INK = colors.HexColor(PALETTE['ink'])
MUTED = colors.HexColor(PALETTE['muted'])
LINE = colors.HexColor(PALETTE['line'])
SURFACE = colors.HexColor(PALETTE['surface'])
PRIMARY = colors.HexColor(PALETTE['primaryOnLight'])
PUBLIC_SOURCE = 'https://github.com/spacelibres/StarlinkDownlinkAnalysis/blob/main/reports/stage-2-paper/'


def register_fonts():
    candidates = (
        (Path('C:/Windows/Fonts'), ('arial.ttf', 'arialbd.ttf', 'ariali.ttf', 'arialbi.ttf', 'consola.ttf')),
        (Path('/usr/share/fonts/truetype/liberation2'),
         ('LiberationSans-Regular.ttf', 'LiberationSans-Bold.ttf', 'LiberationSans-Italic.ttf',
          'LiberationSans-BoldItalic.ttf', 'LiberationMono-Regular.ttf')),
        (Path('/usr/share/fonts/truetype/liberation'),
         ('LiberationSans-Regular.ttf', 'LiberationSans-Bold.ttf', 'LiberationSans-Italic.ttf',
          'LiberationSans-BoldItalic.ttf', 'LiberationMono-Regular.ttf')),
    )
    names = ('PaperSans', 'PaperBold', 'PaperItalic', 'PaperBoldItalic', 'PaperMono')
    for directory, filenames in candidates:
        if all((directory/name).is_file() for name in filenames):
            for alias, file in zip(names, filenames, strict=True):
                pdfmetrics.registerFont(TTFont(alias, str(directory/file)))
            pdfmetrics.registerFontFamily('PaperSans', normal='PaperSans', bold='PaperBold',
                                           italic='PaperItalic', boldItalic='PaperBoldItalic')
            return names
    return 'Helvetica', 'Helvetica-Bold', 'Helvetica-Oblique', 'Helvetica-BoldOblique', 'Courier'


def options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--language', choices=('en', 'zh'), default='en')
    parser.add_argument('--cjk-font', type=Path, default=Path('C:/Windows/Fonts/msyh.ttc'))
    parser.add_argument('--cjk-bold', type=Path, default=Path('C:/Windows/Fonts/msyhbd.ttc'))
    return parser.parse_args()


ARGS = options() if __name__ == '__main__' else argparse.Namespace(language='en')
CHINESE = ARGS.language == 'zh'
SANS, BOLD, ITALIC, _, MONO = register_fonts()
if CHINESE:
    if ARGS.cjk_font.is_file() and ARGS.cjk_bold.is_file():
        SANS, BOLD = 'PaperCJK', 'PaperCJKBold'
        pdfmetrics.registerFont(TTFont(SANS, str(ARGS.cjk_font), subfontIndex=0))
        pdfmetrics.registerFont(TTFont(BOLD, str(ARGS.cjk_bold), subfontIndex=0))
    else:
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        SANS = BOLD = 'STSong-Light'
        pdfmetrics.registerFont(UnicodeCIDFont(SANS))
    ITALIC = SANS
    pdfmetrics.registerFontFamily(SANS, normal=SANS, bold=BOLD, italic=SANS, boldItalic=BOLD)


def style(name, size, leading, font=SANS, **values):
    return ParagraphStyle(name, fontName=font, fontSize=size, leading=leading, textColor=INK,
                          wordWrap='CJK' if CHINESE else None, **values)


BODY = style('body', 10.3, 14.7, spaceAfter=6.5, allowWidows=0, allowOrphans=0)
SMALL = style('references', 8.8, 11.5, spaceAfter=4, leftIndent=16, firstLineIndent=-16)
TITLE = style('title', 24.0, 28.3, BOLD, spaceAfter=13)
BYLINE = style('byline', 10.7, 15.1, spaceAfter=7)
H2 = style('section', 15.3, 19.0, BOLD, spaceBefore=12, spaceAfter=6, keepWithNext=True)
H3 = style('section3', 11.6, 15.0, BOLD, spaceBefore=8, spaceAfter=4, keepWithNext=True)
CAPTION = style('caption', 8.9, 12.2, spaceAfter=6, spaceBefore=4)
CELL = style('cell', 8.95, 12.1)
CELL_HEAD = style('cell-head', 8.95, 12.1, BOLD)
CODE = style('algorithm', 8.5, 11.9, SANS if CHINESE else MONO,
             leftIndent=9, rightIndent=9, spaceBefore=5, spaceAfter=9)
EQUATION = style('equation', 10.3, 14.7, ITALIC, leftIndent=13, rightIndent=13, spaceAfter=8)


def inline(text):
    text = text.replace(' ', ' ').replace('‑', '-')
    html = escape(text, quote=False)
    def link(match):
        label, destination = match.groups()
        if not destination.startswith(('https://', 'http://', 'mailto:')):
            destination = PUBLIC_SOURCE + destination
        return f'<link href="{destination}" color="{PALETTE["primaryOnLight"]}"><u>{label}</u></link>'

    html = re.sub(r'\[([^\[\]]+)\]\(([^)\s]+)\)', link, html)
    html = re.sub(r'`([^`]+)`', rf'<font name="{MONO}">\1</font>', html)
    html = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', html)
    return re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<i>\1</i>', html)


def para(text, appearance=BODY):
    return Paragraph(inline(text), appearance)


def logo():
    source = ET.parse(VISUAL / 'horizontal-primary.svg').getroot()
    if source.get('data-color-mode') != 'monochrome':
        for element in source.iter():
            role = element.get('data-stroke-role', '')
            if role == 'planet':
                element.set('stroke', PALETTE['planetOnLight'])
            if element.get('fill', '').lower() == '#007bff':
                element.set('fill', PALETTE['primaryOnLight'])
            if element.get('fill', '').lower() == '#002044':
                element.set('fill', PALETTE['ink'])
    width, height = map(float, source.get('viewBox').split()[2:])
    pixels = resvg_py.svg_to_bytes(svg_string=ET.tostring(source, encoding='unicode'), width=760)
    return ImageReader(BytesIO(pixels)), width / height


def table_for(lines):
    parsed = [list(map(str.strip, r.strip().strip('|').split('|'))) for r in lines]
    cells = [row for row in parsed if not all(re.fullmatch(r':?-+:?', x) for x in row)]
    if not cells or len({len(r) for r in cells}) != 1:
        raise ValueError('A manuscript table has inconsistent columns')
    columns = len(cells[0])
    weights = [min(34, max(9, max(len(r[i]) for r in cells))) for i in range(columns)]
    widths = [CONTENT_W*w/sum(weights) for w in weights]
    rows = [[para(t, CELL_HEAD if i == 0 else CELL) for t in row] for i, row in enumerate(cells)]
    tb = Table(rows, colWidths=widths, repeatRows=1, hAlign='LEFT')
    tb.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('BACKGROUND', (0, 0), (-1, 0), SURFACE),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor(PALETTE['card'])]),
        ('LINEABOVE', (0, 0), (-1, 0), 1.05, PRIMARY),
        ('LINEBELOW', (0, 0), (-1, 0), .55, LINE),
        ('LINEBELOW', (0, -1), (-1, -1), .7, LINE),
    ]))
    return tb


def figure(number, caption):
    paths = sorted((HERE / 'figures').glob(f'{number:02d}-*.svg'))
    if len(paths) != 1:
        raise ValueError(f'Expected one vector file for figure {number}, found {len(paths)}')
    drawing = svg2rlg(str(paths[0]))
    if not drawing or drawing.width <= 0 or drawing.height <= 0:
        raise ValueError(f'Cannot read figure {paths[0].name}')
    scale = min(CONTENT_W / drawing.width, 300 / drawing.height)
    drawing.scale(scale, scale)
    drawing.width *= scale
    drawing.height *= scale
    drawing.hAlign = 'CENTER'
    return KeepTogether([drawing, para(caption, CAPTION), Spacer(1, 4)])


def blocks(text):
    """Keep fenced algorithms intact while reading the manuscript as paragraphs."""
    in_code = False
    block = []
    for line in text.splitlines():
        if line.startswith('```'):
            if in_code:
                yield 'code', '\n'.join(block)
                block = []
            elif block:
                yield 'text', '\n'.join(block)
                block = []
            in_code = not in_code
        elif not line.strip() and not in_code:
            if block:
                yield 'text', '\n'.join(block)
                block = []
        else:
            block.append(line)
    if in_code:
        raise ValueError('Unclosed fenced algorithm')
    if block:
        yield 'text', '\n'.join(block)


def manuscript_flow(source):
    story, figures = [], set()
    title = None
    references = False
    for kind, raw in blocks(source):
        if kind == 'code':
            algorithm = [Preformatted(raw, CODE, maxLineLength=100)]
            if story and getattr(story[-1], 'style', None) is H3:
                algorithm.insert(0, story.pop())
            story.append(KeepTogether(algorithm))
            continue
        match = re.fullmatch(r'###\s+(Figure|Fig\.)\s*(\d+)\.?\s+(.+)', raw, re.I)
        if match:
            number = int(match[2])
            if number in figures:
                raise ValueError('Duplicate figure marker')
            figures.add(number)
            label = f'图 {number}。' if CHINESE else f'Figure {number}.'
            story.append(figure(number, f'**{label}** {match[3]}'))
        elif raw.startswith('# '):
            title = raw[2:]
            story.append(para(title, TITLE))
        elif raw.startswith('## '):
            references = bool(re.search(r'\breferences\b|参考文献', raw, re.I))
            story.append(para(raw[3:], H2))
        elif raw.startswith('### '):
            story.append(para(raw[4:], H3))
        elif raw.startswith('|'):
            story.extend([table_for(raw.splitlines()), Spacer(1, 8)])
        elif raw.startswith('> '):
            story.append(para(' '.join(s.removeprefix('> ') for s in raw.splitlines()), EQUATION))
        elif not any(getattr(item, 'style', None) is H2 for item in story) and (
                 'vistone868@gmail.com' in raw or raw.strip() == 'stone'):
            for line in raw.splitlines():
                story.append(para(line, BYLINE))
        elif references:
            story.extend(para(line, SMALL) for line in raw.splitlines())
        elif all(re.match(r'^(?:[-*] |\d+\. )', s) for s in raw.splitlines()):
            story.extend(para(line, BODY) for line in raw.splitlines())
        else:
            story.append(para(raw.replace('\n', ' '), BODY))
    if title is None or figures != set(range(1, 10)):
        raise ValueError(f'Manuscript title or nine figure markers missing, found {sorted(figures)}')
    return title, story


def render():
    source = HERE / ('manuscript.zh.md' if CHINESE else 'manuscript.md')
    file_suffix = '-zh' if CHINESE else ''
    output = HERE / f'starlink-ku-downlink-stage-ii-stone{file_suffix}.pdf'
    title, story = manuscript_flow(source.read_text(encoding='utf8'))
    mark, mark_aspect = logo()
    doc = BaseDocTemplate(str(output), pagesize=A4, invariant=1, title=title, author='stone',
                          subject='Stage II research article on the Starlink Ku-band downlink')
    frame = Frame(MARGIN, 51, CONTENT_W, PAGE_H - 105, leftPadding=0, rightPadding=0,
                  topPadding=0, bottomPadding=0)

    def page(canvas, _):
        canvas.saveState()
        canvas.setStrokeColor(LINE)
        canvas.setLineWidth(.55)
        canvas.line(MARGIN, PAGE_H - 45, PAGE_W - MARGIN, PAGE_H - 45)
        canvas.line(MARGIN, 40, PAGE_W - MARGIN, 40)
        canvas.setFillColor(MUTED)
        canvas.setFont(SANS, 8)
        canvas.drawString(MARGIN, PAGE_H - 30, 'STARLINK KU 波段下行' if CHINESE else 'STARLINK KU-BAND DOWNLINK')
        canvas.drawString(MARGIN, 26, '第二阶段研究论文' if CHINESE else 'STAGE II RESEARCH ARTICLE')
        canvas.drawRightString(PAGE_W - MARGIN, 26, str(doc.page))
        height = 94 / mark_aspect
        canvas.drawImage(mark, PAGE_W - MARGIN - 94, PAGE_H - 38, width=94, height=height, mask='auto')
        canvas.restoreState()

    doc.addPageTemplates([PageTemplate(id='body', frames=[frame], onPage=page)])
    doc.build(story)
    print(output)


if __name__ == '__main__':
    render()
