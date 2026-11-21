"""Render a deterministic two-column research manuscript from its editable source."""

from html import escape
from math import log10
from pathlib import Path
import re

from reportlab.graphics.shapes import Circle, Drawing, Line, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import BaseDocTemplate, Frame, FrameBreak, KeepTogether, PageTemplate, Paragraph, Spacer, Table, TableStyle


HERE = Path(__file__).resolve().parent
PAGE_WIDTH, PAGE_HEIGHT = letter
MARGIN = 44
GUTTER = 16
WIDTH = PAGE_WIDTH - 2 * MARGIN
COLUMN = (WIDTH - GUTTER) / 2
BLACK = colors.HexColor('#161616')
GRAY = colors.HexColor('#4b4b4b')


def style(name, font='Times-Roman', size=9.35, leading=11.4, **options):
    return ParagraphStyle(name, fontName=font, fontSize=size, leading=leading, textColor=BLACK, **options)


BODY = style('body', alignment=TA_JUSTIFY, spaceAfter=5, allowWidows=0, allowOrphans=0)
H2 = style('heading', 'Times-Bold', 10.4, 12.4, spaceBefore=8, spaceAfter=4, keepWithNext=True)
H3 = style('subheading', 'Times-Italic', 9.7, 11.7, spaceBefore=5, spaceAfter=3, keepWithNext=True)
TITLE = style('title', 'Times-Roman', 23, 25, alignment=TA_CENTER, spaceAfter=8)
EDITION = style('edition', 'Helvetica', 8.1, 10, alignment=TA_CENTER, spaceAfter=12)
ABSTRACT = style('abstract', 'Times-Roman', 10, 12.4, alignment=TA_JUSTIFY, spaceAfter=6)
CELL = style('table-body', size=8.0, leading=9.5)
CELL_HEAD = style('table-head', 'Times-Bold', 8.0, 9.5)
CAPTION = style('caption', 'Times-Roman', 8.1, 10, spaceAfter=4)
REFERENCE = style('reference', size=8.6, leading=10.4, alignment=TA_LEFT,
                  leftIndent=17, firstLineIndent=-17, spaceAfter=4, allowWidows=0, allowOrphans=0)
EQUATION = style('equation', 'Times-Italic', 9, 12, alignment=TA_CENTER, spaceBefore=2, spaceAfter=6)


def paragraph(line, paragraph_style=BODY):
    line = line.translate(str.maketrans({'–': '-', '−': '-', '—': '-', '\u2003': ' '}))
    markup = escape(line, quote=False)
    markup = re.sub(r'\[([^\[\]]+)\]\((https?://[^)]+)\)', r'<link href="\2"><u>\1</u></link>', markup)
    markup = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', markup)
    markup = re.sub(r'`([^`]+)`', r'<font name="Courier">\1</font>', markup)
    for key, subscript in [('t_GPS', 'GPS'), ('H_u', 'u'), ('H_p', 'p'), ('H_E', 'E'), ('H_O', 'O'),
                           ('c_E', 'E'), ('c_O', 'O')]:
        markup = markup.replace(key, key[0] + f'<sub>{subscript}</sub>')
    return Paragraph(markup, paragraph_style)


def table_rows(lines, caption, heading):
    data = []
    for line in lines:
        cells = [part.strip() for part in line.strip('|').split('|')]
        if all(re.fullmatch(r'[-:]+', part) for part in cells):
            continue
        data.append([paragraph(part, CELL if data else CELL_HEAD) for part in cells])
    widths = (0.24, 0.33, 0.43) if not caption.startswith('TABLE IV') else (0.27, 0.36, 0.37)
    table = Table(data, colWidths=[COLUMN * f for f in widths], repeatRows=1, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4), ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4), ('LINEABOVE', (0, 0), (-1, 0), 0.7, BLACK),
        ('LINEBELOW', (0, 0), (-1, 0), 0.45, BLACK), ('LINEBELOW', (0, -1), (-1, -1), 0.6, BLACK),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f3f3f3')]),
    ]))
    return KeepTogether([*heading, paragraph(caption, CAPTION), table, Spacer(1, 7)])


def toa_figure(caption):
    drawing = Drawing(COLUMN, 111)
    left, right, bottom, top = 72, COLUMN - 26, 27, 105

    def x(value):
        return left + (log10(value)-1)*(right-left)/4

    for exponent in range(1, 6):
        coord = x(10 ** exponent)
        drawing.add(Line(coord, bottom, coord, top, strokeColor=colors.HexColor('#cccccc'), strokeWidth=.5))
        drawing.add(String(coord, 17, f'{10**exponent:,}', fontName='Times-Roman', fontSize=7, textAnchor='middle'))
    for y, name, value in ((93, 'Decoded state', 46.89), (68, 'Site +1° longitude', 5581.11),
                           (43, 'Range omitted', 33298.28)):
        drawing.add(String(0, y-2.5, name, fontName='Times-Roman', fontSize=8))
        drawing.add(Line(left, y, x(value), y, strokeColor=BLACK, strokeWidth=.65))
        drawing.add(Circle(x(value), y, 2.6, fillColor=BLACK, strokeColor=None))
        drawing.add(String(x(value), y+5, f'{value:,.2f}', fontName='Times-Bold', fontSize=7, textAnchor='middle'))
    drawing.add(String((left+right)/2, 5, 'TOA residual RMS (ns)', fontName='Times-Roman', fontSize=8, textAnchor='middle'))
    return KeepTogether([drawing, paragraph(caption, CAPTION), Spacer(1, 3)])


def layout(text, destination):
    blocks = [block.strip() for block in text.split('\n\n') if block.strip()]
    title, edition, abstract, terms, *body = blocks
    front = [paragraph(title.removeprefix('# '), TITLE), paragraph(edition, EDITION),
             paragraph(abstract, ABSTRACT), paragraph(terms, ABSTRACT)]
    front_height = sum(item.wrap(WIDTH, PAGE_HEIGHT)[1] + item.getSpaceAfter() for item in front)
    flow = []
    caption = None
    in_references = False
    for block in body:
        if block.startswith('### TABLE'):
            caption = block[4:]
        elif block.startswith('|'):
            if caption is None:
                raise ValueError('Table missing caption')
            heading = [flow.pop()] if flow and getattr(flow[-1], 'style', None) is H2 else []
            flow.append(table_rows(block.splitlines(), caption, heading))
            caption = None
        elif block.startswith('### FIG. 1.'):
            flow.append(toa_figure(block[4:]))
        elif block.startswith('## '):
            flow.append(paragraph(block[3:], H2))
            in_references = block == '## References'
        elif block.startswith('### '):
            flow.append(paragraph(block[4:], H3))
        elif block.startswith('> '):
            flow.append(paragraph(block[2:], EQUATION))
        else:
            if in_references and block.startswith('[7] '):
                flow.append(FrameBreak())
            flow.append(paragraph(block, REFERENCE if in_references else BODY))

    doc = BaseDocTemplate(str(destination), pagesize=letter, invariant=1,
                          title=title.removeprefix('# '), subject='Interim research manuscript',
                          creator='StarlinkBeacon manuscript renderer')

    def frames(top):
        return [Frame(x, 43, COLUMN, top-43, leftPadding=0, rightPadding=0,
                      topPadding=0, bottomPadding=0) for x in (MARGIN, MARGIN+COLUMN+GUTTER)]

    def page(canvas, document):
        canvas.saveState()
        canvas.setFillColor(GRAY)
        canvas.setFont('Helvetica', 7.5)
        canvas.drawString(MARGIN, 27, 'INTERIM RESEARCH MANUSCRIPT')
        canvas.drawRightString(PAGE_WIDTH-MARGIN, 27, str(document.page))
        if document.page == 1:
            y = PAGE_HEIGHT - 40
            for item in front:
                height = item.wrap(WIDTH, PAGE_HEIGHT)[1]
                y -= height
                item.drawOn(canvas, MARGIN, y)
                y -= item.getSpaceAfter()
        else:
            canvas.drawString(MARGIN, PAGE_HEIGHT-26, 'STARLINK KU-BAND DOWNLINK CONTROL RECORDS')
        canvas.restoreState()

    doc.addPageTemplates([
        PageTemplate(id='first', frames=frames(PAGE_HEIGHT-52-front_height), onPage=page, autoNextPageTemplate='body'),
        PageTemplate(id='body', frames=frames(PAGE_HEIGHT-42), onPage=page),
    ])
    doc.build(flow)


if __name__ == '__main__':
    layout((HERE/'manuscript.md').read_text(encoding='utf-8'), HERE/'starlink-ku-downlink-interim.pdf')
