"""Check the three manuscripts, print figures, equations and evidence hashes."""

import argparse
from hashlib import file_digest, sha256
import json
from pathlib import Path
import re
import subprocess
from xml.etree import ElementTree

import pymupdf

from verify_vectors import verify


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NUMBERS = ('1,009', '41,169', '43,040', '2,184', '2,792', '1,145', '1.08')
FIGURES = set(range(1, 10))
INTERIM = HERE.parent / 'interim-paper'
SIGNED_INTERIM_SHA256 = '9482f58fb20a6a8a97f9895f755921d2935567efbe54f56a00bd0968710d1568'
PRIVATE_REFERENCE = re.compile(r'(?<![A-Za-z0-9])E[0-9]{3}(?![A-Za-z0-9])|'
                               r'\b(?:runs|experiments|artifacts)[/\\]|'
                               r'\b[A-Za-z]:[\\/]|README\.md|\.json\b', re.I)


def check_edition(paper, chinese, expected_figures, expected_equations, numbers):
    source = paper / ('manuscript.zh.md' if chinese else 'manuscript.md')
    text = source.read_text(encoding='utf8')
    figures = re.findall(r'^### Figure (\d+)\.', text, re.MULTILINE)
    assert list(map(int, figures)) == list(range(1, expected_figures + 1)), source
    equations = re.findall(r'^> \((\d+)\) \$[^$\n]+\$$', text, re.MULTILINE)
    assert list(map(int, equations)) == list(range(1, expected_equations + 1)), source
    assert 'vistone868@gmail.com' in text
    assert not PRIVATE_REFERENCE.search(text), PRIVATE_REFERENCE.search(text)
    if chinese:
        assert '码字' not in text, source
    for destination in re.findall(r'\]\(([^)\s]+)\)', text):
        assert destination.startswith(('http://', 'https://', 'mailto:')), destination
    study = 'interim' if paper == INTERIM else 'stage-ii'
    suffix = '-zh' if chinese else ''
    path = paper / f'starlink-ku-downlink-{study}-stone{suffix}.pdf'
    equation_numbers = []
    with pymupdf.open(path) as pdf:
        assert pdf.metadata['author'] == 'stone'
        assert len(pdf) > 1
        contents = []
        for page in pdf:
            logos = [rect for img in page.get_images(full=True) for rect in page.get_image_rects(img[0])
                     if rect.y1 < 46 and rect.x0 > page.rect.width / 2]
            assert len(logos) == 1, (path.name, page.number + 1)
            for x0, y0, x1, y1, text, *_ in page.get_text('blocks'):
                assert -0.1 <= x0 <= x1 <= page.rect.width + 0.1, (path.name, page.number + 1)
                assert -0.1 <= y0 <= y1 <= page.rect.height + 0.1, (path.name, page.number + 1)
                match = re.fullmatch(r'\((\d+)\)', text.strip())
                if match and x1 > page.rect.width - 60:
                    box = pymupdf.Rect(69, y0 - 30, x0 - 18, y1 + 30)
                    assert any(box.contains(d['rect']) for d in page.get_drawings()), (path, match[1])
                    equation_numbers.append(int(match[1]))
            for link in page.get_links():
                assert link.get('uri', '').startswith(('http://', 'https://', 'mailto:')), link
            contents.append(page.get_text())
        extracted = ''.join(contents)
        assert equation_numbers == list(range(1, expected_equations + 1)), (path, equation_numbers)
        assert not PRIVATE_REFERENCE.search(extracted), PRIVATE_REFERENCE.search(extracted)
        assert all(value in extracted for value in numbers), path
        assert '\ufffd' not in extracted and 'vistone868@gmail.com' in extracted, path
        caption = r'图\s*{}\s*。' if chinese else r'Figure\s+{}\.'
        assert all(re.search(caption.format(i), extracted) for i in range(1, expected_figures + 1)), path
        return len(pdf)


def check_sources():
    figures = json.loads((HERE / 'data/evidence_summary.json').read_bytes())['sources']
    vectors = json.loads((HERE / 'data/verification_vectors.json').read_bytes())['sources_sha256']
    expected = {v['path']: v['sha256'] for v in figures.values()}
    for path, digest in vectors.items():
        assert expected.get(path, digest) == digest, path
        expected[path] = digest
    for path, digest in expected.items():
        with (ROOT / path).open('rb') as stream:
            actual = file_digest(stream, 'sha256').hexdigest()
        if actual != digest:
            historic = subprocess.check_output(['git', 'show', f'v0.0.255:{path}'], cwd=ROOT)
            assert sha256(historic).hexdigest() == digest, path
    with (INTERIM / 'starlink-ku-downlink-interim-stone.pdf').open('rb') as stream:
        assert file_digest(stream, 'sha256').hexdigest() == SIGNED_INTERIM_SHA256
    print(f'Verified {len(expected)} full-archive source hashes.')


def main(check_local_sources=False):
    verify()
    en = check_edition(HERE, False, 9, 9, NUMBERS)
    zh = check_edition(HERE, True, 9, 9, NUMBERS)
    interim = check_edition(INTERIM, True, 8, 3, ('359.58', '0.314', '46.89', '1,037'))
    for number in FIGURES:
        svg, = sorted((HERE / 'figures').glob(f'{number:02d}-*.svg'))
        assert ElementTree.parse(svg).getroot().tag.endswith('}svg')
        with pymupdf.open(svg.with_suffix('.pdf')) as panel:
            assert len(panel) == 1 and panel[0].get_drawings()
            assert not panel[0].get_images()
            assert not PRIVATE_REFERENCE.search(panel[0].get_text()), svg
    if check_local_sources:
        check_sources()
    print(f'Verified layouts, captions, links and vector equations in {en} English stage II pages, '
          f'{zh} Chinese stage II pages and {interim} Chinese interim pages.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-sources', action='store_true', help='Hash inputs in the full archive')
    main(parser.parse_args().local_sources)
