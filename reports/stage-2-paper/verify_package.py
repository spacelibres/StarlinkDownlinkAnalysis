"""Check both editions, vector panels and optionally local evidence hashes."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
import re
from xml.etree import ElementTree

import pymupdf

from verify_vectors import verify


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NUMBERS = ('1,009', '41,169', '43,040', '2,184', '2,792', '1,145', '1.08')
FIGURES = set(range(1, 10))


def check_edition(suffix, caption):
    source = HERE / ('manuscript.zh.md' if suffix else 'manuscript.md')
    text = source.read_text(encoding='utf8')
    figures = re.findall(r'^### Figure (\d+)\.', text, re.MULTILINE)
    assert list(map(int, figures)) == sorted(FIGURES), source
    assert len(re.findall(r'^## [1-9]\.', text, re.MULTILINE)) == 9, source
    assert len(re.findall(r'^> \([1-9]\)', text, re.MULTILINE)) == 9, source
    assert 'vistone868@gmail.com' in text
    for destination in re.findall(r'\]\(([^)\s]+)\)', text):
        if not destination.startswith(('http://', 'https://', 'mailto:')):
            assert (HERE / destination.split('#')[0]).is_file(), destination
    path = HERE / f'starlink-ku-downlink-stage-ii-stone{suffix}.pdf'
    with pymupdf.open(path) as pdf:
        assert pdf.metadata['author'] == 'stone'
        assert len(pdf) > 1
        contents = []
        for page in pdf:
            assert len(page.get_images(full=True)) == 1, (path.name, page.number + 1)
            logo = page.get_image_rects(page.get_images()[0][0])
            assert len(logo) == 1 and logo[0].y1 < 46 and logo[0].x0 > page.rect.width / 2
            for x0, y0, x1, y1, *_ in page.get_text('blocks'):
                assert -0.1 <= x0 <= x1 <= page.rect.width + 0.1, (path.name, page.number + 1)
                assert -0.1 <= y0 <= y1 <= page.rect.height + 0.1, (path.name, page.number + 1)
            contents.append(page.get_text())
        extracted = ''.join(contents)
        assert all(value in extracted for value in NUMBERS), path
        assert '\ufffd' not in extracted and 'vistone868@gmail.com' in extracted, path
        assert all(re.search(caption.format(i), extracted) for i in FIGURES), path
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
            assert file_digest(stream, 'sha256').hexdigest() == digest, path
    print(f'Verified {len(expected)} full-archive source hashes.')


def main(check_local_sources=False):
    verify()
    en = check_edition('', r'Figure\s+{}\.' )
    zh = check_edition('-zh', r'图\s*{}\s*。')
    for number in FIGURES:
        svg, = sorted((HERE / 'figures').glob(f'{number:02d}-*.svg'))
        assert ElementTree.parse(svg).getroot().tag.endswith('}svg')
        with pymupdf.open(svg.with_suffix('.pdf')) as panel:
            assert len(panel) == 1 and panel[0].get_drawings()
            assert not panel[0].get_images()
    if check_local_sources:
        check_sources()
    print(f'Verified signed PDF layouts ({en} English pages, {zh} Chinese pages) and nine vector figure pairs.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-sources', action='store_true', help='Hash inputs in the full archive')
    main(parser.parse_args().local_sources)
