"""Build the stage II vector figures from the compact evidence export."""

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.ticker import FuncFormatter, PercentFormatter


HERE = Path(__file__).resolve().parent
EVIDENCE = json.loads((HERE / 'data' / 'evidence_summary.json').read_text(encoding='utf-8'))
D = EVIDENCE['data']
P = EVIDENCE['palette']
INK, BLUE, GRAY = P['ink'], P['primaryOnLight'], P['muted']
LINE, SURFACE, CARD = P['line'], P['surface'], P['card']
GREEN, GOLD, RED = P['success'], P['warning'], P['danger']
MILLIMETERS_PER_INCH = 25.4
WIDTH = 174 / MILLIMETERS_PER_INCH

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': 8.5, 'text.color': INK, 'axes.labelcolor': INK,
    'xtick.color': GRAY, 'ytick.color': INK, 'axes.edgecolor': LINE,
    'axes.linewidth': 0.6, 'grid.color': LINE, 'grid.linewidth': 0.45,
    'pdf.fonttype': 42, 'svg.hashsalt': 'starlink-stage-2', 'savefig.facecolor': 'white',
})


def figure(height):
    return plt.figure(figsize=(WIDTH, height / MILLIMETERS_PER_INCH))


def panel(fig, rect, label, title, grid=None):
    ax = fig.add_axes(rect)
    ax.set_title(f'{label}   {title}', loc='left', fontsize=10, fontweight='bold', pad=8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(length=0)
    if grid:
        ax.set_axisbelow(True)
        ax.grid(axis=grid)
    return ax


def note(fig, message):
    fig.text(.02, .03, message, color=GRAY, fontsize=8.2, va='bottom', linespacing=1.4)


def n(value):
    return f'{value:,}'


def box(ax, at, size, title, detail, color=BLUE, fill=CARD):
    x, y = at
    w, h = size
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.008,rounding_size=.016',
                                facecolor=fill, edgecolor=color, linewidth=.7))
    ax.text(x+w/2, y+.69*h, title, ha='center', va='center', fontsize=9.1,
            fontweight='bold', color=color)
    ax.text(x+w/2, y+.30*h, detail, ha='center', va='center', fontsize=8.1, linespacing=1.28)


def diagram(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    return ax


def arrow(ax, start, end, bend=0):
    ax.annotate('', xy=end, xytext=start,
                arrowprops={'arrowstyle': '-|>', 'color': GRAY, 'lw': .85,
                            'shrinkA': 0, 'shrinkB': 0, 'connectionstyle': f'arc3,rad={bend}'})


def capture_evidence():
    fig = figure(71)
    counts = D['captures']
    ax = diagram(fig, [.025, .17, .41, .71])
    ax.text(0, 1.05, 'A   Two-header capture inventories', fontsize=10, fontweight='bold')
    labels = ['Qin public archive', 'Independent legacy 480 MS/s', 'UT 250 MS/s main time grid']
    for y, label, key in zip([.83, .53, .23], labels, counts, strict=True):
        ax.hlines(y-.13, 0, .98, color=LINE, linewidth=.6)
        ax.text(.01, y, label, va='center')
        ax.text(.97, y, n(counts[key]), va='center', ha='right', fontsize=15, color=BLUE,
                fontweight='bold')

    ax = diagram(fig, [.49, .17, .49, .71])
    ax.text(0, 1.05, 'B   Evidence by decoding layer', fontsize=10, fontweight='bold')
    ax.text(.015, .86, 'Waveform and H1 / H2 geometry', fontweight='bold')
    ax.text(.015, .66, 'Channel models across 25 observed MCS labels', fontweight='bold')
    code_labels = ['Full\nlength', 'Shortened\ncore', 'Shared\npartial', 'Other\npartial', 'Unresolved']
    for i, (label, value) in enumerate(zip(code_labels, D['codes'].values(), strict=True)):
        x = .02 + .196 * i
        ax.text(x, .49, str(value), color=BLUE if i < 4 else GOLD, fontsize=12, fontweight='bold')
        ax.text(x, .40, label, fontsize=7.6, va='top', linespacing=1.04)
    ax.text(.015, .15, 'Control fields and record boundaries', fontweight='bold')
    ax.text(.015, 0, f"Application-layer messages verified   {D['application_messages']}", color=GOLD)
    note(fig, 'Qin and the UT 250 MS/s inventory overlap at source. Capture counts are reported separately.')
    return fig


def full_word_coverage():
    data = D['full']
    stages = data['stages']
    fig = figure(76)
    settings = [('words', 'Full codewords', 'word_denominator', (39000, 44000)),
                ('closed', 'Closed allocations', 'allocation_denominator', (2050, 2740))]
    for i, (key, title, total_key, scale) in enumerate(settings):
        total = data[total_key]
        values = [x[key] for x in stages]
        ax = panel(fig, [.105 + .49*i, .25, .38, .59], 'AB'[i], title, 'y')
        ax.plot(range(len(stages)), values, marker='o', color=BLUE, markersize=3.4, linewidth=1.3)
        ax.axhline(total, color=GRAY, linewidth=.7, linestyle=(0, (4, 2)))
        ax.text(5.05, total, f'{n(total)} total', ha='right', va='bottom', color=GRAY, fontsize=7.7)
        ax.set(xlim=(-.35, 5.4), ylim=scale, xticks=range(len(stages)))
        ax.set_xticklabels([x['experiment'] for x in stages], fontsize=7.4)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda tick, _: n(int(tick))))
        for idx in [0, len(stages)-1]:
            ax.annotate(n(values[idx]), (idx, values[idx]), xytext=(0, 7), textcoords='offset points',
                        fontsize=8, ha='left' if idx == 0 else 'right', color=BLUE, fontweight='bold')
    note(fig, 'Qin MCS 66, 70, 85 and 88. Dashed lines show the fixed inventories after all frame ends closed.')
    return fig


def partial_core_coverage():
    data = D['partial']
    rows = data['formats']
    fig = figure(79)
    ax = panel(fig, [.15, .27, .53, .52], 'A', 'Accepted systematic / high-parity cores', 'x')
    for idx, r in enumerate(rows):
        before, added, total = (r[k] for k in ('before', 'added', 'total'))
        ax.barh(idx, 1, height=.54, color=SURFACE)
        ax.barh(idx, before/total, height=.54, color=BLUE)
        ax.barh(idx, added/total, left=before/total, height=.54, color=GREEN)
        ax.text(1.025, idx, f'{n(before+added)} / {n(total)}   {(before+added)/total:.1%}',
                va='center', fontsize=8.3, clip_on=False)
        ax.text(1.48, idx, f'+{added}', va='center', ha='right', color=GREEN, fontsize=8.3,
                fontweight='bold', clip_on=False)
    ax.set(xlim=(0, 1), yticks=range(len(rows)))
    ax.set_yticklabels([f"MCS {r['mcs']}" for r in rows])
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.spines['left'].set_visible(False)
    fig.legend(handles=[Patch(color=BLUE, label='E230 accepted'), Patch(color=GREEN, label='E247 additions')],
               loc='upper left', bbox_to_anchor=(.147, .965), frameon=False, ncol=2,
               borderaxespad=0, fontsize=8)
    total = sum(r['total'] for r in rows)
    accepted = sum(r['before']+r['added'] for r in rows)
    closed = data['closed_before'] + data['newly_closed']
    note(fig, f"{n(accepted)} / {n(total)} cores accepted. Systematic bytes closed in "
              f"{closed} / {data['allocation_denominator']} allocations.")
    return fig


def estimation_and_acceptance():
    fig = figure(88)
    ax = diagram(fig, [.025, .15, .95, .81])
    box(ax, (.015, .73), (.29, .24), 'Received Code 5', 'Observed symbols from the\nfirst two words in the frame')
    box(ax, (.355, .73), (.29, .24), 'Per-carrier estimates', 'Complex gain in both arms\nFlat or fitted residual variance')
    box(ax, (.695, .73), (.29, .24), 'Target likelihoods', 'Permanently erase low-address\nparity and boundary symbols', GOLD)
    arrow(ax, (.315, .85), (.345, .85))
    arrow(ax, (.655, .85), (.685, .85))

    branches = [
        (.015, 'Base', 'Permanent mask only'),
        (.355, 'Mask 1', 'Extra whole symbols'),
        (.695, 'Mask 2', 'Different whole symbols'),
    ]
    for x, title, detail in branches:
        box(ax, (x, .35), (.29, .17), title, detail, fill=SURFACE)
    ax.plot([.16, .84], [.626, .626], color=GRAY, linewidth=.85)
    ax.plot([.84, .84], [.72, .626], color=GRAY, linewidth=.85)
    for x, _, _ in branches:
        arrow(ax, (x+.145, .626), (x+.145, .539))
        ax.plot([x+.145, x+.145], [.34, .28], color=GRAY, linewidth=.85)
    ax.plot([.16, .84], [.28, .28], color=GRAY, linewidth=.85)
    arrow(ax, (.5, .28), (.5, .22))
    box(ax, (.11, .015), (.78, .19), 'Accept the systematic and high-address parity bits',
        'Three-way agreement, independent encoding, and held symbols that beat shifted controls', GREEN)
    note(fig, 'Acceptance applies to each estimation arm. Low-address observations remain archived under their own masks.')
    return fig


def receiver_and_crc():
    data = D['mcs57_receiver']
    fig = figure(88)
    positions = [[.13, .26, .23, .52], [.45, .26, .18, .52], [.746, .26, .231, .52]]
    pn1, pn2 = data['pn222'], data['pn245']
    groups = [('E222 PN15 errors', [pn1['slot_soft'], pn1['mixture'], pn1['archive_hard']],
               ['Slot soft', 'Mixture', 'Archive hard'], [GRAY, BLUE, P['deep']]),
              ('E245 PN15 errors', [pn2['pn_baseline_errors'], pn2['pn_errors']],
               ['Baseline', 'Code 5 joint'], [GRAY, BLUE])]
    for i, (title, values, labels, palette) in enumerate(groups):
        ax = panel(fig, positions[i], 'AB'[i], title, 'x')
        for j, value in enumerate(values):
            ax.barh(j, value, height=.47, color=palette[j])
            ax.text(value*.97, j, n(value), ha='right', va='center', color='white', fontsize=8)
        ax.set(xticks=[0, 200, 400] if i==0 else [0, 250, 500, 750], yticks=range(len(values)),
               xlim=(0,400 if i==0 else 750), ylim=(-.52, 2.52))
        ax.set_yticklabels(labels, fontsize=7.7)
        ax.invert_yaxis()
        ax.spines['left'].set_visible(False)
        total = pn1['bits'] if i==0 else pn2['pn_common_bits']
        ax.set_xlabel(f'{n(total)} common bits', fontsize=8.1)
    ax = panel(fig, positions[2], 'C', 'CRC24 directory sites', 'y')
    cumulative = 0
    for idx, value in enumerate(data['crc24'].values()):
        ax.bar(idx, cumulative, width=.58, color=SURFACE, edgecolor=LINE, linewidth=.4)
        ax.bar(idx, value, bottom=cumulative, width=.58, color=BLUE)
        cumulative += value
        ax.text(idx, cumulative+1, str(cumulative), ha='center', color=BLUE, fontsize=8.1)
    ax.set(ylim=(0, 51), xticks=range(3), yticks=[0, 20, 40])
    ax.set_xticklabels(['E178', 'E234', 'E245'])
    note(fig, f"E222 also favored 16QAM in {n(data['qam_support'])} / {n(data['scorable_fragments'])} scored slot fragments. "
              'CRC24 sites are cumulative and deduplicated.')
    return fig


def repeated_observations():
    data = D['mcs57_repeat']
    fig = figure(83)
    ax = diagram(fig, [.025, .20, .95, .68])
    ax.text(0, 1.04, 'A   Whole-symbol holdout', fontsize=10, fontweight='bold')
    ax.text(.57, 1.04, 'B   Prior six-view erasures', fontsize=10, fontweight='bold')
    ax.text(.19, .79, 'Aligned', color=BLUE, fontsize=8, fontweight='bold')
    ax.text(.37, .79, 'Permuted', color=GRAY, fontsize=8, fontweight='bold')
    ax.text(.69, .79, 'Target supports', color=GREEN, fontsize=8, fontweight='bold')
    ax.text(.86, .79, 'Target opposes', color=RED, fontsize=8, fontweight='bold')
    for idx, (label, r) in enumerate(zip(['One donor','Two donors'], data['routes'], strict=True)):
        y = .62-.34*idx
        ax.hlines(y-.21, 0, 1, color=LINE, linewidth=.6)
        ax.text(0, y, label, va='center', fontweight='bold')
        ax.text(0, y-.12, f"{r['targets']} targets", va='center', color=GRAY, fontsize=8)
        compared = r['compared']
        for x, key, color in [(.19,'nominal_disagreements',BLUE),(.37,'permuted_disagreements',GRAY)]:
            count = r[key]
            ax.text(x, y, f'{count/compared:.3%}', va='center', color=color, fontweight='bold')
            ax.text(x, y-.12, n(count), va='center', color=GRAY, fontsize=8)
        ax.text(.57, y, n(r['donor_candidate_bits']), va='center', fontweight='bold')
        ax.text(.57, y-.12, 'candidates', va='center', color=GRAY, fontsize=8)
        ax.text(.69, y, n(r['native_corroborated_bits']), va='center', color=GREEN, fontweight='bold')
        ax.text(.86, y, n(r['native_opposed_bits']), va='center', color=RED, fontweight='bold')
    note(fig, f"Compared bits, one donor {n(data['routes'][0]['compared'])}, two donors {n(data['routes'][1]['compared'])}. "
              f"Donors share the target archive group. The best target retains {data['minimum_residual_erasures']} erasures.")
    return fig


def aligned_projections():
    data = D['aligned']
    fig = figure(84)
    ax = diagram(fig, [.025, .19, .95, .70])
    ax.text(0, 1.04, 'A   Common rightmost 512-bit low block', fontsize=10, fontweight='bold')
    ax.text(.67, 1.04, 'B   Verified high controls', fontsize=10, fontweight='bold')
    for x, title in [(0, 'Fold'), (.12, 'Rank / features'), (.40, 'Contradictions'),
                     (.67, 'Models passed'), (.84, 'Held errors / rows')]:
        ax.text(x, .90, title, fontsize=8, color=GRAY)
    for i, (low, high) in enumerate(zip(data['low'], data['high'], strict=True)):
        y = .80-.108*i
        ax.hlines(y-.05, 0, 1, color=LINE, linewidth=.45)
        for x, label, color in [
            (0, str(low['fold']), INK), (.12, f"{n(low['rank'])} / {n(low['features'])}", INK),
            (.40, n(low['contradictions']) if low['calibration_rows'] else '—', RED if low['contradictions'] else GRAY),
            (.67, f"{high['tested']} / {high['tested']}", GREEN),
            (.84, f"{n(high['holdout_errors'])} / {n(high['holdout_rows'])}", GREEN),
        ]:
            ax.text(x, y, label, color=color, va='center', fontsize=8.5)
        ax.hlines(y, .255, .358, color=LINE, linewidth=2.1)
        ax.hlines(y, .255, .255+.103*low['rank']/low['features'], color=BLUE, linewidth=2.1)
    scope = data['scope_counts']
    note(fig, f"Across 98 low models, {scope['low_no_observation']} had no complete projection, "
              f"{scope['low_rank_deficient']} were rank deficient, and {scope['low_inconsistent']} were inconsistent. "
              'Dash indicates no rows.')
    return fig


def headers_and_narrowband():
    data = D['headers']
    fig = figure(95)
    ax = panel(fig, [.09,.57,.33,.28], 'A', 'UT 250 MS/s headers', 'x')
    for y, key in enumerate(['h1_accepted','h2_accepted']):
        ax.barh(y,data['ut_events'],height=.48,color=SURFACE)
        ax.barh(y,data[key],height=.48,color=BLUE)
        ax.text(data[key]-1,y,f"{data[key]} / {data['ut_events']}",ha='right',va='center',color='white')
    ax.set(yticks=[0,1],xticks=[0,10,20,30,40,50],xlim=(0,51))
    ax.set_yticklabels(['H1','H2'])
    ax.invert_yaxis()
    ax.spines['left'].set_visible(False)

    ax = diagram(fig, [.49,.51,.49,.34])
    ax.text(0, 1.09, 'B   Weak H2 allocations, CRC unresolved', fontsize=10, fontweight='bold')
    for y, row in zip([.86,.59,.32],data['weak_h2'],strict=True):
        label = '   +   '.join(f'MCS {mcs} × {count}' for mcs, count in row['allocations'])
        ax.text(0,y,f"{row['event_ms']:.6f} ms", color=GOLD)
        ax.text(.37,y,label)
        tail_start, tail_stop = row['tail_interval_resources']
        ax.text(.37, y-.115, f'Tail resources {n(tail_start)}–{n(tail_stop)}', fontsize=7.5, color=GRAY)
    crc = data['weak_crc']
    ax.text(0,.015,f"{crc['hex']} at {crc['event_ms']:.6f} ms   {crc['selected']} / {crc['comparisons']} pooled  ·  "
            f"{crc['control_matches']} / {crc['control_comparisons']} shifted or wrong-PN", fontsize=8.1,color=GRAY)

    leo = data['leo']
    ax = diagram(fig,[.025,.15,.95,.28])
    ax.text(0,1.02,'C   Qin H2 descriptor rank through paired LEO 2.5 MS/s passbands',
            fontsize=10,fontweight='bold')
    summary = [(f'{leo["at_least_one_visible_symbol"]} / {n(leo["headers"])}', 'with ≥1 visible resource'),
               *[(str(count),f'rank {rank}') for rank,count in leo['descriptor_rank_histogram'].items()],
               (str(leo['unique_descriptors']), 'unique descriptors')]
    for x, (value, label) in zip([0,.27,.42,.54,.66,.80], summary, strict=True):
        ax.text(x,.55,value,color=BLUE,fontsize=12,fontweight='bold')
        ax.text(x,.28,label,fontsize=8.2)
    note(fig, 'Weak H2 rows remain candidates. LEO ranks describe projected reference observability.')
    return fig


def vhf_doppler():
    data = D['vhf']
    fig = figure(89)
    models = data['models']
    ax = panel(fig, [.244,.24,.35,.54], 'A', 'Differential Doppler', 'x')
    keys = ['measurement', 'gps_j2', 'gps_spherical', 'gps_inertial_velocity_components', 'unix_j2']
    labels = ['Received preambles', 'GPS epoch + J2', 'GPS epoch + spherical', 'Inertial velocity components', 'Unix epoch + J2']
    values = [data['measurement_hz']] + [models[k]['predicted_differential_hz'] for k in keys[1:]]
    ax.set(xlim=(-3650,-2770),ylim=(4.6,-.6),yticks=range(5),xticks=[-3600,-3400,-3200,-3000,-2800])
    ax.set_yticklabels(labels,fontsize=8)
    ax.tick_params(axis='x', labelsize=7.6)
    ax.set_xlabel('Frequency difference (Hz)',fontsize=8)
    for i, (hz, color) in enumerate(zip(values,[BLUE,GREEN,GREEN,GOLD,RED],strict=True)):
        ax.plot(hz,i,'o',color=color,markersize=4.3)
        ax.annotate(f'{hz:,.2f}'.replace('-', '−'), (hz,i),xytext=((-5 if i==4 else 5),0),
                    textcoords='offset points',ha='right' if i==4 else 'left',va='center',color=color,
                    fontsize=7.8)

    ax = panel(fig, [.67,.24,.30,.54], 'B', 'GPS + J2 residual', 'x')
    residual = models['gps_j2']['observed_minus_predicted_hz']
    alternate = data['alternate_gps_j2_residual_hz']
    lo, hi = min(data['split_preamble_residuals_hz']), max(data['split_preamble_residuals_hz'])
    ax.axvline(0,color=GRAY,linewidth=.7)
    ax.plot(residual,0,'o',color=BLUE,markersize=4)
    ax.plot(alternate,1,'D',color=GREEN,markersize=3.8)
    ax.hlines(2,lo,hi,color=GRAY,linewidth=1.6)
    ax.plot([lo,hi],[2,2],'|',color=GRAY)
    for y, text_label in enumerate([f'Primary  {residual:+.2f} Hz', f'Alternate  {alternate:+.2f} Hz',
                                     f'Split span  {lo:+.2f} to {hi:+.2f} Hz']):
        ax.text(-5.4,y-.21,text_label,fontsize=8)
    ax.set(xlim=(-6,8),ylim=(2.6,-.55),yticks=[],xticks=[-5,0,5])
    ax.set_xlabel('Observed − predicted (Hz)',fontsize=8)
    ax.spines['left'].set_visible(False)
    note(fig, f"21CMA, {data['channel_hz']/1e6:.3f} MHz. Two packets, one usable state vector. "
              'The split-preamble span is descriptive.')
    return fig


FIGURES = [
    ('01-capture-evidence', capture_evidence),
    ('02-qin-full-word-coverage', full_word_coverage),
    ('03-qin-partial-core-coverage', partial_core_coverage),
    ('04-channel-estimation-acceptance', estimation_and_acceptance),
    ('05-mcs57-receiver-crc', receiver_and_crc),
    ('06-mcs57-repeated-observations', repeated_observations),
    ('07-qin-512-projections', aligned_projections),
    ('08-ut250-headers-narrowband', headers_and_narrowband),
    ('09-vhf-differential-doppler', vhf_doppler),
]


def main():
    output = HERE / 'figures'
    output.mkdir(exist_ok=True)
    for name, build in FIGURES:
        fig = build()
        title = name[3:].replace('-', ' ')
        fig.savefig(output / f'{name}.pdf', metadata={'Title': title, 'CreationDate': None})
        fig.savefig(output / f'{name}.svg', metadata={'Title': title, 'Date': None})
        plt.close(fig)


if __name__ == '__main__':
    main()
