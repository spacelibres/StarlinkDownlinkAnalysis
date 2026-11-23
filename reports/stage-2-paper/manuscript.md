# Recovering Starlink Ku-Band Downlink Codewords with In-Frame Channel References

stone  
[vistone868@gmail.com](mailto:vistone868@gmail.com)  
[spacelibres/StarlinkDownlinkAnalysis](https://github.com/spacelibres/StarlinkDownlinkAnalysis)

Research article · 23 November 2026

**Abstract.** How many unresolved Starlink Ku-band codewords can be recovered by improving the receiver when the code is already known, and what prevents recovery when only part of the code is known? We reprocess a public archive of 1,009 OFDM frames using received symbols from previously verified words in the same frame to estimate the residual channel. Across four modulation and coding schemes (MCS), 41,169 of 43,040 words pass full channel and withheld-waveform checks, up from 39,415 after frame-end closure. Across four schemes with a shared verified parity region, 2,184 of 2,792 words pass the systematics and high-address parity checks, up from 1,648. For MCS 57, the revised receiver adds eight CRC-24 directories. Same-group repeats corroborate 1,145 erased positions and oppose 53 on the target’s low-margin observations. Every target still contains erasures. Mask-aware 512-bit projections validate all 28 high-address controls but identify no low-address relation. A separate 137.055 MHz VHF test compares a received orbital state with differential preamble Doppler and finds a +1.08 Hz residual for one packet pair. These results separate recovery limited by the receiver, by missing channel-code constraints, and by the available observations.

**Index terms.** Channel estimation, erasure masks, OFDM, passive observations, quasi-cyclic codes, Starlink.

## 1. Introduction and observations

The earlier study [1] extracted timing, satellite numbering and Earth-fixed state from the first allocation, called Code 5, and tested the state against relative arrival times. Here we use those received words to recover later allocations and test missing parity relations. The waveform and pilot models follow Humphreys et al. [2] and Qin et al. [3]. The two-header geometry and coding models build on Liu et al. [4]. H1 denotes the first header. H2 gives subsequent allocations as MCS labels and word counts. An allocation is a contiguous span of codeword resources. [Earlier paper](../interim-paper/manuscript.md) · [Liu et al., v2](https://arxiv.org/abs/2610.29387v2)

The principal input is the Qin archive of complex frequency-domain observations [3]. All 1,009 archived frames have measured ends consistent with H2. The archive and the 250 MS/s IQ capture overlap at the source. Their agreement checks transfer across representations. The older 480 MS/s capture is distinct. Native narrowband IQ and the VHF voltage recordings address separate measurement questions. A replay, a masked branch, or a repeated numerical result shares the underlying RF evidence. [Frame-end audit](data/evidence_summary.json) · [Input and header ledger](README.md)

| Source | Data and units | Role |
| --- | --- | --- |
| Qin Ku archive | 1,009 processed frames | Coverage and channel recovery |
| 250 MS/s Ku | 50 ms complex IQ | Ten main-grid headers, plus offset events |
| Legacy 480 MS/s Ku | About 24 ms complex IQ | 18 reference headers |
| LEO Tracker Ku | Ten native 2.5 MS/s windows | H2 aperture masks |
| 21CMA VHF | Two 480 MS/s real-voltage windows | One Doppler difference |

### Figure 1. Capture counts and the evidence hierarchy from measured waveforms to interpreted fields. Counts describe different units. The Qin archive and 250 MS/s capture share source evidence.

## 2. Recovery census

For MCS 66, 70, 85 and 88, a full-channel acceptance refers to all modeled systematic and parity positions. Allocation closure requires every word in that allocation, so the word and allocation denominators measure different outcomes. Starting with the measured frame-end baseline, the additional passes recover 1,754 full words and close 215 allocations. The resulting coverage is 95.65% by word and 89.60% by allocation. Of 1,871 words still lacking acceptance, 1,661 belong to MCS 85 or 88. CRC-24–verified directories reach 375 across these four formats. [Baseline](data/evidence_summary.json) · [Final word and allocation counts](data/evidence_summary.json)

| Step | Full words added | Full words total | Allocations closed |
| --- | ---: | ---: | ---: |
| Measured frame-end baseline | — | 39,415 | 2,146 |
| PN15-assisted terminal recovery | 83 | 39,498 | 2,194 |
| Same-allocation references | 222 | 39,720 | 2,214 |
| Same-frame, other MCS | 500 | 40,220 | 2,284 |
| Initial Code 5 transfer | 141 | 40,361 | 2,284 |
| Coverage sweep | 808 | 41,169 | 2,361 |
| Total eligible | — | 43,040 | 2,635 |

### Figure 2. Cumulative full-word acceptance and allocation closure for MCS 66, 70, 85 and 88. The two panels retain their own denominators.

MCS 71, 75, 78 and 81 share a partially verified encoder, with a systematic region and verified high-address parity. Transferred Code 5 estimates recover 536 additional cores. The 763 of 952 allocations with closed systematic bytes include 177 newly closed allocations. Closing gaps also makes 240 new outer records continuously readable across 721,766 bytes, including previously recovered bytes within those intervals. One directory CRC-24 is verified in this four-format set. The low-address parity region retains separate bit values and erasures. [Partial-core replay](data/evidence_summary.json)

| MCS | Retried words | Added cores | Accepted cores | Rate |
| --- | ---: | ---: | ---: | ---: |
| 71 | 579 | 290 | 1,126 / 1,415 | 79.58% |
| 75 | 230 | 79 | 412 / 563 | 73.18% |
| 78 | 263 | 144 | 490 / 609 | 80.46% |
| 81 | 72 | 23 | 156 / 205 | 76.10% |
| Total | 1,144 | 536 | 2,184 / 2,792 | 78.22% |

### Figure 3. Cumulative high-address core acceptance and new cores among retried words by MCS. Core acceptance covers the systematics and verified high-address parity.

## 3. Channel estimation and waveform acceptance

The Qin FFT archive has already been equalized. The receiver estimates its residual frequency-selective response after removing a complex scalar per reference slot. For a known reference symbol xᵢ and complex observation yᵢ, the scalar âₜ is the least-squares projection in slot t. A local residual gain ĝₖ pools the power and cross-products after scalar correction, over a 17-data-carrier window Wₖ centered at k. Residual power supplies the variance estimate, with normalized variance clipped to [0.0125, 0.5]. On target slots the scalar uses the opposite carrier parity from the symbols being scored. [Estimator and masks](data/evidence_summary.json) · [Code 5 transfer](data/evidence_summary.json)

> (1) âₜ = Σᵢ conj(xᵢ)yᵢ / Σᵢ |xᵢ|², summed over retained reference symbols in slot t.

> (2) ĝₖ = Σᵢ∈Wₖ conj(xᵢ)(yᵢ/âₜ) / Σᵢ∈Wₖ |xᵢ|².

Given a normalized constellation, the receiver sums symbol likelihoods over each bit label. It uses either a fixed variance or the transferred estimate. The log-likelihood ratio is positive when label 0 has greater likelihood. Each withheld bit receives zero log-likelihood ratio and its symbol is excluded from the scalar fit in that branch. A separate re-encoder tests the result against the withheld complex observations z using their energy to normalize the squared error of prediction p. [Matched acceptance criteria](data/evidence_summary.json)

> (3) Lᵦ(z) = log [Σₓ:ᵦ₌₀ exp(−|z − x|²/σ̂²) / Σₓ:ᵦ₌₁ exp(−|z − x|²/σ̂²)].

> (4) E(z,p) = Σᵢ |zᵢ − pᵢ|² / Σᵢ |zᵢ|².

### Algorithm 1. Transfer the channel and assess the word

```text
Freeze the reference words, constellation, code, and target coordinates.
For known QAM words, regenerate symbols using each word’s own MCS.
For Code 5, take BPSK signs only where the native mask records reception.
Estimate slot scalars, local carrier gains, and residual variances.
Run a branch with all symbols and two branches with disjoint masks.
Keep erased symbols out of each branch’s fit and assign them zero LLR.
Require all three branches to agree and pass the applicable parity tests.
Re-encode. Score masked systematic and parity samples separately.
Accept when both masks beat unit error and the shifted-block prediction.
Store every failed test, erasure, and target coordinate for audit.
```

Each mask contains 32 systematic and 32 parity symbols. The partial-core receiver also excludes low-address parity and symbols straddling its boundary. Terminal recovery uses the frozen PN15 sequence only after a directly detected padding boundary. The two masks remain erased even when their values could be supplied by PN15. The 83 accepted terminal words agree across all branches. For 69, no branch reproduces the accepted word when the PN15 prior is disabled. This tests the contribution of the reference on identified padding. [Terminal receiver](data/evidence_summary.json) · [Partial-core receiver](data/evidence_summary.json)

| Targets and paired receiver change | Eligible | Accepted | Matched comparison |
| --- | ---: | ---: | --- |
| Nearest verified word in allocation | 298 | 222 | Flat receiver passes 3 |
| Verified word in same frame, other MCS | 1,709 | 500 | Flat passes 2 |
| Two Code 5 references, initial test | 347 | 141 | Gain only 133, variance only 0, flat 0 |
| Two Code 5 references, coverage sweep | 2,473 | 775 | 731 also pass gain only |
| New same-frame QAM references | 80 | 33 | 26 also pass gain only |

Across 249 frames in the sweep and its controls, estimating one Code 5 word from the other gives 498 directed comparisons. Correctly aligned carrier estimates outperform flat gain in 490 and a 251-carrier shift in all 498. Median normalized errors are 0.0350, 0.0430 and 0.0540, respectively. The comparisons reuse frames and quantify the paired receiver effect on this archive. [Same-allocation comparison](data/evidence_summary.json) · [Cross-MCS comparison](data/evidence_summary.json) · [Coverage sweep](data/evidence_summary.json)

### Figure 4. Paired channel estimation and waveform validation. The retained carrier fit feeds the unmasked branch and two disjoint whole-symbol erasure branches. Withheld positions stay erased until the re-encoded word is tested against the observed waveform.

## 4. MCS 57 remains a code-identification problem

The Code 5 receiver replays all 14,330 word windows across 487 MCS 57 allocations using the frozen 16QAM labeling. At the same 0.08 decision-margin threshold, the joint estimate adds 1,447,124 visible bits and returns 730,320 formerly visible bits to erasure. Among common observations, 50,142 of 113,705,831 decisions disagree. On the narrower frozen PN15 comparison of 237,930 visible positions, bit errors fall from 651 to 612, or 0.274% to 0.257%. PN15 provides a known-padding control. The unknown-bit disagreements remain archived. [MCS 57 replay and masks](data/evidence_summary.json)

The revised receiver confirms 31 directory coordinates, eight new. The deduplicated historical total is 45. It also adds 24 outer-record coordinates with 69,379 observed bits across their headers and bodies. All 24 retain erasures. At the strong margin there is no complete nonterminal word for a masked rank fit. Across six registrations for each hypothesized period of 128 and 256, all 285 received four-term candidates appear once. The prespecified two-nomination gate admits none to held groups. A synthetic four-term control passes 3,583 same-mask comparisons. Its shifted control passes 49.68%. [Directory and sparse-check evidence](data/evidence_summary.json)

### Figure 5. MCS 57 receiver effects at the frozen margin, error counts on common PN15 positions, and the deduplicated directory CRC-24 history.

Repeated observations test a different source of evidence. Candidate pairs qualify on systematic and parity regions separately and are evaluated in both directions on withheld whole symbols. Selected donors are in the same archive group as the target. Their 16-symbol log likelihoods are summed at matched coordinates, in the manner of soft combining [5]. The resulting bit is proposed only if each donor agrees, the donor margin clears the strong threshold, and all six earlier receiver views erased that bit. Target views estimated from the two Code 5 words then corroborate or oppose the proposal at the lower margin of 0.02. [Repeated-word replay](data/evidence_summary.json) · [Chase combining](https://doi.org/10.1109/TCOM.1985.1096314)

> (5) ℓpool(x) = Σd ℓd(x). The pooled bit score uses the constellation subsets with labels 1 and 0.

| Donors | Targets | Held bits | Disagree | Shifted disagree | Target agrees / opposes |
| --- | ---: | ---: | ---: | ---: | ---: |
| One | 211 | 419,869 | 266 | 211,021 | 660 / 42 |
| Two | 75 | 142,974 | 33 | 71,133 | 485 / 11 |

The 181 qualifying pairs cover 286 target words, of which 219 are new accepted coordinates for this waveform comparison. The paired frames share H1, H2 and Code 5 time and counter within each pair. That dependence limits causal interpretation of the duplicate observations, including how they arose at transmission or in the archive. Of 1,726 donor-proposed positions, 1,145 have target corroboration, 53 face target opposition, and 528 are undecided by the target. The most complete word still has eight erasures. The one- and two-donor groups contain different targets, so their error rates describe those groups. [Held-waveform comparison](data/evidence_summary.json)

### Figure 6. Held-position disagreements with one or two donors and low-margin support or opposition from the target’s own waveform. The donor groups use different target words.

## 5. Locating missing parity with masked projections

The partial encoder aligns the systematic bits by right shortening. The assessment uses 512-bit windows from both edges of the low-address interval, preserving the unavailable part of each fragment as masked. The first verified 512-bit high-address block is a positive control. A fold of width m maps each residue j to the XOR of all bit positions in that residue. Its mask is the conjunction of the corresponding observations. Unknown bits therefore cannot contribute a complete output equation. [Folding evidence and bit conventions](data/evidence_summary.json) · [512-bit assessment](data/evidence_summary.json)

> (6) Fₘ(b)[j] = XORᵣ b[j + rm] and Mₘ[j] = ANDᵣ seen[j + rm], for 0 ≤ r < 512/m.

For each visible projection the design uses the circularly aligned systematic folds and an affine intercept. Let A be the resulting binary design, θ its coefficients and q the observed projected parity. Rank measures identifiability. Any inconsistent reduced row rules out a single exact fit under the model and retained observations.

> (7) Aθ = q over GF(2). Unique calibration requires rank(A) = dim(θ) and no inconsistent rows.

### Algorithm 2. Screen a cyclic parity hypothesis

```text
Right-align each systematic word with zero shortening and fold its blocks.
For moduli 1, 2, 4, 8, 16, 32, and 64, fold each candidate parity window.
Include a row only if every constituent output bit is directly observed.
Use whole assigned archive units for calibration and holdout.
Fit an affine relation by GF(2) elimination and record rank and residuals.
Score a unique nonconstant fit on held rows and shifted systematic blocks.
Accept only with enough held rows, zero nominal errors, and shift errors.
Keep rank deficits and contradictions as separate outcomes.
```

The inherited partition contains 30 calibration and 30 holdout units after excluding earlier calibration groups and coupling identified repeats. No new partition is fitted to the result. The 28 format-specific high-address controls all reach full rank and give zero held errors. Of the 98 format-specific low-address tests, 70 lack complete calibration projections, 17 have rank deficits with no contradictions, and 11 have contradictions. In the pooled test of the 512-bit window immediately before the verified high region, both the 32- and 64-residue models reach full rank, yet their equations conflict. A shortage of independent rows explains some tests. The contradictory cases reject exact agreement given the masks and bit decisions. Their origin could include decision errors or differences in encoding structure. [Whole-group design](data/evidence_summary.json) · [Per-window and pooled JSON](data/evidence_summary.json)

| Pooled low-block fold | Rank / terms | Calibration rows | Contradictions |
| --- | ---: | ---: | ---: |
| 1, 2, 4 or 8 | 0 | 0 | 0 |
| 16 | 24 / 369 | 25 | 0 |
| 32 | 737 / 737 | 906 | 80 |
| 64 | 1,473 / 1,473 | 10,084 | 4,370 |

### Figure 7. Attained rank and calibration contradictions for 512-bit projections. The pooled rightmost low-address window becomes full rank at 32 and 64 residues, while the format-specific verified high-address controls remain consistent.

## 6. Weak headers and native narrowband limits

The 250 MS/s observation contains 51 deduplicated synchronization events, including offset tracks. Verified coverage is 37 H1 and 26 H2. Two offset events have allocation information supported by complementary convolutional-output subsets and measured body structure. At 19.425854 ms, fixing the non-CRC bits and blindly ranking all 256 CRC bytes on 42 coded resources returns `EC` in 24 of 24 comparisons and 0 of 216 shifted or wrong-PN controls. The complementary halves fail to reproduce it consistently. The halves at 26.092421 ms also conflict. Three historical positive controls fail this stricter half-split gate. [Event ledger](data/evidence_summary.json) · [CRC scoring and distance-spectrum partitions](data/evidence_summary.json)

| Time (ms) | Body-supported allocations | CRC evidence |
| ---: | --- | --- |
| 15.425910 | MCS 70 × 40, MCS 92 × 1 | Historical H2, split test unstable |
| 19.425854 | MCS 70 × 43, MCS 78 × 1 | Pooled `EC`, unstable halves |
| 26.092421 | MCS 70 × 3, MCS 85 × 41 | `7D` in one half, conflict |

For the ten native 2.5 MS/s LEO Tracker windows [6], the frequency mask can be projected onto the 1,009 Qin H2 references. This conditional calculation fixes H1, H2 start, prefix, descriptor count and padding and assumes correct in-band decisions. The two-receiver union touches encoded resources in 175 references and constrains descriptors in 87. Its maximum rank is five. No allocation list is unique. A pair of measured Qin headers with matching geometry and identical five visible coded bits still has different allocations. The result quantifies the information that those observed apertures can carry under the assumptions. [Native-mask rank audit](data/evidence_summary.json)

### Figure 8. H1 and H2 coverage among 51 events, three split-CRC cases with body-supported allocations, and conditional descriptor ranks under native 2.5 MS/s receiver masks.

## 7. A separate VHF orbital-state check

The two public 21CMA voltage windows at 137.055 MHz are separated by 99.987 s [7]. Both packets pass the PHY-header and payload CRCs. The first has a valid GPS-tagged ECEF state and the second has a zero GPS week and all-zero state fields. Opposite chirp directions supply a first-order time-offset cancellation. With signed chirp slope κ and timing error δt, the dechirped peaks have equal and opposite timing contributions. [Raw replay](data/evidence_summary.json) · [21CMA paper, v1](https://arxiv.org/abs/2610.37158v1)

> (8) f̂CFO = (median f↑ + median f↓)/2, with f↑, f↓ ≈ fCFO ± κδt.

Propagate the first packet’s state using its GPS epoch and a J₂ gravity model, then project the velocity along the line of sight from the fixed published station position rₛ. Differencing the two predicted received frequencies cancels a common constant offset. [Published station and events](https://arxiv.org/html/2610.37158v1#S2) · [Doppler computation](data/evidence_summary.json)

> (9) fD = −f₀ (r − rₛ)·v / (c ‖r − rₛ‖). The predicted difference is fD(t₂) − fD(t₁).

| State or epoch interpretation | Predicted Δf (Hz) | Observed − predicted (Hz) |
| --- | ---: | ---: |
| GPS, ECEF velocity, J₂ | −3,525.00 | +1.08 |
| GPS, ECEF velocity, spherical gravity | −3,524.86 | +0.94 |
| GPS, treat velocity components as inertial | −3,454.92 | −69.00 |
| Unix field as state epoch | −2,888.22 | −635.69 |

The measured difference is −3,523.92 Hz. The alternate downconversion gives a +1.11 Hz residual. Two disjoint preamble subsets across the two algorithms give residuals from −3.51 to +6.39 Hz. These are within-pair sensitivity results. One packet pair supports the GPS epoch and Earth-fixed state interpretation. Light time, site effects and clock offsets lack independent calibration, and the comparison leaves the Unix field’s purpose open. This VHF result has its own signal, receiver and packet format. [Measured frequencies and model variants](data/evidence_summary.json)

### Figure 9. 21CMA differential Doppler from one packet pair and the alternatives for the received state. Preamble-subset sensitivity bounds apply only to this VHF comparison.

## 8. Conclusions and collaboration

In-frame channel estimates account for substantial additional recovery where a verified encoding model already exists. For a partial encoder, the same receiver expands the set of readable systematic bytes while retaining the unverified parity region. MCS 57 offers evidence for directories and repeated waveform positions, but the lack of complete high-margin nonterminal words and reproducible sparse constraints still blocks full-code identification. The 512-bit tests provide a usable calibration baseline for separating absent observations from mutually inconsistent equations. Outer-record boundaries, CRCs and checks on received waveforms establish specific transport-level evidence. Verified application-message count remains zero. [Current evidence ledger](README.md)

We invite collaborators with experience in error-correcting code identification, OFDM soft receivers, or timing and orbit estimation. Concrete contributions include testing candidate MCS 57 checks against frozen whole-group splits, explaining contradictory low-address rows with a model that preserves the verified high-address controls, reproducing weak H2 on the archived IQ, and independently evaluating the one-pair Doppler estimate. Reports of failed hypotheses are welcome alongside runnable analyses and explicit masks. Existing public recordings suffice for these contributions. Contact [stone](mailto:vistone868@gmail.com), or visit the [project repository](https://github.com/spacelibres/StarlinkDownlinkAnalysis).

## 9. Data and reproducibility

The archive records per-input hashes, coordinates, masks, exclusions, reference provenance, group assignments, and measured controls. The two completed 512-bit executions are deterministic replays on fixed inputs. Receiver scores compare the same targets across branches, and counts are descriptive of their archived units. All work uses existing recordings. Public input sources are listed below. The local evidence archive indexes the detailed results used for these figures and tables. [Provenance and replay instructions](README.md) · [Archive status](README.md)

| Result | Evidence and replay |
| --- | --- |
| Full words and allocations | [E240](data/evidence_summary.json), [E241](data/evidence_summary.json), [E242](data/evidence_summary.json), [E243](data/evidence_summary.json), [E244](data/evidence_summary.json) |
| Masked MCS 57 and repeats | [E245](data/evidence_summary.json), [E246](data/evidence_summary.json) |
| Partial core and 512-bit model | [E247](data/evidence_summary.json), [E248](data/evidence_summary.json), [completed run](data/evidence_summary.json), [repeat](data/evidence_summary.json) |
| Weak and narrowband H2 | [E218](data/evidence_summary.json), [E220](data/evidence_summary.json), [E236](data/evidence_summary.json) |
| VHF voltage and orbit check | [E206](data/evidence_summary.json), [E237](data/evidence_summary.json) |

## References

[1] stone, “Control Records in the Starlink Ku-Band Downlink: Frame Layout, Timing and Orbital State,” 22 Nov. 2026. [Interim manuscript](https://github.com/spacelibres/StarlinkDownlinkAnalysis/releases/tag/paper-2026-11-22).

[2] T. E. Humphreys, P. A. Iannucci, Z. M. Komodromos, and A. M. Graff, “Signal Structure of the Starlink Ku-Band Downlink,” IEEE Trans. Aerosp. Electron. Syst., 59(5), 6016–6030, 2023. [Journal article](https://doi.org/10.1109/TAES.2023.3268610).

[3] W. Qin, M. L. Psiaki, J. R. Bowman, and T. E. Humphreys, “Pilots and Other Predictable Elements of the Starlink Ku-Band Downlink,” npj PNT 2, 69, 2026. [Journal article](https://doi.org/10.1038/s44459-026-00075-6) · [arXiv v1](https://arxiv.org/abs/2602.02627v1) · [Author data and code](https://rnl-data.ae.utexas.edu/datastore/supplementaryMaterial/qin-starlink-pilots/).

[4] H. Liu, Y. Ma, H. Zhao, C. Li, and M. Yang, “Protocol Data Unit Header Decoding and Payload Coding Recovery in the Starlink Ku-Band Downlink,” arXiv:2610.29387v2, Oct. 2026. [Preprint v2](https://arxiv.org/abs/2610.29387v2) · [250 MS/s data release v2.0.0](https://github.com/misaka666/Starlink-Downlink-Analysis/releases/tag/v2.0.0).

[5] D. Chase, “Code Combining—A Maximum-Likelihood Decoding Approach for Combining an Arbitrary Number of Noisy Packets,” IEEE Trans. Commun., 33(5), 385–393, 1985. [Original article](https://doi.org/10.1109/TCOM.1985.1096314).

[6] LEO Tracker contributors, “leo-tracker-reduxredux,” receiver software and observations. [Repository and data index](https://github.com/misko/leo-tracker-reduxredux).

[7] Y. Yang et al., “Detection and characterisation of Starlink VHF TT&C emissions at 137.055 MHz with the 21 CentiMeter Array,” arXiv:2610.37158v1, Oct. 2026. [Preprint v1](https://arxiv.org/abs/2610.37158v1) · [Voltage data](https://zenodo.org/records/23309576).
