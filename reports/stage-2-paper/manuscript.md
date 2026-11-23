# Recovering Starlink Ku-Band Downlink Codewords with In-Frame Channel References

stone  
[vistone868@gmail.com](mailto:vistone868@gmail.com)  
[spacelibres/StarlinkDownlinkAnalysis](https://github.com/spacelibres/StarlinkDownlinkAnalysis)

Research article · 23 November 2026

**Abstract.** We estimate the residual channel in Starlink Ku-band downlink recordings using received symbols from verified codewords in the same orthogonal frequency-division multiplexing (OFDM) frame. In a public dataset of 1,009 frames, the number of codewords meeting complete parity and held-out waveform criteria increases from 39,415 to 41,169 of 43,040 across four modulation and coding schemes (MCS). Across four schemes with a known subset of parity equations, the number of blocks for which the systematic and high-address parity bits are verified increases from 1,648 to 2,184 of 2,792. For MCS 57, the revised receiver identifies eight additional directories that pass CRC-24. Repeated observations within recording groups agree with the target’s lower-margin decisions at 1,145 erased positions and disagree at 53. Every target retains erasures. Masked 512-bit projections recover the verified high-address relation in all 28 controls. The tested low-address relations either lack data, are underdetermined, or yield inconsistent equations. In a separate 137.055 MHz VHF observation, a state vector from one packet predicts the differential preamble Doppler of a packet pair with a residual of +1.08 Hz (observed minus predicted). The results distinguish limits imposed by the receiver, the unknown code, and the available bandwidth.

**Index terms.** Channel estimation, erasure masks, OFDM, passive observations, quasi-cyclic codes, Starlink.

## 1. Introduction and observations

An earlier study [1] extracted timing, satellite numbering and Earth-fixed state from Code 5, the first allocation in a frame, and tested the state against relative arrival times. Here we use received Code 5 symbols and other in-frame references to estimate the channel for subsequent allocations. The waveform and pilot models follow Humphreys et al. [2] and Qin et al. [3]. The two-header layout and coding models build on Liu et al. [4]. H1 denotes the first header. H2 describes subsequent allocations by modulation and coding scheme and codeword count. Each allocation occupies a contiguous span of coded resources.

The primary dataset consists of equalized complex frequency-domain samples published by Qin et al. [3]. The measured ends of all 1,009 frames agree with the ends predicted from H2. These data overlap in acquisition with the 250 MS/s complex in-phase and quadrature (IQ) recording [4], so agreement between them checks the two representations of shared observations. The legacy 480 MS/s recording is a separate acquisition. The LEO Tracker narrowband observations and the 21CMA VHF recordings address separate measurement questions [6, 7]. Comparisons between receiver configurations use samples from the same acquisition.

| Source | Data and units | Role |
| --- | --- | --- |
| Qin Ku archive | 1,009 processed frames | Coverage and channel recovery |
| 250 MS/s Ku | 50 ms complex IQ | Ten main-grid headers, plus offset events |
| Legacy 480 MS/s Ku | About 24 ms complex IQ | 18 reference headers |
| LEO Tracker Ku | Ten native 2.5 MS/s windows | H2 aperture masks |
| 21CMA VHF | Two 480 MS/s real-voltage windows | One Doppler difference |

### Figure 1. Capture counts and the evidence hierarchy from measured waveforms to interpreted fields. Counts describe different units. The Qin archive and 250 MS/s capture share source evidence.

## 2. Codeword and allocation recovery

For MCS 66, 70, 85 and 88, a verified codeword satisfies all modeled parity equations and the waveform comparisons in Section 3. An allocation is fully recovered when every codeword within it meets those criteria. After confirming the measured frame ends, the receiver methods below increased the number of verified codewords by 1,754 and the number of fully recovered allocations by 215. The resulting coverage is 95.65% by codeword and 89.60% by allocation. Of the 1,871 codewords that remain unresolved, 1,661 belong to MCS 85 or 88. Across the four formats, 375 directories pass CRC-24.

| Receiver method | Additional codewords | Cumulative codewords | Fully recovered allocations |
| --- | ---: | ---: | ---: |
| Initial coverage with measured frame ends | — | 39,415 | 2,146 |
| PN15-assisted decoding of final codewords | 83 | 39,498 | 2,194 |
| Same-allocation references | 222 | 39,720 | 2,214 |
| Same-frame, other MCS | 500 | 40,220 | 2,284 |
| Initial Code 5 transfer | 141 | 40,361 | 2,284 |
| Channel estimation on remaining eligible codewords | 808 | 41,169 | 2,361 |
| Total eligible | — | 43,040 | 2,635 |

### Figure 2. Cumulative recovery of complete codewords and allocations for MCS 66, 70, 85 and 88. Dashed lines indicate the total eligible counts.

For MCS 71, 75, 78 and 81, the available encoding model covers the systematic bits and the high-address parity region. Code 5 channel estimates allow both regions to be verified in 536 additional blocks. The systematic bytes are fully recovered in 763 of 952 allocations, including 177 gained through this analysis. Within the newly readable intervals, 240 outer records span 721,766 bytes, including bytes already recovered before the gaps were filled. One directory passes CRC-24. The remaining low-address parity is represented by direct bit decisions and erasures.

| MCS | Previously unresolved blocks examined | Additional verified cores | Total verified cores | Rate |
| --- | ---: | ---: | ---: | ---: |
| 71 | 579 | 290 | 1,126 / 1,415 | 79.58% |
| 75 | 230 | 79 | 412 / 563 | 73.18% |
| 78 | 263 | 144 | 490 / 609 | 80.46% |
| 81 | 72 | 23 | 156 / 205 | 76.10% |
| Total | 1,144 | 536 | 2,184 / 2,792 | 78.22% |

### Figure 3. Verified systematic and high-address parity regions for four MCS formats with partially known codes. Rates use the total number of blocks in each format.

## 3. Channel estimation and waveform validation

Initial equalization follows the processing of the published Qin Fourier-domain data [3]. To estimate the residual frequency-selective response, we first remove a complex scalar per reference slot. Let x_i and y_i denote the known reference symbol and its measured complex value. I_t is the set of retained symbols in slot t, and an asterisk denotes complex conjugation. Equation (1) gives the least-squares scalar estimate. Equation (2) gives the local residual gain at carrier k. The set W_k contains reference samples in a window of 17 data carriers centered on k, and t(i) is the slot of sample i. The normalized residual variance is bounded between 0.0125 and 0.5. Within target slots, the scalar used to score odd carriers is estimated from the even carriers, and vice versa.

> (1) $\hat{a}_t=\frac{\sum_{i\in I_t}x_i^{*}y_i}{\sum_{i\in I_t}|x_i|^2}$

> (2) $\hat{g}_k=\frac{\sum_{i\in W_k}x_i^{*}(y_i/\hat{a}_{t(i)})}{\sum_{i\in W_k}|x_i|^2}$

For a normalized observation z, the receiver computes a log-likelihood ratio at bit position b by summing over the constellation subsets with label 0 or 1 at that position. The variance in Equation (3) is fixed or estimated from the reference symbols. A positive ratio favors 0. Each held-out symbol is excluded from the scalar estimate and its bit likelihood ratios are set to zero. The decoded bits are re-encoded and remodulated to obtain predicted complex samples p_i. The normalized squared error in Equation (4) compares these predictions with the withheld observations z_i.

> (3) $L_b(z)=\log\frac{\sum_{x\in\mathcal{X}_{b,0}}\exp(-|z-x|^2/\hat{\sigma}^{2})}{\sum_{x\in\mathcal{X}_{b,1}}\exp(-|z-x|^2/\hat{\sigma}^{2})}$

> (4) $E(z,p)=\frac{\sum_i|z_i-p_i|^2}{\sum_i|z_i|^2}$

### Algorithm 1. Estimate the channel from in-frame references and validate the recovered bits

```text
Specify the reference blocks, constellation, encoder and target positions.
Remodulate verified QAM reference blocks using their respective MCS.
For Code 5, use only directly observed BPSK signs.
Estimate slot scalars, local carrier gains, and residual variances.
Decode with the applicable permanent erasures, then with each of two disjoint sets of additional whole-symbol erasures.
Exclude erased symbols from the slot fit and set their bit log-likelihood ratios to zero.
Require agreement across all three decoding results and satisfaction of the applicable parity equations.
Re-encode and compare the predicted and held-out waveforms for systematic and parity symbols separately.
Require each normalized held-out error to be below one and below the error of the within-block shifted prediction.

```

Each set of withheld symbols contains 32 systematic and 32 parity symbols. When only part of the code is known, low-address parity and symbols that straddle its boundary are excluded from all three decoding attempts. For a final codeword, PN15 is used as a prior after the padding boundary is detected in the received samples. The two sets of withheld symbols are excluded even where PN15 specifies their values. The three decoding results agree on all 83 recovered final codewords. When the PN15 prior is removed, none of the three attempts reproduces the verified result in 69 of those 83 cases.

| Channel reference | Target codewords | Verified codewords | Comparison on the same targets |
| --- | ---: | ---: | --- |
| Nearest verified word in allocation | 298 | 222 | Flat receiver passes 3 |
| Verified word in same frame, other MCS | 1,709 | 500 | Flat passes 2 |
| Two Code 5 references, initial test | 347 | 141 | Gain only 133, variance only 0, flat 0 |
| Two Code 5 references, remaining eligible targets | 2,473 | 775 | 731 also pass using the gain estimate alone |
| New same-frame QAM references | 80 | 33 | 26 also pass gain only |

A paired comparison in 249 frames uses each of the two Code 5 blocks in turn to estimate the channel of the other, yielding 498 directed comparisons. The correctly aligned carrier estimates produce a lower error than flat gain in 490 comparisons and a lower error than an estimate shifted by 251 carriers in all 498. The corresponding median normalized errors are 0.0350, 0.0430 and 0.0540. Both directions within each frame share the same recorded samples.

### Figure 4. Channel estimation from in-frame references and comparison with withheld waveforms. The decoder uses the permanent erasures and then two disjoint sets of additional whole-symbol erasures. The temporarily withheld systematic and parity symbols are used to evaluate the re-encoded predictions.

## 4. MCS 57 receiver performance and code identification

We apply the channel estimates from Code 5 to all 14,330 codeword windows in 487 MCS 57 allocations, using the same 16QAM bit labeling in both receiver configurations. At a decision margin of 0.08, the joint estimate yields decisions at 1,447,124 previously erased positions and leaves 730,320 previously decided positions erased. The two configurations disagree at 50,142 of 113,705,831 common positions. Among 237,930 common positions in the known PN15 padding, the errors decrease from 651 to 612, or from 0.274% to 0.257%. Outside the known padding, a disagreement indicates sensitivity to the receiver configuration without establishing the transmitted bit.

The revised receiver recovers 31 directories that pass CRC-24. Eight are additional to those already identified, yielding 45 distinct directory positions across the analyses. It also identifies the bounds of 24 additional outer records. Their headers and bodies contain 69,379 observed bits, and every record retains erasures. At the 0.08 margin, every non-final codeword contains at least one erased bit, leaving no complete sample on which to estimate rank. Six searches per hypothesized period of 128 or 256 bits yield 285 four-term parity candidates, each appearing in only one search. The analysis requires a candidate to occur in at least two searches before testing it on separate recording groups. A synthetic four-term relation evaluated under the same observation masks passes all 3,583 comparisons. Shifting one input of that synthetic relation gives a 49.68% pass rate.

### Figure 5. MCS 57 receiver comparisons on common PN15 positions and cumulative counts of distinct directories that pass CRC-24.

Pairs of potentially repeated MCS 57 codewords are selected using systematic and parity positions separately, then compared in both directions on whole symbols withheld from selection. The observations used as references and their target belong to the same recording group. Following soft combining [5], the log likelihood at constellation point x is summed over the set of references D, where the subscript d identifies the contributing observation. The pooled bit likelihood sums over the corresponding bit-label subsets. We consider a proposed bit only at positions erased by all six previous receiver configurations, and require the references to agree at a decision margin of at least 0.08. For comparison, the target waveform is evaluated separately using each Code 5 channel reference. Agreement between those two target estimates at a margin of 0.02 determines whether the target supports or contradicts the proposed bit.

> (5) $\ell_{\mathrm{pool}}(x)=\sum_{d\in\mathcal{D}}\ell_d(x)$

| Reference observations | Target codewords | Held-out bits | Aligned disagreements | Shifted disagreements | Target agreements / disagreements |
| --- | ---: | ---: | ---: | ---: | ---: |
| One | 211 | 419,869 | 266 | 211,021 | 660 / 42 |
| Two | 75 | 142,974 | 33 | 71,133 | 485 / 11 |

The 181 qualifying pairs cover 286 target codewords, including 219 additional positions at which the waveform comparison succeeds. Each pair shares H1, H2 and the time and counter fields in Code 5. The observations cannot determine whether the repetition originated in the transmission or in the construction of the dataset. Of 1,726 bit candidates from repeated observations, the target supports 1,145 and contradicts 53, with 528 unresolved by the target. Even the most complete codeword retains eight erasures. The groups with one and two reference observations contain different targets, so the error rates describe the respective groups.

### Figure 6. Held-out disagreements using one or two repeated reference observations, and agreement or disagreement with the target waveform at a lower decision margin. The two groups contain different targets.

## 5. Testing unknown parity with masked projections

The known part of the encoder aligns the systematic bits using right shortening. Candidate low-address parity is examined in 512-bit windows aligned to either boundary of the interval. Unobserved positions, including those outside a partial window, are treated as erasures. The first verified high-address block of 512 bits is the positive control. For fold width m, each residue j from 0 to m − 1 gives an output F obtained by XOR of the corresponding parity bits b. The indicator s is one at observed positions and zero elsewhere. Its product M in Equation (6) equals one only when every constituent bit is observed.

> (6) $F_m(b)_j=\bigoplus_{r=0}^{512/m-1}b_{j+rm},\qquad M_m(s)_j=\prod_{r=0}^{512/m-1}s_{j+rm}$

For each observed projection, the binary design matrix A contains the circularly aligned systematic folds and an intercept. The vector θ contains the unknown coefficients and q the observed projected parity. Equation (7) has a unique solution only when the system is consistent and the column rank equals the number of coefficients. A reduced row with zero coefficients and a nonzero parity value indicates that the model cannot fit the retained observations exactly.

> (7) $A\theta=q\quad\mathrm{over}\ \mathrm{GF}(2)$

### Algorithm 2. Screen a cyclic parity hypothesis

```text
Right-align each systematic block using zero shortening and fold it by residues.
For moduli 1, 2, 4, 8, 16, 32, and 64, fold each candidate parity window.
Include a row only if every constituent output bit is directly observed.
Fit on the calibration groups and test on separate groups of recordings.
Fit an affine relation by binary elimination and record rank and inconsistent equations.
Evaluate each unique, nonconstant fit on held-out observations and on shifted systematic-block controls.
Retain a model if sufficiently many held-out rows are available, the aligned error count is zero, and the shifted comparison produces errors.
Distinguish underdetermined systems from inconsistent equations.
```

Groups used to develop the existing code model are excluded. Related repeats are kept together and the remaining data are partitioned into 30 calibration and 30 held-out groups. The 28 format-specific high-address controls all reach full rank and give zero errors on held-out observations. Of the 98 format-specific low-address tests, 70 have no complete calibration projections, 17 are rank deficient with no inconsistent equations, and 11 yield inconsistent equations. When pooling the 512-bit low-address window adjacent to the verified high-address region, the models with 32 and 64 residues both reach full rank but produce inconsistent equations. Missing projections or rank deficiency limit some tests. The inconsistent equations indicate that the observed bits and the proposed model cannot hold simultaneously. Potential explanations include bit-decision errors and a mismatch in the assumed code structure.

| Pooled low-block fold | Rank / terms | Calibration rows | Contradictions |
| --- | ---: | ---: | ---: |
| 1, 2, 4 or 8 | 0 | 0 | 0 |
| 16 | 24 / 369 | 25 | 0 |
| 32 | 737 / 737 | 906 | 80 |
| 64 | 1,473 / 1,473 | 10,084 | 4,370 |

### Figure 7. Attained rank and calibration contradictions for 512-bit projections. The pooled rightmost low-address window becomes full rank at 32 and 64 residues, while the format-specific verified high-address controls remain consistent.

## 6. Weak headers and native narrowband limits

Across the main and offset time grids of the 250 MS/s recording, there are 51 distinct synchronization events, with 37 verified H1 and 26 verified H2 headers. Two offset events have allocation fields supported by disjoint subsets of the convolutional outputs and by observed frame-body boundaries. At 19.425854 ms, the non-CRC header bits are held fixed and all 256 CRC byte values are ranked by their agreement with 42 coded resources. The combined resources favor hex EC in all 24 comparisons. None of the 216 comparisons using shifted resources or an incorrect PN sequence favors it. The two complementary subsets do not consistently select the same byte. At 26.092421 ms the subsets also disagree. Three previously verified comparison headers likewise fail this stricter subset-agreement criterion.

| Time (ms) | Body-supported allocations | CRC evidence |
| ---: | --- | --- |
| 15.425910 | MCS 70 × 40, MCS 92 × 1 | Historical H2, split test unstable |
| 19.425854 | MCS 70 × 43, MCS 78 × 1 | Pooled `EC`, unstable halves |
| 26.092421 | MCS 70 × 3, MCS 85 × 41 | `7D` in one half, conflict |

The occupied bands in ten native 2.5 MS/s LEO Tracker windows [6] are projected onto the encoded H2 headers of the 1,009 Qin frames. The calculation assumes known H1, H2 start, header prefix, descriptor count and padding, and correct decisions on all available in-band symbols. In the paired-receiver view, at least one coded resource is visible for 175 H2 headers, and the visible positions constrain an allocation descriptor for 87. The maximum rank is five and no complete allocation list is uniquely determined. Two measured Qin headers illustrate the ambiguity. Their H2 geometries and five visible coded bits agree, although their transmitted allocations differ.

### Figure 8. H1 and H2 coverage among 51 events, three split-CRC cases with body-supported allocations, and conditional descriptor ranks under native 2.5 MS/s receiver masks.

## 7. A separate VHF orbital-state check

The two public 21CMA voltage windows at 137.055 MHz contain packets received 99.987 s apart [7]. Both packets pass the physical-layer header and payload CRCs. The first contains a valid GPS epoch and an Earth-centered, Earth-fixed (ECEF) position and velocity. The second contains a zero GPS week and zero state fields. For a chirp-slope magnitude κ and timing offset δt, the frequency peaks after dechirping are shifted by κδt with opposite signs for up- and down-chirps. Equation (8) averages the median up- and down-chirp peaks to estimate the carrier frequency offset.

> (8) $\hat{f}_{\mathrm{CFO}}=\frac{\mathrm{median}(f_{\uparrow})+\mathrm{median}(f_{\downarrow})}{2}$

The first packet’s state is propagated from its GPS epoch with a J2 gravity model. In Equation (9), the satellite position and velocity are r and v, the published station position is r_s [7], f_0 is the carrier frequency, and c is the speed of light. The predicted difference is the Doppler at the second preamble minus the Doppler at the first. Taking the difference cancels a constant receiver frequency offset common to the two events.

> (9) $f_{\mathrm{D}}=-\frac{f_0(\mathbf{r}-\mathbf{r}_s)\cdot\mathbf{v}}{c\Vert\mathbf{r}-\mathbf{r}_s\Vert}$

| State or epoch interpretation | Predicted Δf (Hz) | Observed − predicted (Hz) |
| --- | ---: | ---: |
| GPS, ECEF velocity, J2 | −3,525.00 | +1.08 |
| GPS, ECEF velocity, spherical gravity | −3,524.86 | +0.94 |
| GPS, treat velocity components as inertial | −3,454.92 | −69.00 |
| Unix field as state epoch | −2,888.22 | −635.69 |

The measured difference is −3,523.92 Hz. An alternative downconversion gives a residual of +1.11 Hz. The two disjoint preamble subsets, each processed with both downconversions, yield residuals from −3.51 to +6.39 Hz. This range describes sensitivity within one packet pair. These observations support interpreting the state epoch as GPS time and the state as Earth-fixed. Light-time effects, site effects and oscillator offsets lack independent calibration, and the meaning of the Unix field remains open. This VHF measurement uses a separate signal, receiver and packet format from the Ku-band observations.

### Figure 9. 21CMA differential Doppler from one packet pair and the alternatives for the received state. Preamble-subset sensitivity bounds apply only to this VHF comparison.

## 8. Conclusions and collaboration

Channel estimates from the same frame increase the number of recoverable codewords for the four formats with complete code models. For the four formats with partially known codes, they extend the range of recoverable systematic bytes. The low-address parity equations remain undetermined. For MCS 57, directory CRCs and repeated waveform positions constrain portions of the stream. Erasures in non-final codewords and the lack of reproducible sparse parity relations leave the full code unresolved. The 512-bit projections distinguish missing observations from inconsistent equations. The observed record boundaries, CRCs and received waveform comparisons support transport-layer conclusions. Application-layer message contents have yet to be verified.

Further tests can compare candidate MCS 57 parity relations on separate recording groups, seek a low-address parity model compatible with the high-address controls, and assess weak H2 headers against the published IQ. Independent analysis of the measured VHF Doppler would help characterize the sensitivity to timing and downconversion. Correspondence may be addressed to the author.

## 9. Data availability

The analysis uses previously collected measurements. The public inputs and their roles are listed below. Codeword and allocation statistics refer to the Qin recordings. The conclusions about narrower bandwidth condition on the measured LEO Tracker passbands, and the VHF comparison refers to one 21CMA packet pair.

| Public data | Use in this study |
| --- | --- |
| Qin et al. [3] | Complex frequency-domain frame samples for coverage, channel estimation and parity hypotheses |
| Liu et al. [4] | 250 MS/s Ku-band IQ for the header and frame-boundary comparisons |
| LEO Tracker [6] | Native narrowband windows and paired receiver passbands |
| Yang et al. [7] | Two 21CMA VHF voltage windows and the published station coordinates |

## References

[1] stone, “Control Records in the Starlink Ku-Band Downlink: Frame Layout, Timing and Orbital State,” 22 Nov. 2026. [Interim manuscript](https://github.com/spacelibres/StarlinkDownlinkAnalysis/releases/tag/paper-2026-11-22).

[2] T. E. Humphreys, P. A. Iannucci, Z. M. Komodromos, and A. M. Graff, “Signal Structure of the Starlink Ku-Band Downlink,” IEEE Trans. Aerosp. Electron. Syst., 59(5), 6016–6030, 2023. [Journal article](https://doi.org/10.1109/TAES.2023.3268610).

[3] W. Qin, M. L. Psiaki, J. R. Bowman, and T. E. Humphreys, “Pilots and Other Predictable Elements of the Starlink Ku-Band Downlink,” npj PNT 2, 69, 2026. [Journal article](https://doi.org/10.1038/s44459-026-00075-6) · [arXiv v1](https://arxiv.org/abs/2602.02627v1) · [Author data and code](https://rnl-data.ae.utexas.edu/datastore/supplementaryMaterial/qin-starlink-pilots/).

[4] H. Liu, Y. Ma, H. Zhao, C. Li, and M. Yang, “Protocol Data Unit Header Decoding and Payload Coding Recovery in the Starlink Ku-Band Downlink,” arXiv:2610.29387v2, Oct. 2026. [Preprint v2](https://arxiv.org/abs/2610.29387v2) · [250 MS/s data release v2.0.0](https://github.com/misaka666/Starlink-Downlink-Analysis/releases/tag/v2.0.0).

[5] D. Chase, “Code Combining—A Maximum-Likelihood Decoding Approach for Combining an Arbitrary Number of Noisy Packets,” IEEE Trans. Commun., 33(5), 385–393, 1985. [Original article](https://doi.org/10.1109/TCOM.1985.1096314).

[6] LEO Tracker contributors, “leo-tracker-reduxredux,” receiver software and observations. [Repository and data index](https://github.com/misko/leo-tracker-reduxredux).

[7] Y. Yang et al., “Detection and characterisation of Starlink VHF TT&C emissions at 137.055 MHz with the 21 CentiMeter Array,” arXiv:2610.37158v1, Oct. 2026. [Preprint v1](https://arxiv.org/abs/2610.37158v1) · [Voltage data](https://zenodo.org/records/23309576) · [Station coordinates](https://arxiv.org/html/2610.37158v1#S2).
