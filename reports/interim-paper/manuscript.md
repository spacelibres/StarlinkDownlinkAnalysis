# From Waveform Structure to Control-Record Recovery in the Starlink Ku-Band Downlink

Interim research manuscript · Evidence snapshot v0.0.144 · 21 November 2026

**Abstract.** Public studies have established the Starlink Ku-band orthogonal frequency-division multiplexing (OFDM) waveform, predictable pilots, and several channel codes. We extend these results from frame acquisition and channel decoding to the interpretation of selected control records. Frozen header and cyclic-redundancy checks reproduce both headers in 1,009 Qin archive frames, 18 frames from a separate 480 MS/s capture, and 10 frames from a 250 MS/s replay with source overlap with the Qin archive. We identify reproducible Code 5 bit fields for the satellite number, Earth-fixed position and velocity, and time. An integer timing model agrees with embedded Global Positioning System (GPS) time tags to within 1 ns in 1,037 evaluated codewords. In 381 group-held archive frames, the time and state predict relative time of arrival with a 46.89 ns root-mean-square residual after fitting a constant on calibration frames. Complete equivalent channel codes are recovered for four additional modulation and coding labels, with whole-symbol masking and received-waveform checks. Three Code 28 shortened cores are verified across capture sources. Native narrowband dual-receiver decisions recover a known ten-bit header prefix uniquely in 28 of 46 frames. The results establish selected control semantics and explain where observation bandwidth and receiver decisions still prevent complete decoding.

**Index Terms.** Channel coding, control records, low Earth orbit, OFDM, passive measurements, reproducibility, Starlink.

## I. Introduction

The Starlink Ku-band downlink can be studied by passive reception across several levels. Timing correlations reveal repetition in the waveform. Demodulation maps observed samples to symbols, and channel decoding constrains the sequence of bits. Field interpretation requires additional evidence linking decoded values to quantities measured outside the candidate field. A clock or orbit value therefore requires a different validation from an agreement with a known pilot sequence.

Humphreys et al. established the 240 MHz waveform and its synchronization structure [1]. Qin et al. measured GPS steering and timing irregularities [2], then published edge-pilot sequences, a reference template, and post-header tessellation codes with exemplar frames [3], [4]. Liu et al. reported two terminated convolutionally coded protocol data unit headers, a CRC-8, a conditional allocation layout, and equivalent quasi-cyclic (QC) channel codes [5]. Their reported 18-frame sample is one continuous capture. The public repository distributes the recording and technical notes [6]. These results are the starting point for the present analysis.

This interim manuscript reports three extensions. First, it describes fields inside Code 5, including a time relation checked against both recorded tags and separate propagation observables. Second, it covers wider sets of modulation and coding schemes (MCSs), and recovers record boundaries at multiple levels. Third, it measures which parts of the interpretation survive narrower observation windows. The scope is the measured release through v0.0.144.

## II. Prior Work and Attribution

Table I distinguishes public evidence, project reuse, and the extension supported by our experiments. Here, a project contribution means a result added by the reported analysis relative to the cited sources. The literature comparison is bounded by those sources and their available versions.

### TABLE I. PUBLIC RESULTS AND PROJECT EXTENSIONS

| Source | Public result used | Result added in this study |
|---|---|---|
| Humphreys et al. [1] | OFDM numerology, synchronization and signal structure. | Controlled replay and event analysis of the user WAV; limits on which bits its window can identify. |
| Qin et al. [2]–[4] | Arrival-time behavior, 15 s steering, pilots, reference waveform and tessellation. | Bit-level timing reconstruction and state-field interpretation; CRC and allocation survey across the 1,009 exemplars. |
| Liu et al. [5], [6] | H1/H2, CRC-8, allocations, selected QC code families and Code 28/37 shortened cores. | Cross-capture header/encoder transfer; four additional full-length MCS models, Code 28 masked parity witnesses, and Code 5 internal parsing. |
| Morgan et al. [7] and SpaceX [8] | Timing/allocation model and example downlink behavior; mock fused PNT uses separately supplied orbit/clock products. | Observed Code 5 field offsets and integer relations tested on decoded bits. |
| LEO Tracker [9] and `starlink_beacon_iq` [10] | Native receiver pipeline and exported observations; SDR++ IQ visualization. | Native DS9 projection through the H1 code, and controlled WAV spectral/temporal analysis. |

Historical orbit data [11] serve as external comparators. The Starlink name and catalog identifier come from different sources: a radio-field value is compared with a historical catalog entry only after parsing and propagation. Patent claims and example tables inform hypotheses about scrambling and padding [8]. They do not determine the observed location or meaning of extra bits.

## III. Data and Validation Protocol

### TABLE II. MEASUREMENT AND REPLAY COHORTS

| View | Input and coverage | Use in this work |
|---|---|---|
| Qin exemplars | 1,009 frequency-domain frames from one capture; hard and selectively accessible soft symbols. | Header survey, record semantics, group-held evaluation. |
| UT IQ | 250 MS/s, about 50 ms; nine complete main-track frames and one tail-truncated frame. | Raw-waveform reconstruction and symbol masking; overlap with Qin. |
| Legacy IQ | 480 MS/s, about 24 ms; 18 frames in a separate capture. | Cross-capture transfer and older layouts. |
| User WAV | 2.4 MS/s complex baseband reconstructed from stereo IQ. | Event timing, drift and narrow-window observations. Acquisition time and location remain unverified. |
| LEO Tracker | Ten unique 20 ms raw IQ windows at 2.5 MS/s; other derived observations including 46 DS9 frames. | Native receiver baseline, projected H1 test and visibility limits. |

The 250 MS/s waveform and Qin symbol archive contain matched frames (E034). A result on both paths may demonstrate receiver transfer, but the two paths share RF evidence. The legacy 480 MS/s waveform supplies a second capture. LEO Tracker native recordings, sign exports and the modeled 25 MS/s windows are kept distinct. In particular, applying a software filter to wideband IQ changes its observed bandwidth without adding another acquisition. The WAV, the raw LEO windows and the DS9 sign exports have distinct receiver descriptions and cannot be treated as interchangeable.

The wideband pipeline applies timing and frequency corrections, synchronization-symbol and pilot-based equalization, de-rotation, hard or soft decisions, channel decoding, and record parsing. For the first two headers we use a 32-bit retained input block and six termination bits in a rate-1/3 convolutional code, yielding 114 coded bits per block [5]. Header prefix and termination conventions select candidates before checking the CRC. The working descriptor model uses 20-bit entries, with unresolved interpretations of an index bit and quantity behavior. We use CRC passes and measured allocation endpoints as separate checks when both are available.

For a binary systematic code, write the transmitted word as c = (u, p) and the check equation as H_u u + H_p p = 0 over GF(2). If H_p has full column rank, a consistent information word determines a unique parity vector. Experiments mask measured air-interface symbols or parity positions before solving, then compare predictions with the withheld observations at their original coordinates. For erasures E and observed indices O, the tested linear system is H_E c_E = H_O c_O. A unique completion requires consistency and full column rank for H_E. The observations excluded by receiver uncertainty, unknown label bits and unmodeled code regions remain explicitly marked.

Training and validation split entire correlated groups where fitting is part of the analysis. Reports distinguish fixed control comparisons, repeated replays, overlapping records and post-selection witness checks. Near-duplicate or overlapping Qin groups are excluded from training/holdout comparisons when relevant. Negative controls include wrong frequency windows, resource or bit offsets, reversed count direction, wrong PN phase, shifted parity, and incorrect receiver longitude. Reported sample counts describe their particular cohorts without a per-bit independent-trials assumption. Re-encoding and CRC checks constrain the observable bitstream; transmitter BER and payload authenticity require additional evidence.

## IV. Results

### A. Waveform and Header Reproduction

Within the original WAV event, 33 of 33 held windows showed a correlation peak at the nominal 1.333 ms interval, versus 0 of 9 windows after the event (E047). At about 13.635 s, the sideband component matched to the E042 periodic template fell by about 17–19 dB, against a roughly 0.09 dB change in the fixed center band. Relative sideband termination estimates were consistent with a conditional interval of about −1.3 to +5.3 ms (E090). Uncertainty on this single event leaves same-frame gating and physical attribution open. A cyclic shift of the repeated pattern better explained changes in the visible positions than flipping three fixed positions (E116). The identity associated with this baseband remains unresolved.

The StarLoc recordings [12] provide an additional comparison using a three-receiver capture. A 43.945 kHz comb fitted on one receiver transferred to its two peers in 12 of 17 fixed windows. None of these 12 windows passed the outer-band 1.333 ms correlation criterion on any receiver (E054). In the measured windows the comb and frame-period signatures can therefore occur separately. Neither signature, by itself, supplies a protocol field.

Both wideband headers passed the CRC on 1,009/1,009 Qin exemplars, 18/18 legacy frames and 10/10 UT main-track frames (E010, E014, E036). This extends the frame-by-frame check of the public model, with the legacy count reproducing the original publication's capture. The main-track UT count includes one tail-truncated frame whose headers remain visible. The weaker offset track in the same waveform required separate acquisition, layout and CRC gates. E110 and E119 preserve specific accepted weak candidates and failures instead of merging them with the strong cohort. The working allocation model and independently measured terminal structure agree in the tested strong layouts, with remaining descriptor and weak-track uncertainties treated separately.

### B. Code 5 Control Fields

The directory pointer and an affine CRC-24 are reproducible in two raw wideband paths and 429 held archive first codewords (E091). The 135- and 163-byte first-record spans predict the next marker throughout the 1,009-frame archive. Read offsets below are zero-based within the first record after descrambling, with bit numbering least-significant bit first within each byte.

### TABLE III. SELECTED CODE 5 FIRST-RECORD FIELDS

| Bits | Reading | Validation |
|---|---|---|
| 11–25 | Little-endian unsigned satellite number. | 31848 on Qin/UT and 2472 on legacy. Historical catalog ranks favor NORAD 59784 and 48130, respectively (E096). |
| 60–347 | Three binary64 ECEF coordinates in meters and three binary32 velocities in m/s. | 22 held Qin groups and raw replays (E094). |
| 348–409 | 32-bit GPS seconds followed by 30-bit nanoseconds. | Integer range, rollover and propagation tests (E095). |
| 421–434 | Low 14 bits of relative frame count N. | Frozen count comparisons across the wideband sources (E079, E097). |
| 453–475; 485–519 | Clock correction Q and interval anchor A. | Redundant time reconstruction and counterfactual clock rates (E097). |

The inferred relation, in seconds, is

> t_GPS = E + N/750 + (4,719,600 − Q)/540,000,000, A = 2E + 1. (1)

In the evaluated records E is a GPS-second interval origin divisible by 15. Constants were fitted on 416 directly observed Qin calibration frames and then frozen. The reconstruction agrees within 1 ns with 1,037 embedded tags (1,009 Qin, 10 overlapping UT-path and 18 legacy). On 323 fully observed held frames, 540 MHz gives 0.295 ns RMS residual, versus 62.19 and 61.95 ns for 539 and 541 MHz with fitted constants (E097). This agreement quantifies an encoded relation. The physical clock and constant delay source remain unverified.

An independent field-semantic check compares recorded time of arrival (TOA) with t_GPS + distance/c plus a calibration constant. The distance is computed from the decoded state and the fixed UT reference site. In 381 group-held normal archive frames, the relative TOA residual is 46.89 ns RMS and the residual range-rate variation is 0.498 m/s RMS after calibration. A 1° longitude error gives 5,581.11 ns and 36.117 m/s. Omitting range propagation gives 33,298.28 ns (E095). The Earth-fixed interpretation is supported by timing, Doppler and historical trajectory comparison within these captures. The errors do not represent independently measured absolute ranging or clock accuracy. One off-axis example carries a time differing by about five frame intervals and is retained with its distinct interpretation.

### FIG. 1. Relative TOA RMS on the same 381 held archive frames, shown on a logarithmic scale (E095).

For the two observed number values, the historical catalog object closest in both position and velocity is also the object whose Starlink designation has that suffix (E096). The Qin suffix was used for field discovery. The frozen offset and orientation then transfer to the legacy recording. The two numbers identify the tested wideband captures. The native WAV and LEO captures have unresolved identities. The evidence establishes the bit start and the minimum width required for these two observed values.

### C. Records, Boundaries and Scheduling

Code 5 records with working types `01` and `03` contain a count n, a 12-bit termination marker and an eight-bit tail length q. In the nonempty packed layout the marker begins at p = 161 + 60n relative to the record, and the length in bytes is ceil((p + 24)/8) + q (E141). A legacy `01` has a separate 16 + 7m byte rule. The packed and empty-record checks account for 1,428 lengths in Qin, with 493/493 on the group-held split and 197/197 tested next-record positions. The legacy formula reproduces 18/18 comparisons. These counts measure record extents, with semantic content only for independently interpreted fields.

For a controlled narrow replay with unknown record start, masking the Qin archive to nominal 25 MS/s edge windows yields 20 lower-edge and two upper-edge correct boundaries. One upper-edge false proposal lies within the known preceding records. Filtering legacy IQ to three 25 MS/s interior tunings, then reacquiring synchronization, recovers 14/18 old-format boundaries. Those tunings were chosen from known layouts, and no native LEO record decode follows from them (E142).

The first-record tail also gives a six-bit countdown and a three-bit grouping count. A 53-frame reload rule reproduces 384 held active Qin frames. After calibrating phase with its first frame, the legacy capture predicts 17/17 remaining frames, without observing a reload (E137). Four-bit status positions point to future candidate slots. As an example, 258 of 368 status-1 targets coincide with an archival frame; none of 54 status-0 targets do (E138). The archive is incomplete as a record of transmission, and some boundary states and actual legacy observations contradict a universal presence/absence law (E139). The fixed `03` to `01` transform predicts 82 known pairs byte for byte, including 35/35 whole-cycle held pairs (E140). This result identifies a conditional content relation in one archive.

Outside Code 5, verified packed lengths separate 69 outer records and 499 subintervals across 22 wideband allocations. Two CRC-24-verified MCS 66 directories predict 27 lengths in subsequent MCS 66, 70, 85 and 88 allocations, all matching the relevant field observations (E134). These intervals do not have assigned application meanings. A bounded set of application framing and checksum tests did not yield a validated message (E126, E128, E133).

### D. Channel Codes and Shortened-Core Transfer

Across the reported wideband header cohorts we observe 25 distinct MCS identifiers. Twelve have a full-length equivalent encoder or check model, with some legacy models supported by one capture or cross-format mapping. Two have verified shortened cores. Four have a verified subset of the parity checks, and seven lack a verified full-length code or shortened core. The inventory describes the observed sample and working header convention.

### TABLE IV. MCS EVIDENCE AT THE SNAPSHOT

| Evidence | Observed labels | Result |
|---|---|---|
| Prior families reproduced | 3, 5, 9, 10, 12, 21, 27, 39 | Eight full-length equivalent models; scope varies by format. |
| Added full-length models | 66, 85; 70, 88 | N = 16,384. K = 12,288 and 12,800, respectively. |
| Shortened cores | 28, 37 | 256 and 384 middle positions remain outside the verified core. |
| Added partial checks | 71, 75, 78, 81 | Shared rank 4,096; 512–1,536 parity positions remain to be constrained. |
| Full code unresolved | 6, 35, 42, 57, 68, 92, 93 | Some have allocation, modulation, padding or tentative check evidence. |

MCS 66 and 85 share eight 512-row check families with parity rank 4,096. In the 250 MS/s held strong-frame cohort, 196/196 and 130/130 words agree with the re-encoded results after masking whole parity symbols. Remodulated masked-sample normalized mean square errors (NMSEs) are 0.03320 and 0.03238, compared with 1.99678 and 1.99537 for shifted parity (E128). A related seven-family matrix spans 3,584 transmitted parity degrees of freedom for MCS 70 and 88. The raw held examples are 50/50 and 1/1, with masked-sample NMSEs 0.03298 and 0.02533 (E131). These are receiver-equivalent binary codes. The comparisons establish neither the internal encoder implementation nor performance across a larger receive population.

Liu et al. reported Code 28's shortened (16,128, 8,960) core on legacy IQ [5]. With its generator frozen, 1 of 1,088 Qin hard words is uniquely completed after removing off-lattice symbols and the +511 carrier. Seven separately masked parity blocks predict all 7,015 withheld observations of that word (E145). Two additional cores are recovered from six cached soft words after masking the weakest quarter of the core bits; 10,424/10,424 separately withheld parity observations match and 17 differences in known PN intervals disappear (E146). The witness words were selected after the scans. Three valid cross-capture witnesses do not measure a population decoder success rate. Swapping the 256 physical middle bits into the legacy mother-code shortened prefix causes 3,537 disagreements among 7,015 parity observations in the hard witness. The middle positions are observed but remain unexplained. Across 49 hard words affected by four-corner slots, neither full-slot erasure nor quadrant-only constraints produced a compatible word (E146).

### E. Native Narrowband Evidence and Identifiability

Replaying the external LEO Tracker pilot detector on its ten unique native raw IQ windows yields 7/10 historical frame starts at the default frequency search and 8/10 at a wider range (E143). This is a same-data baseline and the historical detector coordinates are used only afterward. On a distinct DS9 sign-export cohort, 24 visible H1 coded positions depend on ten input bits. Enumerating all 1,024 input values and both BPSK signs, with disagreement treated as erasure, gives 28/46 unique matches to the known prefix, 16 tied predictions and two unique mismatches (E144). The prefix is used after enumeration. Several coordinate shifts produce nearly equal distances, making the assumed resource map important.

The projection rank of an equivalent code makes some observation limits explicit. Even a hypothetical union of 200 registered non-pilot carriers gives no isolated parity check or unique individual out-of-band position for the tested MCS 66/85 projections (E130). A sufficient known padding interval can change algebraic rank, whereas measured decisions may remain inconsistent. The original WAV and the native LEO short windows have no verified complete, CRC-valid unknown message; the 25 MS/s LEO source needed for direct edge-record replay is represented locally by acquisition and frequency receipts rather than continuous raw samples.

## V. Discussion and Limitations

The evidence supports public-waveform and header reproduction, additional equivalent channel-code recovery, and a limited set of verified control semantics. Receivers produce different observation masks. Missing frequencies, slicer collapse, incorrect hard decisions and weak overlapping frames limit a complete receiver even where the corresponding code is known. A full-rank code constraint can be reproduced with different Tanner graphs; structural compatibility and an implementation assignment have different evidentiary requirements.

The reported data are convenience samples, with repeated observations and visible overlap between two representations of one capture. Reliability figures are reported per test cohort. Neither the 1 ns internal timing agreement nor the catalog correspondence constitutes absolute PNT validation or authenticated identity. For 13 of the 25 observed format labels no complete equivalent code is established at this evidence snapshot. The unsolved material also includes two extra middle intervals, remaining parity components, versions of record semantics, unresolved weak heads, and application-level framing and content. The total amount of unknown application data cannot be estimated from the sample.

## VI. Conclusion

Published OFDM structure, pilot sequences and header models support a reproducible chain from IQ to selected control records. The strongest project additions are recorded bit-level time and orbit fields, two independently captured number values linked to historical catalog entries, four additional complete receiver-equivalent MCS codes, and quantitative record and narrowband limits. This evidence supports control decoding for the observed layouts. The versioned release preserves inputs, failed branches and ambiguities for independent review.

## Data and Code Availability

The public source waveforms, exemplar data and upstream repositories are cited in [4], [6], [9], [10] and [12]. Internal artifact paths and replay commands are indexed in the accompanying reproducibility note and in the v0.0.144 project history [13]. Redistribution access remains unverified for the original user WAV, the local external-project snapshot and the native 25 MS/s acquisition. Recorded code/manifest hashes bind primary inputs, output tables and exact controls. Correlator windows and erased bits are retained where provided by the experiments.

## References

[1] T. E. Humphreys, P. A. Iannucci, Z. M. Komodromos, and A. M. Graff, “Signal structure of the Starlink Ku-band downlink,” IEEE Trans. Aerosp. Electron. Syst., vol. 59, no. 5, pp. 6016–6030, Oct. 2023. [DOI 10.1109/TAES.2023.3268610](https://doi.org/10.1109/TAES.2023.3268610).

[2] W. Qin, A. M. Graff, Z. L. Clements, Z. M. Komodromos, and T. E. Humphreys, “Timing properties of the Starlink Ku-band downlink,” IEEE Trans. Aerosp. Electron. Syst. [DOI 10.1109/TAES.2025.3627180](https://doi.org/10.1109/TAES.2025.3627180).

[3] W. Qin, M. L. Psiaki, J. R. Bowman, and T. E. Humphreys, “Pilots and other predictable elements of the Starlink Ku-band downlink,” npj PNT, vol. 2, art. 69, 2026. [DOI 10.1038/s44459-026-00075-6](https://doi.org/10.1038/s44459-026-00075-6).

[4] W. Qin et al., supplementary data and MATLAB code for “Pilots and other predictable elements of the Starlink Ku-band downlink,” 2026. [Zenodo DOI 10.5281/zenodo.21404931](https://doi.org/10.5281/zenodo.21404931).

[5] H. Liu, Y. Ma, H. Zhao, C. Li, and M. Yang, “Protocol data unit header decoding and payload coding recovery in the Starlink Ku-band downlink,” arXiv preprint, Oct. 2026. [arXiv:2610.29387v2](https://arxiv.org/abs/2610.29387v2).

[6] H. Liu et al., “Starlink-Downlink-Analysis,” public recordings and technical notes. [GitHub](https://github.com/misaka666/Starlink-Downlink-Analysis).

[7] S. C. Morgan et al., “Toward pseudorange-based PNT with Starlink: A mock fused LEO GNSS implementation,” arXiv preprint, Nov. 2026. [arXiv:2611.13655](https://arxiv.org/abs/2611.13655).

[8] Space Exploration Technologies Corp., “Configurable orthogonal frequency division multiplexing (OFDM) signal and transmitter and receiver for satellite to user terminal downlink communications,” U.S. Patent 12 074 683 B1, 2024. [Patent text](https://patents.google.com/patent/US12074683B1/en).

[9] LEO Tracker contributors, “leo-tracker-reduxredux,” source code and research reports. [GitHub](https://github.com/misko/leo-tracker-reduxredux).

[10] lixb123, “starlink_beacon_iq,” SDR++ IQ visualization and analysis code. [GitHub](https://github.com/lixb123/starlink_beacon_iq).

[11] IAU Centre for the Protection of the Dark and Quiet Sky, “SatChecker orbital data access.” [Archive API documentation](https://satchecker.readthedocs.io/en/latest/tools_data_access.html).

[12] I. Janveja, J. Zhang, E. Sie, and D. Vasisht, “StarLoc: Pinpointing transmitting LEO satellites from a single passive array,” 2026. [arXiv:2604.21147](https://arxiv.org/abs/2604.21147). [Dataset](https://zenodo.org/records/19043854). [Preprocessing code](https://github.com/ConnectedSystemsLab/starloc_preproc).

[13] StarlinkBeacon research project, experiment records and archived replays, v0.0.144, commit 1eabe0f, Nov. 2026. Companion reproducibility note and evidence map.
