# Scientific project rules

- Treat user data and historical experimental evidence as immutable. Never invent decoded bits, fields, CRC passes, or independent repetitions.
- Every commit must increment the patch component of `VERSION` exactly once: `0.0.N` becomes `0.0.(N+1)`. The initial imported baseline is `0.0.1`, starting from `0.0.0`.
- Every commit must include a substantive matching entry in `CHANGELOG.md`: Purpose, Changes, Scientific record, Validation, and Limitations. Use a commit subject beginning `v<version>:`.
- Verify the change before committing. Record exact checks and results. Never bypass commit hooks or use `--no-verify` to avoid the version/changelog policy.
- After a clone run `python tools/install_hooks.py`. Use `python tools/release.py` with substantive records to prepare the next patch; it deliberately does not stage or commit. Run unit tests and `python tools/verify_public.py --staged` before the commit, then create the matching annotated tag.
- Preserve published commits and version tags. Do not amend, rewrite, reset, or replace history without explicit user authorization. No remote push is implied.
- Record inputs by hashes, acquisition metadata when known, hypotheses, estimator conventions, random seeds, whole-group assignments, controls, results and uncertainty. Keep synthetic demonstrations separate from measured IQ.
- Randomize whole correlated groups for calibration/holdout comparisons. Do not relabel chronological splits or bootstrap replicates as independent experiments.
- Capture failures and negative findings. A reference-assisted match is not unknown-payload decoding. Retain erasures and unresolved phase/bit conventions.
- No new RF acquisition or transmission is authorized. External reference repositories are read-only.
- Keep root files limited to project governance and entry points. Place each experiment's scripts, results, figures and reports in their documented categories. New experiments require an identifier and an experiment record.
- Use Git LFS for binary IQ and numerical artifacts. Keep input provenance and experiment manifests in normal Git. Never remove evidence merely because a test fails.

# User visual standard

For this user's visual design and development work, use the spacelibre VI skill at `LOCAL_USER_DIRECTORY/.codex/skills/spacelibre-vi/SKILL.md` as the default. Apply its approved SVG logo, full palette roles, continuous orbital composition, typography and output verification. Explicit user instructions, named alternative brands and mandatory templates take precedence. Pure backend work and plain text formats do not require graphic alterations. Preserve original monochrome marks and do not invent new logo geometry. Keep the skill aligned with https://github.com/spacelibres/vi.
