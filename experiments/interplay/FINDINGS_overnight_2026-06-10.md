# Overnight findings — cross-task OV patching + residual decomposition (2026-06-10)

Two jobs, both finished clean (0 skips).
- **run1** `crosstask_ov_patching.py` → `crosstask_ov_nonce_arithmetic_pooled` (54k rows). Confirms the headline.
- **run2** `mechanism_decomposition.py` → `mechanism_decomp` (156,600 rows, arithmetic, 15 prompts × 90 ordered task-pairs per condition, N≈1350/cell). The mechanism answer.

Readout (4-way exact match at the source's query): **acc_src** = source rule still wins; **acc_donor** = TRANSFER (donor's *rule* applied to the *source's* query input); **acc_leak** = donor's own cached answer string; **other** = broken.

---

## 1. Headline (run1, replicates earlier full-scale run)
Patch donor content into source at demo-output positions, all 90 arithmetic pairs:

| condition       | acc_src | acc_donor | other |
|-----------------|--------:|----------:|------:|
| full_resid      | 0.006   | **0.588** | 0.404 |
| all_heads (OV)  | 0.015   | 0.008     | 0.970 |
| pairing (OV)    | 0.174   | 0.012     | 0.810 |
| pairing_rand    | 0.868   | 0.002     | 0.130 |
| aggregation     | 0.918   | 0.000     | 0.082 |
| aggregation_rand| 0.836   | 0.002     | 0.161 |

Full residual transplants the donor task (~59%). Pairing-head OV writes do **not** transfer it — they only **break** the source (other=0.81) while the matched-random control stays inert (src=0.87). Aggregation OV is inert at output positions (acts at the query position). Nonce: even full_resid barely transfers (0.10) — output positions are largely causally irrelevant there (induction story).

**The paradox:** pairing heads are causally *necessary* (knockout breaks the task) but their OV writes are *not sufficient and not transplantable* — so where does the transplantable task code live?

---

## 2. Mechanism answer (run2)

### 2a. Component decomposition (overwrite the named component at every layer, output positions)

| component        | acc_src | acc_donor | other |
|------------------|--------:|----------:|------:|
| full_resid       | 0.014   | **0.599** | 0.385 |
| attn+mlp_all     | 0.014   | 0.599     | 0.386 |
| **mlp_all**      | 0.013   | **0.367** | 0.615 |
| **attn_all**     | 0.008   | **0.006** | 0.979 |

- **MLP writes carry the transplantable rule.** Patching *only* the MLP outputs transfers the donor task 0.37 (61% of the full-resid effect) — with no attention touched.
- **Attention writes carry ~zero standalone rule content** (0.006). All-attention patching merely breaks the source (other=0.98), exactly like the pairing-OV result above.
- **attn+mlp = full_resid** (0.599): the residual-stream-start / embedding contributes nothing extra. So the transplant is fully accounted for by {attn_out, mlp_out}.
- **Synergy = routing.** MLP alone 0.367; +attention → 0.599, a **+0.23** lift that attention produces *only in the presence of* the patched MLP content (attention alone = 0.006). Attention does not *store* the rule; it *routes/binds* the MLP-written content. **This is precisely the pairing-head role: necessary as a router, content-free, hence not transplantable alone.**

### 2b. Rule, not cached answer (key control)
`acc_leak` (donor's literal answer string) stays ~0.002–0.007 in **every** condition while `acc_donor` rises to 0.37–0.60. The MLPs are transferring the donor's **function applied to the source's input**, not the donor's memorized output token. The transplant is a genuine rule transplant.

### 2c. Where the content lives (single-layer resid_post patch, acc_donor)

```
L0–L4:  ~0.03   (dead — nothing transferable yet)
L5:      0.260  <-- switches ON
L5–L13:  0.25–0.31  (peak L11 0.313, L9 0.288, L10 0.281)
L14+:    <0.05  (collapses)
L23+:    ~0.00
```

The transferable rule **enters the residual at L5** and is readable through **~L13**, then is gone — by L14 the source has committed and a late single-layer patch neither transfers nor fully breaks it. Cumulative-prefix patching (`resid_cumul`) shows the same L5 kick (0→0.26) and accrues to the full 0.60 only when the whole L0–L25 prefix is brought along, i.e. the content is **distributed/accumulated across the L5–L13 band**, not localized to one layer (max single-MLP transfer is just 0.049 at L5). **L5 is the pivot:** it is also the layer where attention patching maximally disrupts the source (attn_layer L5: src 0.39) and where MLP patching first disrupts it (mlp_layer L5: src 0.29) — matching the L5 dip you saw in the within-prompt `interchange_pairing` experiment.

---

## 3. Resolution of the puzzle
> Why does patching the full residual at output positions transplant the task, but pairing/aggregation OV (or all-head OV) does not — even though those heads are causally necessary?

Because the transplantable **task code is written by MLPs** across the **L5–L13** band at the demo-output positions, and the **attention heads (incl. pairing heads) carry no standalone copy of the rule — they route/bind that MLP content**. Patching only the attention/OV swaps the *routing* without installing any *rule content*, so the source's computation is derailed (→ "other") but the donor rule is never planted. Full-residual patching works because it carries the MLP-written rule content (and attention adds the matching routing for the last +0.23). "Necessary but not sufficient" for the pairing heads = "they move the information, they don't hold it."

Caveats: transfer tops out at ~0.60 (38% still land in "other") — the mean-over-span intervention is coarse and one-directional; this is a partial transplant, not a clean swap. Effect is arithmetic-specific; nonce output positions don't carry it.

---

## 4. Natural next experiments (not yet run)
1. **Minimal sufficient MLP band:** patch MLPs over sliding windows (e.g. L5–L9, L5–L13, L9–L13) to find the smallest MLP set that reaches ~0.37. Tests whether the band is a unit.
2. **Attention read-side vs write-side:** is attention's +0.23 because heads *read* the patched MLP content (then their native W_O suffices) or because their *writes* must also be donor-consistent? Patch MLPs to donor but freeze attention to source vs. patch both — already partially answered (mlp_all=0.37 with source attention), so the test is: patch MLP-donor + attention-pattern-donor (just `hook_pattern`, not OV) → isolates routing-vs-write.
3. **Which MLP reads what:** at L5–L13, do the rule-carrying MLPs read from the demo-input or demo-output token? Path-patch MLP-in.
