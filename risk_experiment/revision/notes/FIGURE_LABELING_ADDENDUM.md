# Addendum to figure brief — panel labels & lettering (Nature Communications)

Apply to all six regenerated figures. This supersedes any earlier uppercase / oversized panel-letter instruction.

## Panel labels (a, b, c …)
- **Lower-case, bold, upright** (not italic). `a`, `b`, `c` — **not** `A`/`B`/`C`, **not** small caps.
- **Same type size as the other text in the figure** (NC rule — do *not* enlarge them; match the axis-label/tick size). Note this differs from flagship *Nature*, which uses 8 pt; NC wants them the same size as surrounding text.
- Existing figures use uppercase `A, B, C` → switch all to lower-case bold.

```python
# panel letter: lower-case bold, same size as axis labels (e.g. 8 pt)
ax.text(-0.15, 1.05, 'a', transform=ax.transAxes,
        fontsize=8, fontweight='bold', fontstyle='normal',
        va='bottom', ha='right')
```

## All other lettering (axis titles, condition/legend labels, tick text)
- **Lower-case, sentence case** — only the first letter capitalized: `Stake size`, `Proportion risky`, `Decoded numerosity` — **not** `Stake Size` and **not** `STAKE SIZE`.
- Sans-serif (Helvetica/Arial), same font across all figures; text ≥ 6 pt at final size.

## Consistency check
- The figure panel letters must match the legends, which use lower-case bold **a, b, c**.
- After regenerating, verify: every panel letter is lower-case bold at body-text size, and no axis/label text is Title Case or all-caps.
