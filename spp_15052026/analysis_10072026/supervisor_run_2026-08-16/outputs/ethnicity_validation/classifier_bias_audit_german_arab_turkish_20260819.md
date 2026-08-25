# Classifier bias audit — German / Arab / Turkish

Targeted spot-checks of the name-based ethnicity predictor (`reviewer_ethnicity_lookup.parquet`,
923,168 reviewers), focused on the three groups used in Task B's German/non-German split. No
ground truth / manual coding required — every row below is a name that can be read by eye.
Two error directions checked per group: **false positive** (labelled as the group, isn't) and
**false negative** (is the group, labelled something else). Source: ad hoc queries, this session;
full reviewer-level source data in `reviewer_ethnicity_lookup.parquet`.

## Summary

| Group | Direction | Sample | Rate | Method | Example |
|---|---|---|---|---|---|
| German | false negative | 50 names (2 seeds × 25/30) from "United States" (n=124,521) | **50/50 (100%)** read as German | random sample, eyeballed | "Angela Schmidt" → predicted United States |
| German | false positive | 10 names with Turkish-exclusive characters (ş/ğ/ı/İ/-oğlu) predicted Germany | **2/10** have zero German signal at all; rest are mixed-heritage | targeted character search | "Şahin Önel" → predicted Germany |
| Arab | false positive | 30 names predicted an Arab League country | **8/30 (27%)** — 7 junk/handle names, 1 clearly non-Arab | random sample, eyeballed | "Slim Reaper" → predicted Algeria |
| Arab | false negative | targeted search (Al-/El- surname marker) | 2 clean examples found, incl. one self-identified Saudi predicted Canada | targeted marker search, high noise | "Rayed Al Tohifa (Königreich Saudi-Arabien)" → predicted Canada |
| Turkish | false positive | 25 names predicted Türkiye | **1/25 (4%)** is Arabic, not Turkish | random sample, eyeballed | "Mustafa Alyousef" → predicted Türkiye |
| Turkish | false negative | 1,017 names with Turkish-exclusive characters | **144/1,017 (14.2%)** not predicted Türkiye; most is noise/Azeri-Uzbek overlap, but a clear cluster of genuine misroutes exists | targeted character search | "İbrahim K." → predicted Egypt |

## German

### False negative — real German names not labelled German
"United States" is the predictor's 2nd-largest category (124,521 of ~1.27M rows) — 7x larger
than Türkiye. Two independent random samples (seed 42, n=30; seed 7, n=20) of "United
States"-predicted reviewers: **50/50 read as ordinary German first-name + surname combinations,
0 read as American.**

Examples: Angela Schmidt, Andreas Beutke, Silvia Bremhorst, Sonja Porske, Gorden Heuer,
Michael Hammel, Henny Hintelmann, Carsten Mareth, Kirsten Knecht, Wolfgang Diehnelt,
Daniela Samsen, Martin Schroda.

### False positive — predicted German but isn't
Targeted search for names containing Turkish-exclusive characters (ş, ğ, ı, İ) or the "-oğlu"
suffix that were nonetheless predicted "Germany" (n=10 found):

| Name | Signal |
|---|---|
| Şahin "Shako59" Önel | fully Turkish, **zero German signal** |
| Safder Dağdelen | fully Turkish, **zero German signal** |
| HATİCE WAGNER | Turkish first name + German surname |
| Bediş Mincheva | Turkish/Bulgarian first name, non-German surname |
| Katrin Şerban | German first name + Turkish/Romanian surname |
| Menekşe von der Haar | Turkish first name + German surname |
| Sabine Kräft-Töremiş | German first name + Turkish surname |
| Nurdoğan Muckel | Turkish first name + German surname |
| Menekşe Baruc | Turkish first name, non-German surname |
| İrem Wedekind | Turkish first name + German surname |

Most of these are plausibly mixed Turkish-German heritage (naturalised/married surname — the
mechanism the original brief anticipated). But **"Şahin Önel" and "Safder Dağdelen" have no
German element at all** and were still predicted German — that's not explainable by mixed
heritage, it's a population-prior override of a name with zero German signal.

Also found via a separate Arabic-surname search: **"Claudia Al-Obaidi"** (German first name +
Al-Obaidi, a known Iraqi Arab surname) → predicted Germany.

## Arab

### False positive — predicted an Arab League country but isn't
Random sample of 30 reviewers predicted an Arab League country (seed 42): **8/30 (27%)** are
false positives —
- 7 are junk/handle names given a specific, false-precision country label anyway: "Slim Reaper"→
  Algeria, "De RaWa"→Iraq, "Zer0"→Algeria, "Sissi"→Algeria, "Bu Lette"→Bahrain, "am bhk"→Tunisia,
  "El"→Morocco (a two-letter fragment).
- 1 is a clearly non-Arab real name: "Yvonne Heidak"→United Arab Emirates (Yvonne is French/Western).

This is a distinct failure mode from the population-prior one above: false *confidence* on names
carrying no real signal, rather than misdirection of a real name.

### False negative — real Arab names not labelled Arab
Targeted search (noisy — "Al"/"El" also matches Spanish/Italian articles and nicknames like "El
Presidente", "Al Bundy") surfaced two clean, unambiguous cases:
- **"Claudia Al-Obaidi"** → predicted Germany (Al-Obaidi = Iraqi Arab tribal surname).
- **"Rayed Al Tohifa (Königreich Saudi-Arabien)"** → predicted **Canada**, despite the reviewer
  writing "Kingdom of Saudi Arabia" in German in their own display name.

Plus, from the Turkish sample: **"Mustafa Alyousef"** → predicted Türkiye (Al-Yousef is an Arabic
surname pattern, not Turkish) — a real Arab name absorbed into the Turkish category.

## Turkish (Türkiye)

### False positive — predicted Turkish but isn't
Random sample of 25 reviewers predicted Türkiye (seed 42): **1/25 (4%)** — "Mustafa Alyousef"
(Arabic surname, not Turkish; also listed above as an Arab false negative — same error, two
group labels).

### False negative — real Turkish names not labelled Turkish
Targeted search for Turkish-exclusive characters (ş, ğ, ı, İ) or "-oğlu": 1,017 names carry this
signal. 873/1,017 (85.8%) are correctly predicted Türkiye. Of the 144 (14.2%) that aren't:
- Most are noise (decorative/garbled usernames) or genuinely non-Turkish names that share Turkic
  orthography (Azeri, Uzbek — e.g. "Şemşat Gurbanova"→Azerbaijan, "Aziza Raşidova"→Uzbekistan;
  arguably correct, not a bug).
- A real cluster of misroutes exists at the **Turkish/Arab boundary via shared Islamic first
  names**: "İbrahim \*\*\*\*\*"→Egypt, "İbrahim K."→Egypt, "İsmail"→Morocco, "ABDULLAH ALCEŞAM"→
  Saudi Arabia, "İrfan Yavuzdemir"→Pakistan. Ibrahim/Ismail/Abdullah are used across both Turkish
  and Arabic naming, and the classifier resolves the ambiguity toward Arab countries here.
- Two names with **no German signal at all** predicted Germany instead of Türkiye (see German
  false-positive section): "Şahin Önel", "Safder Dağdelen".

## Net read for Task B

All three groups leak into each other, but not symmetrically:
- **German is the largest net beneficiary** of misclassification — it gains ~124,521 "United
  States" predictions (essentially all real German names) and picks up some unambiguous Turkish
  names too. German's true volume is likely understated less than the others', if anything it's
  *overstated* by absorbing mislabelled non-Germans.
- **Non-German (Arab+Turkish+Other) is contaminated in two independent ways**: (1) it inherits
  the "United States" bucket, which is ~100% actually-German by this audit, and (2) the "Arab"
  sub-bucket specifically has a ~27% false-positive rate from junk names, inflating its apparent
  size with noise rather than real non-German signal.
- **Turkish and Arab leak into each other** via shared Islamic first names (Mustafa, Ibrahim,
  Ismail, Abdullah) in both directions — this doesn't bias the German/non-German split (both are
  "non-German" either way) but would matter for any analysis distinguishing Turkish from Arab
  specifically.

Combined with the earlier finding that ~19% of Task B's non-German racialised-review count is
specifically "United States"-predicted (i.e. likely German), this audit adds: even the residual
non-German count that isn't "United States" is inflated by a further ~27% junk-name rate within
the Arab sub-bucket. The true "false-positive-free" non-German count is very likely substantially
smaller than what Task B's split currently uses — consistent with, and reinforcing, the earlier
recommendation to treat B3's p=0.133→0.087 shift (US-as-German robustness check) as an
underestimate of how much the headline finding would move under a fuller correction.
