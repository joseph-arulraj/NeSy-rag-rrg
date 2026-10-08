# RadReport "Rad Chest 2 Views" vs our template report (KG task 5) — PROPOSAL, not applied

Source: `kg/sources/Rad Chest 2 Views.html` (RSNA RadReport template). Ours: `nesy/report.py` (rules P1–P8) and
`kg/findings.yaml: pertinent_negatives`.

## What the template has that we don't
| Template element | Ours | Comment |
|---|---|---|
| Procedure information (views: two views, single AP, AP portable, single PA, …) | not stated | We know PA/AP from metadata and use one frontal image. |
| Clinical information | not stated | We have no indication. |
| Comparison | not stated | We do not use priors. |
| Findings split into **Lungs / Pleural spaces / Heart / Mediastinum / Osseous structures / Additional findings** | one FINDINGS paragraph, ordered present → possible → absent | Main layout difference. |
| Section-level normal sentences ("The lungs are clear.", "No pleural abnormalities", "The heart is normal in size.", "The mediastinal contours are normal.", "There are no osseous abnormalities.") | per-finding negatives only for 4–5 findings | |
| Severity grades (mild / moderate / marked edema and cardiomegaly; trace / small / moderate / large effusions) | none (our "possible small effusion" is a fixed phrase) | We predict no severity. |
| Pulmonary vascular congestion; tortuous / calcified aorta; degenerative spine changes; scoliosis | not modelled | Outside our 13 findings. |
| Device tip position ("PICC tip below the carina") | "Support devices are in place." | We do not localise tips. |
| Interval change ("No significant interval change") | none | No priors. |
| Cardiogenic attribution ("likely cardiogenic edema") | none | |

## What we state that the template doesn't
- Explicit certainty tiers: "Possible …" sentences for the possible band (the template has only plain statements and one differential sentence).
- Positive pneumothorax, lung nodule/mass, fracture and pleural thickening sentences (the template lists only "No pneumothorax" in the impression and "Other" elsewhere).
- Side and zone ("in the right lower zone") from the side head and region scores.
- "No finding meets the reporting threshold." when nothing is stated but the root is not absent.
- Pertinent negative "No focal consolidation." (the template expresses this as "The lungs are clear.").

## Proposed changes (for approval)
1. **Layout:** FINDINGS in the template's subsections, fixed order — Lungs (lung opacity, atelectasis, consolidation, pneumonia, edema, lung lesion); Pleural spaces (pleural effusion, pneumothorax, pleural other); Heart (cardiomegaly); Mediastinum (enlarged cardiomediastinum); Osseous structures (fracture); Lines and tubes (support devices). Within a subsection keep present → possible → absent. IMPRESSION unchanged (P7).
2. **Procedure line** from metadata: "Single frontal view (PA)." / "Single frontal view (AP)." No Clinical information or Comparison lines (we have neither; stating "None" would be wrong).
3. **Pertinent negatives:**
   - Replace "No focal consolidation." by the section sentence **"The lungs are clear."** only when every Lungs finding is in the absent band (stricter than now: one silent finding suppresses it).
   - Keep "No pneumothorax." and "No pleural effusion." (template: "No pleural abnormalities"); keep "Heart size is normal." (template: "The heart is normal in size.").
   - Add **"The mediastinal contours are normal."** when enlarged cardiomediastinum is absent.
   - Do **not** add "There are no osseous abnormalities.": fracture AUROC is 0.68 and only fractures are modelled, so the sentence would claim more than the model checks.
   - Keep the conditional "No pulmonary edema." when cardiomegaly or effusion is present.
4. **Not adopted** (no model output to support them): severity grades, vascular congestion, aortic and spine findings, device tip positions, interval change.
5. Consequence: new sentences need the Stage 8 comparator lexicon updated ("lungs are clear", "mediastinal contours are normal") and the template round-trip re-checked before use.

Status: needs your approval; then radiologist review of the wording.
