# User study

Materials and results of the user study of Section 5.2 (ML1M, user-centric scenario).

## Cases

`scripts/user_study_stimuli.py` drew nine users at random, three from each of three
size strata of the WPCST(degree) summary (1 to 20, 21 to 35, and 36 to 50 edges), and
drew their ST, PCST, and WPCST(degree) summaries as tree diagrams. The study uses six
of them, two per stratum, drawn with `random.Random(42).sample` within each stratum:
U1 and U5 (36 to 50 edges), U3 and U4 (21 to 35), U6 and U9 (1 to 20). They are shown
in the order U1, U3, U4, U5, U6, U9 as "User 1 of 6" to "User 6 of 6".

| File | Content |
|---|---|
| `cases.csv` | the six users: recommender, user id, stratum, recommended movies |
| `stimuli_key.csv` | per case and label (A, B, C): algorithm, size, C, R, terminal coverage. Not shown to participants. |
| `stimuli/` | the 18 diagrams, `<case>_<label>.png` |
| `questionnaire.md` | the questionnaire as shown to participants |

## Procedure

Each participant saw, per user, the list of recommended movies and the three diagrams
without algorithm names, rated each diagram on a 5-point scale for ease of understanding
and for the amount of evidence it provides, and chose one diagram for quick understanding
and one for detailed analysis. Role and familiarity with recommender systems were asked
once. No personal data was collected.

## Analysis

```
python -m scripts.user_study_analysis --responses reported/user_study/responses.csv \
    --key reported/user_study/stimuli_key.csv --out reported/user_study
```

| File | Content |
|---|---|
| `analysis_ratings.csv` | mean, SD, and median of the participant means per algorithm and measure |
| `analysis_tests.csv` | paired Wilcoxon signed-rank tests on participant means, Holm-adjusted |
| `analysis_choices.csv` | choice shares per question and algorithm with bootstrap CIs |
| `analysis_proxies.csv` | Spearman correlation of C with the ease ratings and of R with the evidence ratings over the 18 summaries |
| `analysis_per_summary.csv` | mean ratings per summary next to its C, R, and size |
| `analysis_by_stratum.csv` | mean ratings and choice shares per size stratum |

## Responses
The answers of the participants are available in `responses.csv`.
