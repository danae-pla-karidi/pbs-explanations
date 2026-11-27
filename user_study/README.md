# User study

Materials and results of the user study of Section 5.2 (ML1M, user-centric scenario).

## Cases

The study uses six ML1M users, two from each of three size strata of the WPCST(degree)
summary, drawn at random within each stratum: U1 and U4 (36 to 50 edges), U2 and U3
(21 to 35), U5 and U6 (1 to 20). `scripts/user_study_stimuli.py` draws their ST, PCST,
and WPCST(degree) summaries as tree diagrams. The users are shown in the order
U1 to U6 as "User 1 of 6" to "User 6 of 6".

| File | Content |
|---|---|
| `cases.csv` | the six users: recommender, user id, stratum, recommended movies |
| `example_users.csv` | per case and label (A, B, C): algorithm, size, C, R, terminal coverage. Not shown to participants. |
| `example_user_trees/` | the 18 diagrams, `<case>_<label>.png` |
| `questionnaire.md` | the questionnaire as shown to participants |

## Procedure

Each participant saw, per user, the list of recommended movies and the three diagrams
without algorithm names, rated each diagram on a 5-point scale for ease of understanding
and for the amount of evidence it provides, and chose one diagram for quick understanding
and one for detailed analysis. Role and familiarity with recommender systems were asked
once. No personal data was collected.

The answers of the participants are in `users_responses.csv`: role, familiarity, and then one block
of eight columns per user in the order U1 to U6 (ease A, B, C; evidence A, B, C;
quick choice; detail choice). `python -m scripts.user_study_analysis` reads this file together with
`example_users.csv` and reproduces the numbers of Section 5.2.
