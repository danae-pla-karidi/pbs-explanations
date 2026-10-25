# Study on explanations of movie recommendations

## Notes for building the online form (not shown to participants)

- `google_form_builder.gs` builds the whole Google Form. Its header lists the steps.
- Every question title carries a code, for example `(U1_ease)` or `[U1_A_ease]`. The analysis script finds the columns of the exported CSV by these codes, so keep them.
- If you build the form by hand: one section per user, in the order U1 to U9, with the list of recommended movies, the three images `<case>_A.png`, `<case>_B.png`, `<case>_C.png` captioned Explanation A, B, C, and the eight questions below. Mark all questions as required.
- Do not show `stimuli_key.csv` to participants.
- Export the answers as CSV to `user_study/responses.csv` and run `python -m scripts.user_study_analysis`.

## Welcome page

A recommender system suggested ten movies to a user. An explanation is a diagram that connects the user to the recommended movies through the movie knowledge graph. You will see nine users. For each user there are three explanations, A, B, and C, of the same ten recommendations.
For each explanation you rate two statements. Then you choose one explanation for each of two purposes. There are no right or wrong answers. The study takes about 15 to 20 minutes.

Participation is voluntary and you may stop at any time. The form collects no name, no email address, and no other personal data. The answers are used for research only and will be published in anonymized form together with the article. Contact: Danae Pla Karidi, danae@athenarc.gr. By continuing you agree to take part.

## How to read a diagram

- Each diagram is a tree that starts at the user. Read it from the blue square outward, left to right or top to bottom.
- Blue square: the user who received the recommendations.
- Orange box: a recommended movie. An explanation may leave out some of the ten recommended movies.
- White box: another movie.
- Gray ellipse: another viewer. A viewer keeps the same number in the three explanations of a user.
- Line between a person and a movie: the person rated the movie. A thicker line is a higher rating.
- Example: the chain User, Movie X, Viewer 3, Movie Y means that the user rated X, that Viewer 3 also rated X, and that Viewer 3 rated Y.
- An orange box linked directly to the user is a recommended movie that the user has also rated.

## Background

- `[role]` What describes you best? (undergraduate student, graduate student, researcher or faculty member, other)
- `[familiarity]` How familiar are you with recommender systems? (1 = not at all, 5 = very familiar)

## User 1 of 9

Recommended movies: Gandhi, Eyes Wide Shut, Paulie, Rushmore, My Fair Lady, The Full Monty, Blade Runner, Unforgiven, Lone Star, Entrapment.

Images: `U1_A.png` (Explanation A), `U1_B.png` (Explanation B), `U1_C.png` (Explanation C).

Rate each statement for each explanation (1 = strongly disagree, 2 = disagree, 3 = neutral, 4 = agree, 5 = strongly agree).

- `[U1_A_ease]` Explanation A: This explanation is easy to understand.
- `[U1_A_evidence]` Explanation A: This explanation provides a large amount of evidence for why the movies were recommended.
- `[U1_B_ease]` Explanation B: This explanation is easy to understand.
- `[U1_B_evidence]` Explanation B: This explanation provides a large amount of evidence for why the movies were recommended.
- `[U1_C_ease]` Explanation C: This explanation is easy to understand.
- `[U1_C_evidence]` Explanation C: This explanation provides a large amount of evidence for why the movies were recommended.

Choose one explanation (A, B, or C).

- `[U1_quick]` Which explanation would you choose to quickly understand why these movies were recommended?
- `[U1_detail]` Which explanation would you choose to analyze in detail why these movies were recommended?

## User 2 of 9

Recommended movies: The Grifters, Angel Heart, Animal House, Cop Land, Jurassic Park, As Good as It Gets, Election, Blood Simple, Back to the Future Part III, Back to the Future Part II.

Images: `U2_A.png` (Explanation A), `U2_B.png` (Explanation B), `U2_C.png` (Explanation C).

Rate each statement for each explanation (1 = strongly disagree, 2 = disagree, 3 = neutral, 4 = agree, 5 = strongly agree).

- `[U2_A_ease]` Explanation A: This explanation is easy to understand.
- `[U2_A_evidence]` Explanation A: This explanation provides a large amount of evidence for why the movies were recommended.
- `[U2_B_ease]` Explanation B: This explanation is easy to understand.
- `[U2_B_evidence]` Explanation B: This explanation provides a large amount of evidence for why the movies were recommended.
- `[U2_C_ease]` Explanation C: This explanation is easy to understand.
- `[U2_C_evidence]` Explanation C: This explanation provides a large amount of evidence for why the movies were recommended.

Choose one explanation (A, B, or C).

- `[U2_quick]` Which explanation would you choose to quickly understand why these movies were recommended?
- `[U2_detail]` Which explanation would you choose to analyze in detail why these movies were recommended?

## User 3 of 9

Recommended movies: Quiz Show, Flight of the Navigator, Wizard of Oz, Pleasantville, Mars Attacks!, Beauty and the Beast, Hamlet, Starman, Rain Man, Erin Brockovich.

Images: `U3_A.png` (Explanation A), `U3_B.png` (Explanation B), `U3_C.png` (Explanation C).

Rate each statement for each explanation (1 = strongly disagree, 2 = disagree, 3 = neutral, 4 = agree, 5 = strongly agree).

- `[U3_A_ease]` Explanation A: This explanation is easy to understand.
- `[U3_A_evidence]` Explanation A: This explanation provides a large amount of evidence for why the movies were recommended.
- `[U3_B_ease]` Explanation B: This explanation is easy to understand.
- `[U3_B_evidence]` Explanation B: This explanation provides a large amount of evidence for why the movies were recommended.
- `[U3_C_ease]` Explanation C: This explanation is easy to understand.
- `[U3_C_evidence]` Explanation C: This explanation provides a large amount of evidence for why the movies were recommended.

Choose one explanation (A, B, or C).

- `[U3_quick]` Which explanation would you choose to quickly understand why these movies were recommended?
- `[U3_detail]` Which explanation would you choose to analyze in detail why these movies were recommended?

## User 4 of 9

Recommended movies: Back to the Future, Butch Cassidy and the Sundance Kid, Godfather, Indiana Jones and the Last Crusade, Jaws, Being John Malkovich, Return of the Jedi, Raiders of the Lost Ark, Star Wars, Saving Private Ryan.

Images: `U4_A.png` (Explanation A), `U4_B.png` (Explanation B), `U4_C.png` (Explanation C).

Rate each statement for each explanation (1 = strongly disagree, 2 = disagree, 3 = neutral, 4 = agree, 5 = strongly agree).

- `[U4_A_ease]` Explanation A: This explanation is easy to understand.
- `[U4_A_evidence]` Explanation A: This explanation provides a large amount of evidence for why the movies were recommended.
- `[U4_B_ease]` Explanation B: This explanation is easy to understand.
- `[U4_B_evidence]` Explanation B: This explanation provides a large amount of evidence for why the movies were recommended.
- `[U4_C_ease]` Explanation C: This explanation is easy to understand.
- `[U4_C_evidence]` Explanation C: This explanation provides a large amount of evidence for why the movies were recommended.

Choose one explanation (A, B, or C).

- `[U4_quick]` Which explanation would you choose to quickly understand why these movies were recommended?
- `[U4_detail]` Which explanation would you choose to analyze in detail why these movies were recommended?

## User 5 of 9

Recommended movies: Saving Private Ryan, The Big Lebowski, Sliding Doors, Jurassic Park, The Sixth Sense, Toy Story 2, Back to the Future, The Governess, Return of the Jedi, The Silence of the Lambs.

Images: `U5_A.png` (Explanation A), `U5_B.png` (Explanation B), `U5_C.png` (Explanation C).

Rate each statement for each explanation (1 = strongly disagree, 2 = disagree, 3 = neutral, 4 = agree, 5 = strongly agree).

- `[U5_A_ease]` Explanation A: This explanation is easy to understand.
- `[U5_A_evidence]` Explanation A: This explanation provides a large amount of evidence for why the movies were recommended.
- `[U5_B_ease]` Explanation B: This explanation is easy to understand.
- `[U5_B_evidence]` Explanation B: This explanation provides a large amount of evidence for why the movies were recommended.
- `[U5_C_ease]` Explanation C: This explanation is easy to understand.
- `[U5_C_evidence]` Explanation C: This explanation provides a large amount of evidence for why the movies were recommended.

Choose one explanation (A, B, or C).

- `[U5_quick]` Which explanation would you choose to quickly understand why these movies were recommended?
- `[U5_detail]` Which explanation would you choose to analyze in detail why these movies were recommended?

## User 6 of 9

Recommended movies: Babe: Pig in the City, Total Recall, Saving Private Ryan, The Chamber, Passion Fish, In the Line of Fire, Groundhog Day, The General's Daughter, The Remains of the Day, I Dreamed of Africa.

Images: `U6_A.png` (Explanation A), `U6_B.png` (Explanation B), `U6_C.png` (Explanation C).

Rate each statement for each explanation (1 = strongly disagree, 2 = disagree, 3 = neutral, 4 = agree, 5 = strongly agree).

- `[U6_A_ease]` Explanation A: This explanation is easy to understand.
- `[U6_A_evidence]` Explanation A: This explanation provides a large amount of evidence for why the movies were recommended.
- `[U6_B_ease]` Explanation B: This explanation is easy to understand.
- `[U6_B_evidence]` Explanation B: This explanation provides a large amount of evidence for why the movies were recommended.
- `[U6_C_ease]` Explanation C: This explanation is easy to understand.
- `[U6_C_evidence]` Explanation C: This explanation provides a large amount of evidence for why the movies were recommended.

Choose one explanation (A, B, or C).

- `[U6_quick]` Which explanation would you choose to quickly understand why these movies were recommended?
- `[U6_detail]` Which explanation would you choose to analyze in detail why these movies were recommended?

## User 7 of 9

Recommended movies: Broadcast News, Summer of Sam, Deuce Bigalow: Male Gigolo, Fight Club, The Usual Suspects, Snake Eyes, American Psycho, Magnolia, St. Elmo's Fire, Dirty Dancing.

Images: `U7_A.png` (Explanation A), `U7_B.png` (Explanation B), `U7_C.png` (Explanation C).

Rate each statement for each explanation (1 = strongly disagree, 2 = disagree, 3 = neutral, 4 = agree, 5 = strongly agree).

- `[U7_A_ease]` Explanation A: This explanation is easy to understand.
- `[U7_A_evidence]` Explanation A: This explanation provides a large amount of evidence for why the movies were recommended.
- `[U7_B_ease]` Explanation B: This explanation is easy to understand.
- `[U7_B_evidence]` Explanation B: This explanation provides a large amount of evidence for why the movies were recommended.
- `[U7_C_ease]` Explanation C: This explanation is easy to understand.
- `[U7_C_evidence]` Explanation C: This explanation provides a large amount of evidence for why the movies were recommended.

Choose one explanation (A, B, or C).

- `[U7_quick]` Which explanation would you choose to quickly understand why these movies were recommended?
- `[U7_detail]` Which explanation would you choose to analyze in detail why these movies were recommended?

## User 8 of 9

Recommended movies: Dr. Dolittle, Bowfinger, The Silence of the Lambs, High Fidelity, Groundhog Day, Austin Powers: International Man of Mystery, Jurassic Park, Mystery, Alaska, 12 Monkeys, Aladdin (1992 Disney film).

Images: `U8_A.png` (Explanation A), `U8_B.png` (Explanation B), `U8_C.png` (Explanation C).

Rate each statement for each explanation (1 = strongly disagree, 2 = disagree, 3 = neutral, 4 = agree, 5 = strongly agree).

- `[U8_A_ease]` Explanation A: This explanation is easy to understand.
- `[U8_A_evidence]` Explanation A: This explanation provides a large amount of evidence for why the movies were recommended.
- `[U8_B_ease]` Explanation B: This explanation is easy to understand.
- `[U8_B_evidence]` Explanation B: This explanation provides a large amount of evidence for why the movies were recommended.
- `[U8_C_ease]` Explanation C: This explanation is easy to understand.
- `[U8_C_evidence]` Explanation C: This explanation provides a large amount of evidence for why the movies were recommended.

Choose one explanation (A, B, or C).

- `[U8_quick]` Which explanation would you choose to quickly understand why these movies were recommended?
- `[U8_detail]` Which explanation would you choose to analyze in detail why these movies were recommended?

## User 9 of 9

Recommended movies: Raiders of the Lost Ark, Lethal Weapon, Indiana Jones and the Last Crusade, Butch Cassidy and the Sundance Kid, Aliens, Star Wars, The Princess Bride, Romancing the Stone, Saving Private Ryan, Return of the Jedi.

Images: `U9_A.png` (Explanation A), `U9_B.png` (Explanation B), `U9_C.png` (Explanation C).

Rate each statement for each explanation (1 = strongly disagree, 2 = disagree, 3 = neutral, 4 = agree, 5 = strongly agree).

- `[U9_A_ease]` Explanation A: This explanation is easy to understand.
- `[U9_A_evidence]` Explanation A: This explanation provides a large amount of evidence for why the movies were recommended.
- `[U9_B_ease]` Explanation B: This explanation is easy to understand.
- `[U9_B_evidence]` Explanation B: This explanation provides a large amount of evidence for why the movies were recommended.
- `[U9_C_ease]` Explanation C: This explanation is easy to understand.
- `[U9_C_evidence]` Explanation C: This explanation provides a large amount of evidence for why the movies were recommended.

Choose one explanation (A, B, or C).

- `[U9_quick]` Which explanation would you choose to quickly understand why these movies were recommended?
- `[U9_detail]` Which explanation would you choose to analyze in detail why these movies were recommended?

## End

Thank you for taking part.