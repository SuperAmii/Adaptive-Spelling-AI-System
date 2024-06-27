# Title: Adaptive Spelling AI System
# Author: Amitai Kamp
# Date: May/June 2024
# University of Amsterdam
# Study: Bachelor Kunstmatige Intelligentie (KI)
# Course: Afstudeerproject, Bachelor KI

This repository contains the code developed for my bachelor AI thesis. The thesis written in May/June 2024 answered the following research question: 
How can an adaptive spelling AI system be developed to generate Dutch spelling exercises tailored to the learning gain of primary school students?

1. The file dataset_creation.py contains the code that automatically assigns spelling complexity scores to each word in the dataset by OpenTaal. These complexity scores are based on reference frameworks by Cito, SLO and the CED group and additionally take loanwords and the number of syllables into account.

2. The created dataset, counting the occurences of each morphological feature and syntactic structure together with the total spelling complexity score, can be found in the files:
Final Dataset (A-L) and Final Dataset (L-Z). The dataset has been split up into two files, as it was too large to upload onto GitHub. When utilizing the dataset for the adaptive spelling ai system, both files should be merged together named: final_merge.csv.

4. The file adaptive_spelling_ai_system.py contains the developed adaptive spelling AI system. By employing a refined Elo rating system augmented by additional parameters, the Needleman-Wunsch alignment algorithm and a sophisticated mechanism for detecting and accounting for user input errors, a personalised adaptive spelling AI system is developed.
