# Author: Amitai Kamp
# Date: May/June 2024
# University of Amsterdam
# Study: Bachelor Kunstmatige Intelligentie (KI)
# Course: Afstudeerproject Bachelor KI

# Importing the necessary libraries
import numpy as np
import pyphen
import requests
from bs4 import BeautifulSoup
from unidecode import unidecode
import csv
import pandas as pd
import math
import random
import playsound
from gtts import gTTS
import os
from IPython.display import Audio, display
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
import sys
import tempfile
import pygame
from bs4 import BeautifulSoup
from io import BytesIO
from pywebio.input import input, NUMBER, TEXT
from pywebio.output import put_text, put_markdown
from pywebio.input import *
from pywebio.output import *
from pywebio.session import *
from pywebio.platform.tornado import start_server
from pywebio.output import put_text, put_buttons, clear, toast

# Initialize Pyphen for Dutch, used to count the number of syllables in a word
word_segmenter = pyphen.Pyphen(lang='nl')

# The calculated scores for each spelling rule (morphological and syntactical features)
rules_scores = {'sch': 4125, 'c': 7375, 'ng': 4125, 'nk': 4125, 'isch': 7750, 'ische': 7750,'two_identical_consonants': 4750, 'diaeresis': 8125, 'grave_circumflex': 8250, 'acute_letters': 8250, 'iken': 8250}
for rule in ['f', 'v','s','z']:
    rules_scores[rule] = 4250
for rule in ['long_f', 'long_v']:
    rules_scores[rule] = 6125
for rule in ['long_s', 'long_z']:
    rules_scores[rule] = 6125
for rule in ['je','jes']:
    rules_scores[rule] = 4125
for rule in ['aai','ooi','oei']:
    rules_scores[rule] = 4375
for rule in ['eer','eur','oor']:
    rules_scores[rule] = 4500
for rule in ['be', 'ge', 'ver', 'te', 'el', 'er', 'en', 'te']:
    rules_scores[rule] = 4625
for rule in ['ij','ei']:
    rules_scores[rule] = 4625
for rule in ['long_ij','long_ei']:
    rules_scores[rule] = 6125
for rule in ['d', 'a','o','u']:
    rules_scores[rule] = 4875
for rule in ['dubbele_klinker']:
    rules_scores[rule] = 3500
# Also for -auw and -ouw
for rule in ['au','ou']:
    rules_scores[rule] = 4950
for rule in ['long_au','long_ou']:
    rules_scores[rule] = 5875
for rule in ['uw']:
    rules_scores[rule] = 5125
for rule in ['ch','cht']:
    rules_scores[rule] = 4950
for rule in ['x']:
    rules_scores[rule] = 8125
for rule in ['y']:
    rules_scores[rule] = 7750
for rule in ['th']:
    rules_scores[rule] = 7875
for rule in ['long_ng','long_nk']:
    rules_scores[rule] = 5875
# For -je(s), -(e)tje(s), -pje(s)
for rule in ['long_je','long_jes']:
    rules_scores[rule] = 6125
for rule in ['long_aai','long_oei','long_ooi']:
    rules_scores[rule] = 5375
for rule in ['long_eer','long_eur','long_oor']:
    rules_scores[rule] = 5375
for rule in ['long_be', 'long_ge', 'long_ver', 'long_te', 'long_el', 'long_er', 'long_te']:
    rules_scores[rule] = 5500
for rule in ['long_d']:
    rules_scores[rule] = 6000
for rule in ['long_a','long_o','long_u']:
    rules_scores[rule] = 5375
# For -auw, -ouw, -uw, -eeuw, -ieuw, -uw
for rule in ['long_uw']:
    rules_scores[rule] = 5875
for rule in ['ig','lijk']:
    rules_scores[rule] = 6625
for rule in ['ie']:
    rules_scores[rule] = 6625
for rule in ['long_ie']:
    rules_scores[rule] = 7750
for rule in ['tie']:
    rules_scores[rule] = 7375
for rule in ["'s"]:
    rules_scores[rule] = 7625
for rule in ['heid','teit']:
    rules_scores[rule] = 7125
for rule in ['long_ch']:
    rules_scores[rule] = 7250
for rule in ['b']:
    rules_scores[rule] = 7750
for rule in ['aatje', 'eetje', 'ootje', 'uutje']:
    rules_scores[rule] = 7750
for rule in ['nkje', 'ontd']:
    rules_scores[rule] = 7750
for rule in ['iaal', 'ieel', 'ueel', 'eaal']:
    rules_scores[rule] = 8250
for rule in ['long_en']:
    rules_scores[rule] = 6750
for rule in ['-']:
    rules_scores[rule] = 8500
for rule in ['accent']:
    rules_scores[rule] = 7625
for rule in ['ui','eu']:
    rules_scores[rule] = 3500
for rule in ['em','elen','enen','eren']:
    rules_scores[rule] = 5500
for rule in ['q']:
    rules_scores[rule] = 7500
for rule in ['g']:
    rules_scores[rule] = 6500
for rule in ['k']:
    rules_scores[rule] = 6875
for rule in ["leenwoord_duits","leenwoord_frans","leenwoord_engels","leenwoord_spaans"]:
    rules_scores[rule] = 8250
for rule in ["geen_leenwoord"]:
    rules_scores[rule] = 0

# Function to scrape etymology information (the origin language) of a given word.
# This function checks wheteher the given word is a loanword and if so, of what origin.
def scrape_and_search(word, retry_attempted=False):
    # Strip whitespaces and convert all letters to lowercase
    word = word.strip().lower()
    # Normalize the word to remove accents and special characters
    normalized_word = unidecode(word)

    url = f"https://etymologiebank.nl/trefwoord/{normalized_word}"
    response = requests.get(url)

    # Check if url page is found and find the content
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        odd_book_div = soup.find("div", class_="oddBook")

        if odd_book_div:
            word_content = odd_book_div.find("div", class_="wordContent")
            if word_content:
                # Set of languages to check for
                # Using a set instead of a list, which offers constant time complexity
                languages_to_check = {
                    "ontleend aan frans", "ontleend aan het frans",
                    "ontleend aan engels", "ontleend aan het engels",
                    "ontleend aan duits", "ontleend aan het duits",
                    "ontleend aan amerikaans-engels", "ontleend aan het amerikaans-engels",
                    "ontleend aan mexicaans-spaans", "ontleend aan het mexicaans-spaans",
                    "ontleend aan spaans", "ontleend aan het spaans",
                    "ontleend aan hoogduits", "ontleend aan het hoogduits",
                    "duits ontleend", "frans ontleend", "engels ontleend",
                    "Mexicaans-Spaans ontleend", "hoogduits ontleend",
                    "amerikaans-engels ontleend", "ontleend is aan duits",
                    "ontleend is aan engels", "ontleend is aan frans",
                    "ontleend is aan hoogduits", "ontleend is aan amerikaans-engels",
                    "ontleend is aan mexicaans-spaans", "ontleend aan amerikaans-spaans",
                    "ontleend is aan amerikaans-spaans","ontleend aan het amerikaans-spaans",
                    "ontleend aan amerikaans-spaans", "ontleend is aan catalaans", "ontleend aan catalaans",
                    "ontleend aan het catalaans","ontleend is aan het catalaans", "aan het frans ontleend",
                    "aan het engels ontleend", "aan het duits ontleend", "aan het amerikaans-engels ontleend",
                    "aan het mexicaans-spaans ontleend","aan het spaans ontleend", "aan het hoogduits ontleend",
                }

                # Dictionary to map language to result
                language_result_mapping = {
                    "duits": "leenwoord_duits",
                    "frans": "leenwoord_frans",
                    "engels": "leenwoord_engels",
                    "spaans": "leenwoord_spaans"
                }

                # Check for languages in the word content
                for language in languages_to_check:
                    if any(language in p.text.lower() for p in word_content.find_all("p")):
                        for lang, result in language_result_mapping.items():
                            if lang in language:
                                return result

                # If none of the languages are found in the word content div, search through the entire response text
                for language in languages_to_check:
                    if language in response.text.lower():
                        for lang, result in language_result_mapping.items():
                            if lang in language:
                                return result

                # If none of the languages are found in the response text
                return "geen_leenwoord"
            else:
                # Word wasn't found
                return "geen_leenwoord"
        elif not retry_attempted:
            # Retry to find the word with 1 attachted to it in the url
            # Checks with the boolean that this doesn't get into an inf. loop
            return scrape_and_search(word + "1", retry_attempted=True)
        else:
            return "geen_leenwoord"
    else:
        # If the page doesn't exist
        return "geen_leenwoord"


def count_spelling_rules(word):
    rule_counts = np.zeros(len(rules_scores), dtype=int)
    total_score = 0

    # Calculates the number of syllables in a word
    def split_into_syllables(word):
        word = word.replace('-', '')
        word = word.replace(' ', '')
        syllables = word_segmenter.inserted(word).split('-')
        num_syllables = len(syllables)
        return syllables, num_syllables

    # To determine if a word is a loanword
    def update_rule_counts(word, total_score):
        leenwoord = scrape_and_search(word)
        if leenwoord != "geen_leenwoord":
            leenwoord_index = list(rules_scores.keys()).index(leenwoord)
            rule_counts[leenwoord_index] += 1
            total_score += rules_scores[leenwoord]  # Update total score with loanwoord's score
        else:
            rule_counts[-1] += 1  # Increment the count for "geen_leenwoord" rule
        return total_score

    word = word.strip().lower()
    normalized_word = unidecode(word)

    syllables, num_syllables = split_into_syllables(normalized_word)

    # All additional constrains for counting rules are listed and checked here
    for rule_index, (rule, score) in enumerate(rules_scores.items()):
        count = 0

        # Check if sch has 1 syll and is a prefix
        if rule == 'sch':
            if word.startswith('sch') and num_syllables == 1:
                count = 1
            else:
                count = 0  # Set count to 0 if 'sch' condition not met
        # Check if 'isch', 'ische' is a prefix
        elif rule in ['isch', 'ische']:
            if word.endswith(rule) and num_syllables > 2:
                count = 1
            else:
                count = 0
        # For words with only 1 syllable
        elif rule in ['ng', 'nk', 'aai','ooi','oei']:
            if num_syllables == 1:
                count = word.count(rule)
            else:
                count = 0
        # For words with 2 syllables or less
        elif rule in ['f', 'v','s','z','eer','eur','oor']:
            if num_syllables <= 2:
                count = word.count(rule)
            else:
                count = 0

        # For words with 2 syllables or more
        elif rule in ['x','y','ig','lijk']:
            if num_syllables >= 2:
                count = word.count(rule)
            else:
                count = 0

        elif rule in ['be', 'ge', 'ver', 'te']:
            if num_syllables == 2 and word.startswith(rule):
                count = word.count(rule)
            else:
                count = 0

        elif rule in ['el', 'er', 'en', 'te']:
            if num_syllables <= 2 and word.endswith(rule):
                count = 1
            else:
                count = 0

        elif rule in ['je','jes']:
            if word.endswith(rule) and num_syllables == 2:
                count = 1
            else:
                count = 0

        elif rule in ['long_en']:
            if num_syllables > 2:
                count = word.count('en')
                # Subtract 1 from the count if 'iken' is found in the word
                if 'iken' in word:
                    count -= 1
                # Subtract 1 from the count if 'elen' is found in the word
                if 'elen' in word:
                    count -= 1
                if 'enen' in word:
                    count -= 1
                if 'eren' in word:
                    count -= 1
            else:
                count = 0

        elif rule in ['iken']:
            if word.endswith(rule):
                count = 1
            else:
                count = 0


        # Adding a different score to 'ij' and 'ei' depending on the number of syllables
        # Making sure that each element (ij and lijk) is counted only once to avoid double counting 
        # and preventing unreasonable elevation in the score.
        elif rule in ['ij', 'ei']:
            if 'lijk' not in word or num_syllables == 1:
                count = word.count(rule) if num_syllables <= 2 else 0
            else:
                parts = word.split('lijk')
                if num_syllables <= 2:  # Check if the number of syllables is at most two
                    count = sum(part.count(rule) for part in parts if part != '')
                else:
                    count = 0


        elif rule in ['long_ij', 'long_ei']:
            if num_syllables > 2:
                if 'lijk' not in word:
                    count = word.count('ij') if rule == 'long_ij' else word.count('ei')
                else:
                    parts = word.split('lijk')
                    count = sum(part.count('ij') if rule == 'long_ij' else part.count('ei') for part in parts)
            else:
                count = 0

        # For words ending on -d, -a, -o, -u, and on -eeuw, -ieuw, -uw
        elif rule in ['d','a','o','u', 'ch', 'cht']:
            if word.endswith(rule) and num_syllables <= 2:
                count = 1
            else:
                count = 0

        # Adding a different score to 'au' and 'ou' depending on the number of syllables
        elif rule in ['au','ou','uw']:
            if num_syllables <= 2:
                count = word.count(rule)
            else:
                count = 0
        elif rule in ['long_au','long_ou']:
            if num_syllables > 2:
                count = word.count('au') if rule == 'long_au' else word.count('ou')
            else:
                count = 0

        elif rule in ['long_uw']:
            if num_syllables > 2:
                count = word.count(rule.split('_')[1])
            else:
                count = 0

        elif rule in ['long_f','long_v']:
            if num_syllables > 2:
                count = word.count('f') if rule == 'long_f' else word.count('v')
            else:
                count = 0
        elif rule in ['long_s','long_z']:
            if num_syllables > 2:
                count = word.count('s') if rule == 'long_s' else word.count('z')
            else:
                count = 0
        elif rule in ['long_ng','long_nk']:
            if num_syllables > 2:
                count = word.count('ng') if rule == 'long_ng' else word.count('nk')
            else:
                count = 0
        # Check if word containing 'je' or 'jes' has more than 2 syllables and is the ending of a word.
        elif rule in ['long_je','long_jes']:
            if word.endswith('je') and num_syllables > 2:
                count = word.count('je') if rule == 'long_je' else 0
            elif word.endswith('jes') and num_syllables > 2:
                count = word.count('jes') if rule == 'long_jes' else 0
            else:
                count = 0

        elif rule in ['long_aai', 'long_ooi', 'long_oei']:
            if num_syllables > 2:
                count = word.count(rule.split('_')[1])
            else:
                count = 0

        elif rule in ['long_eer', 'long_eur', 'long_oor']:
            if num_syllables > 2:
                count = word.count(rule.split('_')[1])
            else:
                count = 0

        elif rule in ['long_be', 'long_ge', 'long_ver', 'long_te']:
            if num_syllables > 2 and word.startswith(rule.split('_')[1]):
                count = 1
            else:
                count = 0

        elif rule in ['long_el', 'long_er', 'long_te']:
            if num_syllables > 2 and word.endswith(rule.split('_')[1]):
                count = 1
            else:
                count = 0

        elif rule == 'long_d':
            if num_syllables > 2 and word.endswith(rule.split('_')[1]):
                count = 1
            else:
                count = 0

        elif rule == 'dubbele_klinker':
            # Define all possible double vowel combinations
            dubbele_klinker_combinations = ['aa', 'ee', 'ii', 'oo', 'uu']
            # Count occurrences of all double vowel combinations
            count = sum(word.count(combination) for combination in dubbele_klinker_combinations)

            # Check if any of the exceptions exist in the word
            exceptions = ['ooi','aai','eeuw', 'aatje', 'eetje', 'ootje', 'uutje', 'iaal', 'ieel', 'ueel', 'eaal']
            for exception in exceptions:
                if exception in word:
                    # Subtract 1 from the count for each occurrence of a double vowel combination that is part of an exception
                    count -= 1

        elif rule in ['long_a', 'long_o', 'long_u']:
            if num_syllables > 2 and word.endswith(rule.split('_')[1]):
                count = 1
            else:
                count = 0

        elif rule in ['ie']:
            if 'ie' in word:
                if 't' not in word[:word.index('ie')]:  # Check if 't' is not followed before 'ie'
                    count = word.count(rule) if num_syllables <= 2 else 0
                else:
                    parts = word.split('tie')
                    if num_syllables <= 2:  # Check if the number of syllables is at most two
                        count = sum(part.count(rule) for part in parts if part != '')
                    else:
                        count = 0
            else:
                count = 0

        elif rule in ['long_ie']:
            if 'tie' not in word:
                count = word.count(rule.split('_')[1]) if num_syllables > 2 else 0
            else:
                parts = word.split('tie')
                count = sum(part.count(rule.split('_')[1]) for part in parts if part != '')

        elif rule in ['tie']:
            if num_syllables > 2:
                count = word.count(rule)
            else:
                count = 0

        elif rule in ['long_ch']:
            if num_syllables > 2:
                count = 0
                count = word.count(rule.split('_')[1])
                if word.endswith(('isch', 'ische')):
                    count -= 1
            else:
                count = 0

        elif rule == 'b':
            count = 0
            if not word.startswith('b'):
                count = word.count('b')

        # Count occurence of letter g
        # Substract 'ig' and 'ge' from count
        elif rule in ['g']:
            count = 0
            count = word.count(rule)
            if num_syllables >= 2:
                count_ig = word.count('ig')
                if count_ig > 0:
                    count -= count_ig
            if word.startswith('ge'):
                count -= 1

        # Count occurence of letter k
        elif rule in ['k']:
            count = 0
            count_extra_k = 0
            count = word.count(rule)
            if 'nkje' in word:
                count_extra_k += word.count('nkje')
                if num_syllables > 2 and 'nk' in word:
                    # Otherwise nkje is counted twice
                    count_extra_k += word.count('nk') - 1
            if num_syllables == 1 and 'nk' in word:
                count_extra_k += word.count('nk')
            if num_syllables >= 2 and 'lijk' in word:
                count_extra_k += word.count('lijk')
            if num_syllables > 2 and 'nk' in word and 'nkje' not in word:
                count_extra_k += word.count('nk')
            if word.endswith('iken'):
                count_extra_k += 1

            count -= count_extra_k

        elif rule in ['aatje', 'eetje', 'ootje', 'uutje']:
            if num_syllables > 2 and word.endswith(rule):
                count = 1
            else:
                count = 0

        # Check if the word containing 'c' has more than 2 syllables and is not a combination with 'ch'
        elif rule == 'c':
            if num_syllables > 2:
                count = 0  # Initialize count to 0
                for i in range(len(word) - 1):
                    if word[i] == 'c' and word[i + 1] != 'h':  # Check if 'c' is not followed by 'h'
                        count += 1
            else:
                count = 0
        # Check for 2 adjacent identical consonants
        elif rule == 'two_identical_consonants':
            for i in range(len(word) - 1):
                if word[i].isalpha() and word[i+1].isalpha():
                    if word[i] not in 'aeiou' and word[i] == word[i+1]:  # Check if both characters are identical consonants
                        count += 1

        elif rule == 'diaeresis':
            # Check if it contains a diaeresis
            diaeresis_letters = ['ä','ë', 'ï', 'ü', 'ÿ', 'ö']
            count = sum(word.count(letter) for letter in diaeresis_letters)

        elif rule == 'acute_letters':
            # Check if it contains one of the acute letters
            acute_letters = ['á', 'é', 'í', 'ó', 'ú','ý']
            count = sum(word.count(letter) for letter in acute_letters)

        elif rule == 'accent':
            # Check if it contains an accent
            count = word.count("'")
            if "'s" in word:
                count -= 1

        elif rule == 'grave_circumflex':
            # Check if it contains a grave or circumflex
            grave_circumflex_letters = ['è','ò','ì','à','ỳ','ê','â','î','ô','û','ŷ']
            count = sum(word.count(letter) for letter in grave_circumflex_letters)

        else:
            count = word.count(rule)

        rule_counts[rule_index] = count
        total_score += count * score
    
    # The formula that calculates the total score of a word 
    # with an additional penalty based on the number of syllables
    # proportional to the total word score.
    total_score += (num_syllables - 1) * 0.1 * total_score

    # Call scrape_and_search function here
    scrape_and_search(normalized_word)
    # Update total score
    total_score = update_rule_counts(normalized_word, total_score)

    # Adding score based on conditions if total score is still 0
    if total_score == 0:
            total_score += 1

    return rule_counts, total_score, num_syllables

# Aligning the given word with the word inputted by the user.
def needleman_wunsch_alignment(word1, word2, rule):
    # A character-based alignment
    def align_words(word1, word2):
        # Init. empty string
        alignment_word1 = ""
        alignment_word2 = ""
        differences = {}

        i = 0
        while i < len(word1) and i < len(word2):
            # Checks if character at current index in both words are the same
            if word1[i] == word2[i]:
                alignment_word1 += word1[i]
                alignment_word2 += word2[i]
            # If characters differ a dash is added
            else:
                alignment_word1 += '-'
                alignment_word2 += '-'
                differences[word1[i]] = differences.get(word1[i], 0) - 1
                differences[word2[i]] = differences.get(word2[i], 0) + 1
            i += 1
        # After comparing characters up to the length of the shorter word
        # append remaining characters from either word
        alignment_word1 += word1[i:]
        alignment_word2 += word2[i:]

        return alignment_word1, alignment_word2, differences

    # Set up scoring scheme with penalties
    match = 2
    mismatch = -1
    gap_penalty = -1

    # Initialize scoring matrix
    matrix = np.zeros((len(word1) + 1, len(word2) + 1))

    # Initialize traceback matrix
    traceback = np.zeros((len(word1) + 1, len(word2) + 1))

    # Initialize the first row and column with gap penalties
    for i in range(1, len(word1) + 1):
        matrix[i][0] = i * gap_penalty
        # Indicate deletion in word2
        traceback[i][0] = 1
    for j in range(1, len(word2) + 1):
        matrix[0][j] = j * gap_penalty
        # Indicate deletion in word1
        traceback[0][j] = 2

    # Fill in the scoring and traceback matrices
    for i in range(1, len(word1) + 1):
        for j in range(1, len(word2) + 1):
            match_score = match if word1[i - 1] == word2[j - 1] else mismatch
            scores = [matrix[i - 1][j - 1] + match_score, # Diagonal movement (match/mismatch)
                      matrix[i - 1][j] + gap_penalty, # Up movement (insertion in word1)
                      matrix[i][j - 1] + gap_penalty]  # Left movement (insertion in word2)
            matrix[i][j] = max(scores)
            traceback[i][j] = scores.index(matrix[i][j])

    # Trace back to find the alignment
    alignment_word1 = ""
    alignment_word2 = ""
    i = len(word1)
    j = len(word2)
    while i > 0 or j > 0:
        # Diagonal move
        if traceback[i][j] == 0:
            alignment_word1 = word1[i - 1] + alignment_word1
            alignment_word2 = word2[j - 1] + alignment_word2
            i -= 1
            j -= 1
        # Up move
        elif traceback[i][j] == 1:
            alignment_word1 = word1[i - 1] + alignment_word1
            alignment_word2 = "-" + alignment_word2
            i -= 1
        # Left move
        else:
            alignment_word1 = "-" + alignment_word1
            alignment_word2 = word2[j - 1] + alignment_word2
            j -= 1

    return alignment_word1, alignment_word2, align_words(word1, word2)[2]

def compare_rule_positions(word1, word2):
    # Ensure both words are lowercase
    word1 = word1.lower()
    word2 = word2.lower()

    # Initialize a dictionary to store rule positions for each word
    rule_positions = {}

    # Count spelling rules for both words
    rule_counts_word1, _, _ = count_spelling_rules(word1)
    rule_counts_word2, _, _ = count_spelling_rules(word2)

    # Calculate all rules for each word
    rules_word1 = {rule: count for rule, count in zip(rules_scores.keys(), rule_counts_word1) if count > 0}
    rules_word2 = {rule: count for rule, count in zip(rules_scores.keys(), rule_counts_word2) if count > 0}

    # Determine the common rules between the two words
    common_rules = rules_word1.keys() & rules_word2.keys()

    # Iterate through each common rule and compare positions between words
    for rule in common_rules:
        # Get alignments for the rule in both words
        alignment_word1, alignment_word2, _ = needleman_wunsch_alignment(word1, word2, rule)

        # Initialize counts for the rule in both words
        count_in_word1 = alignment_word1.count(rule)
        count_in_word2 = alignment_word2.count(rule)

        # Calculate the score based on the differences in counts
        score = count_in_word1 - count_in_word2

        # Update the score for the rule
        rule_positions[rule] = score

    # Handle cases where one word contains the rule and the other does not
    for rule, count in rules_word1.items():
        if rule not in common_rules:
            rule_positions[rule] = count

    for rule, count in rules_word2.items():
        if rule not in common_rules:
            rule_positions[rule] = -count

    return rule_positions

###
def elo_score(successful_answered_questions, total_questions, correct_amount_spelling_rules, total_spelling_rules, user_rating, previous_word_score):
    # To resolve any errors with wordscores of 1
    if total_questions == 0 or total_spelling_rules == 0:
        return 1.0     
    # Calculate the win component 'W' for the Elo formula:
    W = (successful_answered_questions + total_questions) * correct_amount_spelling_rules / (2 * total_questions * total_spelling_rules)
    # Calculate the expected score 'E' using the Elo expected score formula:
    E = 1 / (1 + 10**((user_rating - previous_word_score) / 400))
    K = 10000 / (1.05 ** total_questions)
    # Calculate new user rating
    new_user_rating = user_rating + K * (W - E)
    if new_user_rating < 1:
        new_user_rating = 1    

    return new_user_rating

def get_random_word_with_score(user_rating, successful_answered_questions, total_questions, selected_words, previous_word_score, correct_amount_spelling_rules, total_spelling_rules, last_mistake):
    df = pd.read_csv('final_merge.csv', delimiter=',')

    # Define the user rating for the first word to a predefined avg. score of all words
    if user_rating is None:
        average_score = 34069.91559420409
        user_rating = average_score
    else:
        # Use the Elo calculation algorithm to calculate the user rating
        user_rating = elo_score(successful_answered_questions, total_questions, correct_amount_spelling_rules, total_spelling_rules, user_rating, previous_word_score)

    # Determine the range of acceptable word scores based on 1% of the current user rating
    margin = user_rating * 0.01
    lower_bound = user_rating - margin
    upper_bound = user_rating + margin

    # Filter the DataFrame for words within the acceptable score range
    filtered_df = df[(df['Total Score'] >= lower_bound) & (df['Total Score'] <= upper_bound)]
    filtered_df = filtered_df.sample(frac=1)  # Shuffle

    # 80% chance to consider last mistakes.
    #This approach reduces predictability and helps avoid repetitive cycles of the same spelling mistakes, 
    # while simultaneously maintaining a balance to account for previous spelling mistakes.
    if last_mistake and random.random() > 0.2:
          # Identify columns that correspond to spelling rules with mistakes made in last spelled word
          mistake_columns = [rule for rule in last_mistake if rule in df.columns and last_mistake[rule] > 0]

          # Calculate weights for each word based on how many times spelling rules were mistaken
          if mistake_columns:
              # Calculate weight by summing the applicable rule columns multiplied by mistake frequency
              filtered_df['weight'] = filtered_df[mistake_columns].apply(lambda row: sum(row[col] * last_mistake[col] for col in mistake_columns if col in row), axis=1)
              # Consider only those words which include a positive weight
              filtered_df = filtered_df[filtered_df['weight'] > 0]

    # Select a word either randomly or based on weights if they've been calculated
    if not filtered_df.empty:
        # If weights are calculated and valid, use them for weighted random selection
        if 'weight' in filtered_df.columns and filtered_df['weight'].sum() > 0:
            chosen_row = filtered_df.sample(weights='weight', n=1).iloc[0]
        else:
            # Otherwise, select randomly from the filtered DataFrame
            chosen_row = filtered_df.sample(n=1).iloc[0]
        word = chosen_row['Word']
        if word not in selected_words:
            # Add word to selected words set to avoid repetition
            selected_words.add(word)
            score = chosen_row['Total Score']
            spelling_rules_columns = [col for col in df.columns if col not in ['Word', 'Syllables', 'Total Score', 'geen_leenwoord']]
            total_spelling_rules = chosen_row[spelling_rules_columns].sum()
            return word, score, total_spelling_rules, user_rating

    # If no words meet the criteria, return None for all values
    put_text("No suitable word found within the criteria.")
    return None, None, None, user_rating

def print_rule_differences(word1, word2, last_mistake=None, mistakes=None, total_spelling_rules=None, successful_answered_questions=0, total_questions=0):
    last_mistake = {}  # Initialize last_mistake that stores the mistakes made in the last word
    if mistakes is None:
        mistakes = {}  # Initialize mistakes that stores all the mistakes made across all words

    # Get rule positions for both words
    rule_positions = compare_rule_positions(word1, word2)
    # Initialize correct_amount_spelling_rules
    correct_amount_spelling_rules = total_spelling_rules if total_spelling_rules is not None else None

    # Align the words
    alignment_word1, alignment_word2, _ = needleman_wunsch_alignment(word1, word2, "")

    printed = False
    for rule, score in rule_positions.items():
        if score != 0:
            # Calculate correct_amount_spelling_rules
            if correct_amount_spelling_rules is not None:
                correct_amount_spelling_rules -= 1 if score > 0 else 0
            printed = True
            # Store the absolute value of score in last_mistake
            last_mistake[rule] = abs(score)

    if not printed:
        put_text("No differences found.")

    # Add last_mistake to mistakes
    if last_mistake:
        # Update mistakes with last_mistake
        for rule, count in last_mistake.items():
            if count > 0:
                mistakes[rule] = mistakes.get(rule, 0) + count

    # Print correct amount of spelling rules
    return last_mistake, mistakes, correct_amount_spelling_rules, total_spelling_rules, successful_answered_questions, total_questions, not printed  # Return last_mistake, mistakes, and whether the answer was correct

# This function implements the text-to-speech functionality using the gTTS library
def speak(text, lang='nl'):
    tts = gTTS(text=text, lang=lang, slow=False)
    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as temp_file:
        tts.save(temp_file.name)
        pygame.mixer.init()
        pygame.mixer.music.load(temp_file.name)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            continue

# This function web scrapes the Van Dale site to receive a definition of a given word
def get_sentences_with_meanings(word):
    # Construct the URL
    url = f"https://www.vandale.nl/gratis-woordenboek/nederlands/betekenis/{word}"
    response = requests.get(url)

    # Check if the website is reached successfully
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the parent span of the correct class name
        parent_spans = soup.find_all('span', class_='f3 f0g')
        meanings_found = False
        meanings = []

        # Iterate over each parent span
        for idx, parent_span in enumerate(parent_spans, start=1):
            sentences_elem = parent_span.find_all('span', class_='fr')

            # Concatenate all words to list multiple definitions if applicable
            if sentences_elem:
                combined_sentence = ''

                for sentence_elem in sentences_elem:
                    sentence = sentence_elem.get_text(strip=True)
                    end_index = min(sentence.find(';'), sentence.find(':'))
                    if end_index != -1:
                        sentence = sentence[:end_index]
                    combined_sentence += sentence.strip() + ' '

                    if ':' in sentence or ';' in sentence:
                        combined_sentence = combined_sentence.strip()
                        break

                combined_sentence = combined_sentence.rstrip(':;')
                meanings.append(combined_sentence)
                meanings_found = True

        if meanings_found:
            return meanings
        else:
            return ["Geen definitie beschikbaar"]
    else:
        return ["Error accessing the website."]

# Initialize mistakes and counts outside the loop
def main():
    mistakes = {}
    selected_words = set()
    total_questions = 0
    successful_answered_questions = 0
    #max_questions = 30
    user_rating = None
    previous_word_score = None
    correct_amount_spelling_rules = 0
    total_spelling_rules = 0
    last_mistake = {}

    # Prompt the user to enter the number of questions they want to practice with
    while True:
        try:
            max_questions = int(input("Voer het aantal vragen in waarmee u wilt oefenen: ",type=NUMBER))
            if max_questions > 0:  # Check if the number is positive
                break
            else:
                put_text("Voer a.u.b. een positief getal in.")
        except ValueError:
            put_text("Ongeldige invoer. Voer alstublieft een geldig nummer in.")

    # Loop until the maximum number of questions is reached
    while total_questions < max_questions:

        put_text(f"Vraag {total_questions + 1} van {max_questions}:") 

        # Retrieve a word and its associated data
        word1, word1_score, total_spelling_rules, user_rating = get_random_word_with_score(user_rating, successful_answered_questions, total_questions, selected_words, previous_word_score, correct_amount_spelling_rules, total_spelling_rules, last_mistake)
        # If no valid word is found, break the loop
        if not word1:
            put_text("No more unique words available or fit the criteria.")
            break

        put_text("Het te spellen woord:")
        speak(f"{word1}", lang='nl')

        # Callback function for relisten button
        def relisten_callback(btn_val):
            if btn_val == 'Nogmaals luisteren':           
                speak(f"{word1}", lang='nl')
        
        # Add a button to relisten to the given word
        put_buttons(['Nogmaals luisteren'], onclick=relisten_callback)
        
        meaning_button_clicked = False
        
        # Callback function for meaning button
        def meaning_callback(btn_val):
            nonlocal meaning_button_clicked
            if btn_val == 'Ontvang definitie' and not meaning_button_clicked:  # Check if the button is not clicked yet
                meanings = get_sentences_with_meanings(word1)
                put_text('Definitie:')
                for meaning in meanings:
                    put_text(meaning)
                meaning_button_clicked = True 
        # Add a button to get the meaning of word1
        put_buttons(['Ontvang definitie'], onclick=meaning_callback)
        
        word2 = input("Type het woord: ", type=TEXT).strip()

        # Call the function to compare the two words and handle any differences.
        last_mistake, mistakes, correct_amount_spelling_rules, total_spelling_rules, _, _, _ = print_rule_differences(word1, word2, mistakes=mistakes, total_spelling_rules=total_spelling_rules,last_mistake=last_mistake)

        # Determine if the user's input matches the presented word exactly.
        normalized_word2 = word2.replace(" ", "").lower()
        answered_correctly = (word1 == normalized_word2)
        clear()
        if answered_correctly:
            successful_answered_questions += 1
            put_text("Correct!")
        else:
            put_text("Incorrect!")
            put_text("Uw antwoord:", word2)
            put_text("Het Correcte Antwoord:", word1)    
        
        # Callback function for the "Next" button
        next_button_clicked = False
        def next_callback(btn_val):
            nonlocal next_button_clicked
            next_button_clicked = True
        put_buttons(['Volgende Vraag'], onclick=next_callback)
        
        # Wait until the "Next" button is clicked to clear the output and proceed
        while not next_button_clicked:
            pass
        
        clear()  # Clear the output after the "Next" button is clicked
        # Update the user rating for the next cycle
        previous_word_score = word1_score
        total_questions += 1
        put_text("Score:",int(user_rating))
        put_text("Aantal beantwoorde vragen:", total_questions)
        put_text("Percentage goed beantwoorde vragen:", round(successful_answered_questions / total_questions * 100, 1), "%")

    if not mistakes:
        put_text("Goed gedaan! Er zijn geen spellingfouten om weer te geven.")
        return

    # Sorting mistakes by the number of occurrences and displaying them in a bar chart
    sorted_mistakes = sorted(mistakes.items(), key=lambda x: x[1], reverse=True)
    rules, counts = zip(*sorted_mistakes)

    # Create a color map from red to yellow, ensuring the most mistakes are dark red
    cmap = plt.get_cmap('autumn_r')  # Use reversed autumn color map
    norm = plt.Normalize(min(counts), max(counts))
    colors = cmap(norm(counts))

    plt.figure(figsize=(10, 5))
    bars = plt.bar(rules, counts, color=colors)
    plt.xlabel('Spellingregels')
    plt.ylabel('Aantal fouten')
    plt.title('Frequentie van spelfouten')
    plt.xticks(rotation=45, ha="right")
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))  # Ensuring y-axis ticks are integers

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

