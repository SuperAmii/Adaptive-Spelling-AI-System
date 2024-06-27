# Author: Amitai Kamp
# Date: May/June 2024
# University of Amsterdam
# Study: Bachelor Kunstmatige Intelligentie (KI)
# Course: Afstudeerproject Bachelor KI

import numpy as np
import pyphen
import requests
from bs4 import BeautifulSoup
from unidecode import unidecode
import csv

# Initialize Pyphen for Dutch
word_segmenter = pyphen.Pyphen(lang='nl')

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

    # Lettergrepen opdeler & Teller:
    def split_into_syllables(word):
        word = word.replace('-', '')
        word = word.replace(' ', '')
        syllables = word_segmenter.inserted(word).split('-')
        num_syllables = len(syllables)
        return syllables, num_syllables

    # To determine if a word is a leenwoord
    def update_rule_counts(word, total_score):
        leenwoord = scrape_and_search(word)
        if leenwoord != "geen_leenwoord":
            leenwoord_index = list(rules_scores.keys()).index(leenwoord)
            rule_counts[leenwoord_index] += 1
            total_score += rules_scores[leenwoord]  # Update total score with leenwoord's score
        else:
            rule_counts[-1] += 1  # Increment the count for "geen_leenwoord" rule
        return total_score

    word = word.strip().lower()
    normalized_word = unidecode(word)

    syllables, num_syllables = split_into_syllables(word)

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

def main():
    with open("basiswoorden-gekeurd.txt", "r") as file:
        words = file.readlines()

    with open("test_output_basiswoorden-gekeurd.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Word", "Syllables", *rules_scores.keys(), "Total Score"])

        counter = 0
        for word in words:
            word = word.strip()
            rule_counts, total_score, num_syllables = count_spelling_rules(word)
            writer.writerow([word, num_syllables, *rule_counts, total_score])
            counter += 1
            if counter % 500 == 0:
                print(f"{counter} words processed and added to the CSV file.")

if __name__ == "__main__":
    main()
