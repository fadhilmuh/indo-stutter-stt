"""Rule-based stutter text augmentation.

Refactored from the generation notebook so the steps are easy to follow.
Every word is selected with probability P_DISFLUENT. A selected word gets one of
three disfluencies:

    repetition    (0.5)  "kartu" -> "ka-ka-kartu"  or  "kartu kartu"
    prolongation  (0.3)  "bisa"  -> "bbbisa"       (first consonant tripled; if the word
                                                    starts with a vowel, the last consonant)
    interjection  (0.2)  "kartu" -> "emm kartu" / "kartu emm" / "kartu" + filler word
                         (never after the last word, so fillers stay between words)

This file is a reference implementation of the algorithm and is not meant to be run directly.
"""

import random
import re

P_DISFLUENT = 0.8                        # chance that a word is modified at all
P_REPETITION = 0.5                       # type probabilities, applied to modified words
P_PROLONGATION = 0.3
P_INTERJECTION = 0.2

P_SYLLABLE_REPEAT = 0.7                  # repetition: 70% syllable ("ka-ka-kartu"), 30% whole word
REPEAT_RANGE = (2, 3)                    # how many extra copies
PROLONG_CHARS = 3                        # a prolonged letter appears 3 times

CONSONANTS = "bcdfghjklmnpqrstvwxyz"

SHORT_INTERJ = ["e…", "eee…", "emm", "emmm", "hmm", "ehh", "uhhh"]
THINKING_INTERJ = ["apa ya…?", "anu…", "apa namanya…?!", "sebentar…", "ini…", "apa…?"]
TRIGGER_WORDS = ["karena", "jadi", "lalu", "terus", "kemudian",
                 "mau", "bisa", "harus", "akan", "yang"]


def repeat(word):
    """Repetition: syllable repetition ("ka-ka-kartu") or whole-word repetition ("kartu kartu")."""
    if random.random() < P_SYLLABLE_REPEAT:
        match = re.match(rf"^([{CONSONANTS}]{{1,2}}[aeiou])", word.lower())
        syllable = match.group(1) if match else word.lower()[0]
        n = random.randint(*REPEAT_RANGE)
        return "-".join([syllable] * n) + "-" + word
    return " ".join([word] * random.randint(*REPEAT_RANGE))


def prolong(word):
    """Prolongation: triple the first consonant; if the word starts with a vowel, the last consonant."""
    triple = lambda m: m.group(1) * PROLONG_CHARS
    if re.match(rf"^[{CONSONANTS}]", word.lower()):
        return re.sub(rf"^([{CONSONANTS}])", triple, word)
    if re.search(rf"[{CONSONANTS}]$", word.lower()):
        return re.sub(rf"([{CONSONANTS}])$", triple, word)
    return word


def choose_interjection(word, position):
    """Pick a filler that suits the context."""
    # start of the sentence: often a "thinking" filler
    if position == 0 and random.random() < 0.55:
        return random.choice(THINKING_INTERJ)
    # before discourse markers and modal verbs: mostly a short filler
    if word.lower() in TRIGGER_WORDS:
        if random.random() < 0.7:
            return random.choice(SHORT_INTERJ)
        return random.choice(THINKING_INTERJ)
    # default: 80% short filler, 20% thinking filler
    if random.random() < 0.8:
        return random.choice(SHORT_INTERJ)
    return random.choice(THINKING_INTERJ)


def interject(word, position):
    """Return the tokens that replace `word`, with a filler placed around it."""
    filler = choose_interjection(word, position)
    r = random.random()
    if r < 0.7:                          # 70%: before the word
        return [f"{filler} {word}"]
    if r < 0.9:                          # 20%: after the word
        return [f"{word} {filler}"]
    return [filler, word]                # 10%: as a separate filler word before it


def make_disfluent_text(text, seed=None, p_disfluent=P_DISFLUENT):
    """Return (stuttered_text, disfluency_types) for one fluent sentence."""
    if seed is not None:
        random.seed(seed)

    words = text.split()
    result = []
    types = set()

    for i, word in enumerate(words):
        choice = random.random()         # decides the disfluency type
        if random.random() >= p_disfluent:
            result.append(word)          # word left unchanged
            continue

        if choice < P_REPETITION:
            result.append(repeat(word))
            types.add("repetition")
        elif choice < P_REPETITION + P_PROLONGATION:
            result.append(prolong(word))
            types.add("prolongation")
        elif choice < P_REPETITION + P_PROLONGATION + P_INTERJECTION and i != len(words) - 1:
            result.extend(interject(word, i))
            types.add("interjection")
        else:
            result.append(word)          # interjection drawn for the last word: skipped

    return " ".join(result), sorted(types)