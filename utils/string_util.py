""" Create by Ken at 2021 Jan 03 """
import string
from nltk.tokenize import word_tokenize
import re

table = str.maketrans('', '', string.punctuation)
number_re = re.compile(r'^\d+([.,-]\d+)*$')
clause_re = re.compile(r'^\(\d+\)')


def remove_numbering(line):
    match_clause = clause_re.match(line)
    if match_clause:
        line = line[match_clause.span()[1] + 1:]
    return line


def pre_process_text(text):
    """
    Tokenize, convert to lower case, remove punctuations
    :type text: str
    :rtype: list[str]
    """
    tokens = word_tokenize(text)

    # convert to lower case
    tokens = [w.lower() for w in tokens]

    # remove punctuation from each word
    stripped = [w.translate(table) if not number_re.match(w) else w for w in tokens]

    words = [w for w in stripped if w != '']

    return words


def sentence_to_seq(s, text_to_seq_dict):
    """
    Convert text to int sequence
    :type text_to_seq_dict: dict
    :type s: str
    :rtype: list[int]
    """
    seq = []
    s = remove_numbering(s)
    if s == '':
        return None

    terms = pre_process_text(s)
    for term in terms:
        term = term.lower()
        if term in text_to_seq_dict:
            seq.append(text_to_seq_dict[term])
        else:
            seq.append(0)

    return seq
