""" Create by Ken at 2021 Jan 02 """
import re
from nltk import sent_tokenize, word_tokenize

clause_re = re.compile(r'^\(\d+\)')


def remove_numbering(line):
    match_clause = clause_re.match(line)
    if match_clause:
        line = line[match_clause.span()[1] + 1:]
    return line


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

    terms = word_tokenize(s)
    for term in terms:
        term = term.lower()
        if term in text_to_seq_dict:
            seq.append(text_to_seq_dict[term])
        else:
            seq.append(0)

    return seq
