""" Create by Ken at 2021 Jan 03 """
import string
import re
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")

table = str.maketrans('', '', string.punctuation)
number_re = re.compile(r'^\d+([.,:-]\d+)*|^<number>$')
clause_re = re.compile(r'^\(\d+\)')
point_re = re.compile(r'^\([mdclxvi]+\)')


def remove_numbering(line):
    match_clause = clause_re.match(line)
    if match_clause:
        line = line[match_clause.span()[1] + 1:]

    match_point = point_re.match(line)
    if match_point:
        line = line[match_point.span()[1] + 1:]

    return line


def sentence_to_seq(s):
    """
    Convert text to int sequence
    :type s: str
    :rtype: list[int]
    """
    s = remove_numbering(s)
    if s == '':
        return None

    seq = tokenizer.encode(s)
    return seq[1:-1]
