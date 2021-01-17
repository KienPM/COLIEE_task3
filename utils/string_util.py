""" Create by Ken at 2021 Jan 03 """
import string
import re
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

table = str.maketrans('', '', string.punctuation)
number_re = re.compile(r'^\d+([.,-]\d+)*$')
clause_re = re.compile(r'^\(\d+\)')
lemmatizer = WordNetLemmatizer()


def remove_numbering(line):
    match_clause = clause_re.match(line)
    if match_clause:
        line = line[match_clause.span()[1] + 1:]
    return line


def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def pre_process_text(text):
    """
    Tokenize, lemmatize, convert to lower case, remove punctuations
    :type text: str
    :rtype: list[str]
    """
    tokens = word_tokenize(text)

    # lemmatize
    # # find the POS tag for each token
    nltk_tagged = pos_tag(tokens)
    # # tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_tokens = []
    for word, tag in wordnet_tagged:
        if tag is None:
            # if there is no available tag, append the token as is
            lemmatized_tokens.append(word)
        else:
            # else use the tag to lemmatize the token
            lemmatized_tokens.append(lemmatizer.lemmatize(word, tag))

    # convert to lower case
    tokens = [w.lower() for w in lemmatized_tokens]

    # remove punctuation from each word
    stripped = [w.translate(table) if not number_re.match(w) else w for w in tokens]

    words = [lemmatizer.lemmatize(w) for w in stripped if w != '']

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
