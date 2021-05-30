""" Create by Ken at 2021 Jan 10 """
PAD = 0
CLS = 101
SEP = 102


def pad_query(vec, max_query_len):
    vec = vec[:max_query_len]
    if len(vec) < max_query_len:
        vec = [CLS] + vec + [SEP] + [PAD] * (max_query_len - len(vec))
    else:
        vec = [CLS] + vec + [SEP]
    return vec


def pad_sentence(seq, max_sen_len):
    """
    Padding
    Split if len > max seq len
    """
    res = []
    seq_len = len(seq)
    i = 0
    while i < seq_len:
        temp = seq[i:i + max_sen_len]
        temp_len = len(temp)
        if temp_len < max_sen_len:
            temp = [CLS] + temp + [SEP] + ([PAD] * (max_sen_len - temp_len))
        else:
            temp = [CLS] + temp + [SEP]
        res.append(temp)
        i += max_sen_len

    return res


def pad_article(vec, max_num_sen, max_sen_len):
    vec = vec[:max_num_sen]
    pad_sen = [[CLS, SEP] + [0] * max_sen_len]
    if len(vec) < max_num_sen:
        vec.extend(pad_sen * (max_num_sen - len(vec)))
    return vec


def pad_article_flat(vec, max_article_len):
    vec = vec[:max_article_len]
    if len(vec) < max_article_len:
        vec = [CLS] + vec + [SEP] + [PAD] * (max_article_len - len(vec))
    else:
        vec = [CLS] + vec + [SEP]
    return vec
