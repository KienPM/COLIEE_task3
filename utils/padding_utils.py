""" Create by Ken at 2021 Jan 10 """


def pad_query(vec, max_query_len):
    vec = vec[:max_query_len]
    if len(vec) < max_query_len:
        vec.extend([0] * (max_query_len - len(vec)))
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
            temp.extend([0] * (max_sen_len - temp_len))
        res.append(temp)
        i += max_sen_len

    return res


def pad_article(vec, max_num_sen, max_sen_len):
    vec = vec[:max_num_sen]
    if len(vec) < max_num_sen:
        vec.extend([[0] * max_sen_len] * (max_num_sen - len(vec)))
    return vec
