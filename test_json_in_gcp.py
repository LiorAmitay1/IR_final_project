import json

import requests
from time import time


def init():
    global queries
    with open('queries_train.json', 'rt') as f:
        queries = json.load(f)


def average_precision(true_list, predicted_list, k=40):
    true_set = frozenset(true_list)
    predicted_list = predicted_list[:k]
    precisions = []
    for i, doc_id in enumerate(predicted_list):
        if doc_id in true_set:
            prec = (len(precisions) + 1) / (i + 1)
            precisions.append(prec)
    if len(precisions) == 0:
        return 0.0
    return round(sum(precisions) / len(precisions), 3)


def precision_at_k(true_list, predicted_list, k):
    true_set = frozenset(true_list)
    predicted_list = predicted_list[:k]
    if len(predicted_list) == 0:
        return 0.0
    return round(len([1 for doc_id in predicted_list if doc_id in true_set]) / len(predicted_list), 3)


def recall_at_k(true_list, predicted_list, k):
    true_set = frozenset(true_list)
    predicted_list = predicted_list[:k]
    if len(true_set) < 1:
        return 1.0
    return round(len([1 for doc_id in predicted_list if doc_id in true_set]) / len(true_set), 3)


def f1_at_k(true_list, predicted_list, k):
    p = precision_at_k(true_list, predicted_list, k)
    r = recall_at_k(true_list, predicted_list, k)
    if p == 0.0 or r == 0.0:
        return 0.0
    return round(2.0 / (1.0 / p + 1.0 / r), 3)

lst=[]
def results_quality(true_list, predicted_list):
    p5 = precision_at_k(true_list, predicted_list, 5)
    p10 = precision_at_k(true_list, predicted_list, 10)

    lst.append(p10)
    f1_30 = f1_at_k(true_list, predicted_list, 30)
    if p5 == 0.0 or f1_30 == 0.0:
        return 0.0
    return round(2.0 / (1.0 / p5 + 1.0 / f1_30), 3)


assert precision_at_k(range(10), [1, 2, 3], 2) == 1.0
assert recall_at_k(range(10), [10, 5, 3], 2) == 0.1
assert precision_at_k(range(10), [], 2) == 0.0
assert precision_at_k([], [1, 2, 3], 5) == 0.0
assert recall_at_k([], [10, 5, 3], 2) == 1.0
assert recall_at_k(range(10), [], 2) == 0.0
assert f1_at_k([], [1, 2, 3], 5) == 0.0
assert f1_at_k(range(10), [], 2) == 0.0
assert f1_at_k(range(10), [0, 1, 2], 2) == 0.333
assert f1_at_k(range(50), range(5), 30) == 0.182
assert f1_at_k(range(50), range(10), 30) == 0.333
assert f1_at_k(range(50), range(30), 30) == 0.75
assert results_quality(range(50), range(5)) == 0.308
assert results_quality(range(50), range(10)) == 0.5
assert results_quality(range(50), range(30)) == 0.857
# assert results_quality(range(50), [-1] * 5 + list(range(5, 30))) == 0.0


def test_run():
    # http://34.70.74.95:8080//search?query=When+did+the+Black+Death+pandemic+occur?
    url = 'http://34.70.74.95:8080'
    # place the domain you got from ngrok or GCP IP below.
    # url = 'http://c30f-35-227-188-52.ngrok-free.app'
    qs_res = []
    for q, true_wids in queries.items():
        rq = None
        duration, ap = None, None
        t_start = time()
        try:
            res = requests.get(url + '/search', {'query': q}, timeout=35)
            # res = requests.get(url + '/search_bolean_and_cosine', {'query': q}, timeout=35)
            # res = requests.get(url + '/search_bm25_and_cosine', {'query': q}, timeout=35)
            # res = requests.get(url + '/search_with_stemming', {'query': q}, timeout=35)
            duration = time() - t_start
            if res.status_code == 200:
                pred_wids, _ = zip(*res.json())
                rq = results_quality(true_wids, pred_wids)
        except:
            pass
        qs_res.append((q, duration, rq))
        print((q, duration, rq))
        # break
    return qs_res


if __name__ == '__main__':
    init()
    # time = time()
    test_run()
    # rest = time() - time
    print(sum(lst) / len(lst))
