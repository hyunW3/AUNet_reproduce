import hashlib


def _keep(sample_id):
    # same ~1/7 subset (2049 of 14042 items) in every language: Global-MMLU rows are parallel by sample_id
    return int(hashlib.md5(sample_id.encode()).hexdigest(), 16) % 7 == 0


def process_docs(dataset):
    dataset = dataset.filter(lambda d: _keep(d["sample_id"]))
    return dataset.map(lambda d: {"choices": [d["option_a"], d["option_b"], d["option_c"], d["option_d"]],
                                  "gold": "ABCD".index(d["answer"].strip())})
