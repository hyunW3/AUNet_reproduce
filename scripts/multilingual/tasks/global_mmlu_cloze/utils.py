import hashlib


# items whose question/option is empty or None in at least one language of the subset (ar/hi/te/vi);
# dropped everywhere so every language scores the same 2044 items
_BROKEN = {"college_chemistry/test/29", "conceptual_physics/test/17", "high_school_chemistry/test/49",
           "high_school_physics/test/30", "medical_genetics/test/48"}


def _keep(sample_id):
    # same ~1/7 subset (2049 of 14042 items) in every language: Global-MMLU rows are parallel by sample_id
    return int(hashlib.md5(sample_id.encode()).hexdigest(), 16) % 7 == 0 and sample_id not in _BROKEN


def process_docs(dataset):
    dataset = dataset.filter(lambda d: _keep(d["sample_id"]))
    return dataset.map(lambda d: {"choices": [d["option_a"], d["option_b"], d["option_c"], d["option_d"]],
                                  "gold": "ABCD".index(d["answer"].strip())})
