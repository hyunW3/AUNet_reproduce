"""Split BIG-bench tracking_shuffled_objects by object count.

The pooled task mixes 3/5/7-object items (n = 750/1250/1750), which have
different chance floors (1/3, 1/5, 1/7) AND different state-tracking difficulty.
Reporting one aggregate number hides both. These filters give one task per
object count, so accuracy can be read against its own floor and as a difficulty
ladder -- the natural-language analogue of the S5 depth sweep.
"""


def _by_choice_count(dataset, k):
    return dataset.filter(lambda d: len(d["multiple_choice_targets"]) == k)


def filter_3(dataset):
    return _by_choice_count(dataset, 3)


def filter_5(dataset):
    return _by_choice_count(dataset, 5)


def filter_7(dataset):
    return _by_choice_count(dataset, 7)


def filter_mc(dataset):
    """Drop rows with no multiple-choice targets (same as bigbench's own filter)."""
    return dataset.filter(lambda doc: len(doc["multiple_choice_targets"]) > 0)


def temporal_cloze_text(doc):
    """temporal_sequences with the inline choice list stripped -> pure cloze.

    The stock prompt enumerates the four options before the cue line:

        Between what times could Susan have gone to the coffee shop?
          choice: 2pm to 6pm
          ...
          choice: 6pm to 9pm
        Possible times:

    which makes it the ONLY one of the three NL state-tracking tasks whose
    options appear in the context (1000/1000 items; tracking_shuffled_objects
    and navigate are 0/1000 -- already cloze). That listing also gives the three
    distractors a second occurrence (passage + choice line) while the gold gets
    only one, so option frequency in context is confounded with correctness.

    Stripping it leaves: distractors occur once (in the passage), gold zero
    times. Same items, same four scored continuations -- only the listing moves.
    """
    return doc["inputs"].split("  choice:")[0].rstrip() + "\nPossible times:"
