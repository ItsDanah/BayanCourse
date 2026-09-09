"""Lab 3 starter: NER label alignment."""


def align_labels(word_ids, word_labels):
    """
    Align word-level NER labels with tokenizer subwords.

    Rules:
    - Special tokens with word_id=None get -100.
    - The first subword of each word receives the word's BIO label.
    - Additional subwords belonging to the same word get -100.
    """

    aligned_labels = []
    previous_word_id = None

    for word_id in word_ids:
        # Special tokens such as [CLS], [SEP], etc.
        if word_id is None:
            aligned_labels.append(-100)

        # First token/subword of a new word
        elif word_id != previous_word_id:
            aligned_labels.append(word_labels[word_id])

        # Additional subword of the same original word
        else:
            aligned_labels.append(-100)

        previous_word_id = word_id

    return aligned_labels