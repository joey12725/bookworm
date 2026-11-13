from bookworm.text import approx_tokens, fold, snippet, split_sentences, stem, terms


def test_fold_strips_accents_and_curly_quotes() -> None:
    assert fold("Café’s NAÏVE") == "cafe's naive"


def test_stem_is_light() -> None:
    assert stem("daughters") == "daughter"
    assert stem("ladies") == "lady"
    assert stem("bennet's") == "bennet"
    assert stem("glasses") == "glass"
    assert stem("business") == "business"


def test_terms_drop_stopwords_but_keep_numbers() -> None:
    assert terms("The 5 daughters of Mr. Bennet") == ["5", "daughter", "mr", "bennet"]


def test_split_sentences_handles_titles_and_quotes() -> None:
    text = "“Is he married?” “Oh, single!” Mr. Bennet replied that he had not. Done."
    assert split_sentences(text) == [
        "“Is he married?”",
        "“Oh, single!”",
        "Mr. Bennet replied that he had not.",
        "Done.",
    ]


def test_approx_tokens() -> None:
    assert approx_tokens("") == 0
    assert approx_tokens("one two three") == 4


def test_snippet_starts_at_best_sentence() -> None:
    text = "The weather was fine. Darcy was proud and disagreeable. Then supper."
    assert snippet(text, "proud Darcy").startswith("… Darcy was proud")
    assert snippet(text, "weather") == text
    assert snippet("word " * 200, "word", width=40).endswith("…")
