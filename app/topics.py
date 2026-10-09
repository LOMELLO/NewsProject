"""Extract readable *topics* from a pool of fetched news articles.

Used when the user's topic matched nothing: instead of a dead end the app
offers the topics that its sources actually cover right now.
"""

from __future__ import annotations

import re
from collections import Counter

from .scraper import Article

# Generic filler words (Russian + English) that never make a useful topic.
STOPWORDS = set(
    """
    а без более был была были быть в во вот все всего да для до его ее если есть
    же за и из или им их к как ко когда кто ли либо мне может мы на над не него
    ни но ну о об один он она они от по под при про раз с со так такой там те тем
    то того тоже такой у уже что чтобы эта эти это этого чем через чего эту я
    вам вас ваш вашивой во-всего будут будет были бытия сказал сказалы говорит
    сообщает сообщаеть сообщаеть portal новость новости новости new latest news
    the and for with that this from have has had was were are will would about
    into more them they their there which what when who how why all any not but
    you your its his her said says report article after before than then also
    only some such may can could should must over under between during per via
    according against around because really even very much many one two three
    https http www com ru org html php aspx
    это этого эта эти это этих этот этой этим том той тех тому которой которого
    котором которые который которых также потом сейчас всех своей свою себе
    вообще именно такое такая
    заявили заявила заявил сообщил сообщили рассказал рассказала рассказали
    стал стала стали стало стать отпустили отпустил подписали подписал
    объявили объявила начали начал началась началось завершили завершил
    планируют планирует решили решил принял приняли ушел ушла ушли
    призвал призвала назвал назвала назвали считает считают считается
    уточнил уточнила добавил добавила напомнил напомнила подчеркнул
    подтвердили подтвердил пояснил пояснила объяснил объяснила
    могут хочет хотят должен должны нужно требуется становится
    остается продолжает продолжают
    год года году годов годам месяца месяц месяцев млрд тысяч
    января февраля марта апреля мая июня июля августа сентября
    октября ноября декабря
    """.split()
)

# Prepositions. The word right after one is the object of a preposition —
# «в области», «о выходе», «об этом» — and is never a useful topic on its
# own, so that occurrence is dropped.
LINK_WORDS = set(
    """
    в во на о об обо с со к ко по из за для при под над до от у про между
    через без около после перед вокруг среди
    """.split()
)

# Short but meaningful topic words (news is full of them).
ALLOW_SHORT = {"ai", "ии", "it", "рф", "цб", "эс", "nv", "eu", "нао"}

# Single letters matter too: «в», «с», «к» are prepositions that must break a
# run (otherwise «Свитолину в четвертьфинале» reads as one adjacent phrase).
_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё]+")

# Words that look like a real topic: long enough (or explicitly allowed),
# not a stopword, not a pure number.
_KEEP_RE = re.compile(r"[A-Za-zА-Яа-яЁё]")


def _normalize(word: str) -> str:
    return word.lower().replace("ё", "е")


def _runs(text: str) -> list[list[tuple[str, bool]]]:
    """Maximal runs of adjacent *kept* words as ``(word, is_proper)`` pairs.

    Anything that is skipped — a stopword, junk, the object of a
    preposition — breaks the run, so a phrase chip is only built from words
    that really stand next to each other in the original text.

    ``is_proper`` marks a word capitalised in the original that does not
    start a sentence: in sentence-case headlines that is a proper name, the
    strongest signal that a word is a real topic («Песков», «Киеве»).
    """
    runs: list[list[tuple[str, bool]]] = []
    current: list[tuple[str, bool]] = []
    after_link = False
    matches = list(_WORD_RE.finditer(text or ""))

    def flush() -> None:
        nonlocal current
        if current:
            runs.append(current)
            current = []

    for i, match in enumerate(matches):
        raw = match.group(0)
        word = _normalize(raw)
        before = (text or "")[: match.start()].rstrip()
        proper = bool(raw[:1].isupper()) and bool(before) and before[-1] not in ".!?:;"
        if word in ALLOW_SHORT:
            current.append((word, proper))
            after_link = False
            continue
        if word in LINK_WORDS:
            flush()
            after_link = True
            continue
        if after_link:
            flush()          # drop the object: «в области», «о выходе»
            after_link = False
            continue
        if word in STOPWORDS or len(word) < 4 or not _KEEP_RE.search(word):
            flush()
            after_link = False
            continue
        current.append((word, proper))
        after_link = False
    flush()
    return runs


def suggest_topics(articles: list[Article], limit: int = 8) -> list[str]:
    """Topics taken from the headlines of `articles` (best first).

    Only a word or phrase that stands in at least one *headline* can become a
    chip (the body text only adds weight to it), prepositional filler is
    dropped, and multi-word phrases win over single words — so the chips read
    like real topics instead of stray words from article bodies.
    """
    unigrams: Counter[str] = Counter()
    bigrams: Counter[str] = Counter()
    articles_with: dict[str, set[int]] = {}
    bigram_articles: dict[str, set[int]] = {}
    title_unigrams: dict[str, set[int]] = {}
    title_bigrams: dict[str, set[int]] = {}

    for index, article in enumerate(articles):
        for run in _runs(article.title):
            for word, proper in run:
                unigrams[word] += 3 + (6 if proper else 0)
                articles_with.setdefault(word, set()).add(index)
                title_unigrams.setdefault(word, set()).add(index)
            for (left, l_proper), (right, r_proper) in zip(run, run[1:]):
                phrase = f"{left} {right}"
                bigrams[phrase] += 3 + (6 if l_proper or r_proper else 0)
                bigram_articles.setdefault(phrase, set()).add(index)
                title_bigrams.setdefault(phrase, set()).add(index)
        for run in _runs((article.text or "")[:400]):
            for word, proper in run:
                unigrams[word] += 1 + (3 if proper else 0)
                articles_with.setdefault(word, set()).add(index)
            for (left, l_proper), (right, r_proper) in zip(run, run[1:]):
                phrase = f"{left} {right}"
                bigrams[phrase] += 1 + (3 if l_proper or r_proper else 0)
                bigram_articles.setdefault(phrase, set()).add(index)

    # phrase score is boosted so "искусственный интеллект" beats "интеллект"
    scored: list[tuple[float, str, str]] = []  # (score, kind, text)
    for phrase, score in bigrams.items():
        title_spread = len(title_bigrams.get(phrase, ()))
        if not title_spread:
            continue  # a body-only phrase is not a topic
        left, right = phrase.split(" ", 1)
        if left == right:
            continue
        spread = len(bigram_articles.get(phrase, ()))
        scored.append(
            ((score + spread * 2 + title_spread * 3) * 1.5, "phrase", phrase)
        )
    for word, score in unigrams.items():
        title_spread = len(title_unigrams.get(word, ()))
        if not title_spread:
            continue  # body-only filler never becomes a chip
        spread = len(articles_with.get(word, ()))
        scored.append((score + spread * 2 + title_spread * 4, "word", word))

    scored.sort(key=lambda item: item[0], reverse=True)

    picked: list[str] = []
    used_words: set[str] = set()
    for _score, kind, text in scored:
        if len(picked) >= limit:
            break
        if kind == "phrase":
            words = text.split()
            if any(w in used_words for w in words):
                continue
            picked.append(text)
            used_words.update(words)
        else:
            if text in used_words:
                continue
            picked.append(text)
            used_words.add(text)
    return picked
