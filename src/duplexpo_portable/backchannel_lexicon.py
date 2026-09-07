"""
Backchannel Lexicon — Comprehensive Edition
=============================================
A rich taxonomy of backchannel tokens, bigrams, trigrams, and higher-order
n-grams used in conversational speech.  Organised by pragmatic function so
that downstream classifiers / dialogue systems can treat them categorically
when needed.

Categories
----------
AGREEMENT        – signals the listener agrees with the speaker
ACKNOWLEDGMENT   – signals the listener is following / has received info
SURPRISE         – signals unexpected information
EMPATHY          – signals emotional alignment or sympathy
ENCOURAGEMENT    – signals the listener wants the speaker to continue
HESITATION       – filled pauses / floor-holding tokens
DISAPPROVAL      – mild negative feedback or disagreement
AMUSEMENT        – laughter tokens / humour acknowledgment
DISCOURSE_MARKER – tokens that manage conversational flow
INTENSIFIER      – tokens that amplify or emphasise a backchannel
DISMISSAL        – tokens that downplay or brush off
REALISATION      – tokens that signal a new understanding
GRATITUDE        – brief thankful responses
FAREWELL         – conversational closing tokens
"""

from __future__ import annotations

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                          UNIGRAMS (single tokens)                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── Agreement ────────────────────────────────────────────────────────────────
AGREEMENT_TOKENS = {
    "yeah", "yep", "yup", "yes", "ya", "yah", "yea", "yass", "yas",
    "right", "true", "sure", "exactly", "definitely", "absolutely",
    "totally", "indeed", "correct", "precisely", "certainly", "agreed",
    "affirmative", "aye", "yeppers", "yessir", "yessum", "positively",
    "undoubtedly", "ditto", "seconded", "concur", "bingo", "surely",
    "obvi", "obviously", "clearly", "facts", "legit", "bet",
    "word",
}

# ── Acknowledgment ───────────────────────────────────────────────────────────
ACKNOWLEDGMENT_TOKENS = {
    "okay", "ok", "kay", "k", "kk", "okey", "okie", "okiedokie",
    "alright", "aight", "ight", "aite",
    "gotcha", "copy", "roger", "noted", "understood",
    "fair", "valid", "heard",
    "uh-huh", "uh huh", "uhuh", "uhhuh",
    "mm-hmm", "mmhmm", "mm hmm", "mhm", "mmhm", "umhm", "mhmm",
    "hmm", "hm", "hmmm",
    "ah", "ahh", "ahhh", "aha",
    "oh", "ohh", "ohhh",
    "uh", "uhh",
    "yuh",
    "sick", "dope", "tight", "lit", "fire",
}

# ── Surprise ─────────────────────────────────────────────────────────────────
SURPRISE_TOKENS = {
    "wow", "whoa", "woah", "damn", "dang", "darn", "gee", "geez", "jeez",
    "gosh", "golly", "goodness", "gracious", "yikes", "sheesh",
    "really", "seriously", "forreal",
    "huh", "what", "whaaat", "whaat",
    "omg", "omfg", "wtf",
    "crikey", "blimey", "cor", "cripes", "strewth",
    "wowza", "wowee", "dayum", "dayam",
    "unbelievable", "incredible", "insane", "crazy", "wild", "nuts",
    "nooo", "noooo",
    "bruh", "bro",
    "deadass", "cap", "sus",
}

# ── Empathy / Sympathy ──────────────────────────────────────────────────────
EMPATHY_TOKENS = {
    "aww", "aw", "awww", "awe",
    "oof", "ooof", "oooof",
    "yikes",
    "ouch", "ouchy",
    "geez",
    "bummer",
    "rough",
    "damn",
    "sorry",
    "same", "mood", "felt", "relatable",
}

# ── Encouragement / Continuer ───────────────────────────────────────────────
ENCOURAGEMENT_TOKENS = {
    "interesting", "fascinating", "intriguing",
    "cool", "nice", "great", "good", "awesome", "amazing",
    "sweet", "neat", "wonderful", "fantastic", "brilliant", "excellent",
    "perfect", "beautiful", "lovely", "splendid", "superb", "stellar",
    "fire", "sick", "dope", "rad", "epic", "legendary", "goated",
    "incredible",
    "yay", "woohoo", "hooray", "hurray",
    "impressive", "remarkable",
    "noice",
    "poggers", "pog",
    "based", "slay", "bussin", "valid",
}

# ── Hesitation / Filled Pauses ──────────────────────────────────────────────
HESITATION_TOKENS = {
    "um", "umm", "ummm",
    "uh", "uhh", "uhhh",
    "er", "err", "errr",
    "eh", "ehh",
    "hmm", "hm", "hmmm", "hmmmm",
    "well", "so", "like", "basically", "literally",
    "anyway", "anyways", "anyhow",
    "see", "look",
    "right",
    "lemme", "erm",
}

# ── Disapproval / Negative ──────────────────────────────────────────────────
DISAPPROVAL_TOKENS = {
    "no", "nah", "nope", "nay", "nuh-uh", "nuh uh", "nuhuh",
    "ew", "eww", "ewww", "ick", "yuck", "gross",
    "meh", "bleh", "blah",
    "tsk", "tut", "pfft", "pssh", "psh",
    "smh", "bruh", "cringe",
    "cap", "sus", "mid",
    "wack", "whack", "trash",
    "naur",
    "negative", "nein", "nyet",
    "ugh", "uggh",
}

# ── Amusement / Laughter ────────────────────────────────────────────────────
AMUSEMENT_TOKENS = {
    "haha", "hahaha", "hahahaha",
    "hehe", "hehehe",
    "hihi",
    "hoho", "hohoho",
    "lol", "lmao", "lmfao", "rofl", "rotfl",
    "jajaja", "kkkk", "www",
    "xd", "xdd",
    "dead", "dying", "deceased",
    "crying", "wheeze", "snort", "cackle",
    "funny", "hilarious", "priceless",
}

# ── Discourse Markers ───────────────────────────────────────────────────────
DISCOURSE_MARKER_TOKENS = {
    "so", "well", "like", "anyway", "anyways", "anyhow",
    "basically", "literally", "honestly", "actually", "technically",
    "apparently", "supposedly", "evidently",
    "look", "see", "listen", "hear",
    "now", "then", "meanwhile",
    "though", "although", "however", "still",
    "plus", "also", "besides", "furthermore",
    "granted", "regardless",
    "ngl", "tbh", "tbf", "imo", "imho", "fwiw", "btw", "fyi",
}

# ── Realisation ──────────────────────────────────────────────────────────────
REALISATION_TOKENS = {
    "oh", "ohh", "ohhh",
    "aha", "ahh", "ahhh",
    "eureka", "duh",
    "oooh", "ooh",
    "wait", "omg",
    "right",
}

# ── Gratitude ────────────────────────────────────────────────────────────────
GRATITUDE_TOKENS = {
    "thanks", "thankyou", "thx", "ty", "tyvm", "tysm",
    "cheers", "ta",
    "appreciate",
    "gracias", "merci", "danke", "grazie", "arigato",
}

# ── Farewell / Closing ──────────────────────────────────────────────────────
FAREWELL_TOKENS = {
    "bye", "byebye", "goodbye", "cya", "seeya", "later", "laterz",
    "peace", "peaceout", "deuces",
    "ttyl", "ttfn", "tata",
    "night", "nighty", "gnight", "goodnight",
    "adios", "ciao", "sayonara",
}


# ── Master unigram set (union of all categories) ────────────────────────────
BACKCHANNEL_LEXICON: set[str] = (
    AGREEMENT_TOKENS
    | ACKNOWLEDGMENT_TOKENS
    | SURPRISE_TOKENS
    | EMPATHY_TOKENS
    | ENCOURAGEMENT_TOKENS
    | HESITATION_TOKENS
    | DISAPPROVAL_TOKENS
    | AMUSEMENT_TOKENS
    | DISCOURSE_MARKER_TOKENS
    | REALISATION_TOKENS
    | GRATITUDE_TOKENS
    | FAREWELL_TOKENS
)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                               BIGRAMS                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

BACKCHANNEL_BIGRAMS: set[tuple[str, str]] = {
    # ── Agreement ────────────────────────────────────────────────────────
    ("that's", "right"), ("that's", "true"), ("that's", "correct"),
    ("that's", "fair"), ("that's", "valid"),
    ("you're", "right"), ("you're", "correct"),
    ("so", "true"), ("very", "true"),
    ("for", "sure"), ("for", "real"),
    ("of", "course"), ("no", "doubt"),
    ("fair", "enough"), ("fair", "point"), ("good", "point"),
    ("makes", "sense"),
    ("100", "percent"),
    ("damn", "right"), ("hell", "yeah"), ("hell", "yes"),
    ("big", "facts"), ("no", "cap"), ("say", "less"), ("on", "god"),

    # ── Acknowledgment ───────────────────────────────────────────────────
    ("i", "see"), ("i", "know"), ("i", "understand"), ("i", "hear"),
    ("uh", "huh"), ("mm", "hmm"),
    ("got", "it"), ("copy", "that"), ("roger", "that"),
    ("all", "right"),
    ("right", "right"), ("okay", "okay"), ("ok", "ok"),
    ("yeah", "yeah"), ("yes", "yes"), ("sure", "sure"),
    ("go", "ahead"), ("go", "on"),

    # ── Surprise ─────────────────────────────────────────────────────────
    ("oh", "really"), ("oh", "wow"), ("oh", "damn"), ("oh", "snap"),
    ("oh", "shoot"), ("oh", "shit"), ("oh", "crap"),
    ("oh", "dear"), ("oh", "lord"), ("oh", "gosh"),
    ("oh", "geez"), ("oh", "jeez"),
    ("oh", "boy"), ("oh", "man"), ("oh", "god"), ("oh", "goodness"),
    ("oh", "my"), ("oh", "no"),
    ("oh", "great"), ("oh", "fantastic"),
    ("no", "way"), ("no", "kidding"), ("no", "shit"),
    ("get", "out"), ("shut", "up"),
    ("holy", "shit"), ("holy", "cow"), ("holy", "crap"),
    ("holy", "moly"), ("holy", "smokes"),
    ("good", "lord"), ("good", "grief"), ("good", "god"), ("good", "gracious"),
    ("my", "god"), ("my", "goodness"), ("my", "gosh"),
    ("dear", "god"), ("dear", "lord"),
    ("wait", "what"), ("wait", "really"),
    ("say", "what"), ("come", "on"),
    ("you're", "kidding"), ("you're", "joking"),

    # ── Empathy / Sympathy ───────────────────────────────────────────────
    ("oh", "well"), ("so", "sorry"),
    ("that", "sucks"),
    ("that's", "rough"), ("that's", "tough"), ("that's", "hard"),
    ("that's", "sad"), ("that's", "awful"),
    ("that's", "terrible"), ("that's", "horrible"),
    ("poor", "thing"), ("poor", "you"),
    ("how", "awful"), ("how", "terrible"),
    ("i'm", "sorry"),
    ("me", "too"), ("same", "here"),
    ("big", "mood"), ("felt", "that"), ("i", "felt"),

    # ── Encouragement / Positive ─────────────────────────────────────────
    ("oh", "nice"), ("oh", "cool"), ("oh", "awesome"), ("oh", "sweet"),
    ("oh", "ok"), ("oh", "okay"), ("oh", "yeah"), ("oh", "yes"),
    ("sounds", "good"), ("sounds", "great"), ("sounds", "right"),
    ("sounds", "fun"), ("sounds", "cool"), ("sounds", "interesting"),
    ("sounds", "lovely"), ("sounds", "perfect"),
    ("sounds", "about"),
    ("looks", "good"), ("looks", "great"), ("looks", "nice"),
    ("that's", "good"), ("that's", "nice"), ("that's", "great"),
    ("that's", "cool"), ("that's", "awesome"), ("that's", "amazing"),
    ("that's", "wonderful"), ("that's", "fantastic"),
    ("that's", "perfect"), ("that's", "brilliant"),
    ("that's", "incredible"), ("that's", "interesting"),
    ("that's", "hilarious"), ("that's", "wild"),
    ("that's", "sick"), ("that's", "dope"),
    ("that's", "fire"), ("that's", "lit"), ("that's", "epic"),
    ("well", "done"), ("good", "job"), ("nice", "one"),
    ("way", "cool"),
    ("not", "bad"), ("pretty", "cool"), ("pretty", "good"), ("pretty", "nice"),
    ("how", "cool"), ("how", "nice"), ("how", "interesting"),
    ("let's", "go"), ("let's", "gooo"),
    ("i", "bet"), ("you", "bet"),

    # ── Disapproval / Negative ───────────────────────────────────────────
    ("no", "thanks"), ("nah", "fam"),
    ("yeah", "no"), ("yeah", "right"), ("sure", "jan"),
    ("as", "if"),
    ("i", "doubt"),
    ("not", "really"), ("not", "quite"), ("not", "exactly"),
    ("don't", "think"),

    # ── Amusement ────────────────────────────────────────────────────────
    ("ha", "ha"), ("he", "he"),
    ("so", "funny"), ("too", "funny"),
    ("i'm", "dead"), ("i'm", "dying"), ("i'm", "crying"),
    ("i'm", "deceased"), ("i'm", "screaming"), ("i'm", "wheezing"),
    ("i", "can't"), ("stop", "it"),

    # ── Discourse / Hedging ──────────────────────────────────────────────
    ("i", "mean"), ("i", "guess"), ("i", "suppose"), ("i", "think"),
    ("you", "know"), ("i", "dunno"),
    ("sort", "of"), ("kind", "of"),
    ("in", "fairness"),
    ("to", "be"),
    ("not", "gonna"),
    ("low", "key"), ("high", "key"),

    # ── Realisation ──────────────────────────────────────────────────────
    ("oh", "right"), ("oh", "wait"), ("oh", "that's"),
    ("ah", "right"), ("ah", "yes"), ("ah", "ok"), ("ah", "okay"),
    ("oh", "duh"), ("ohhh", "right"), ("wait", "so"),

    # ── Gratitude ────────────────────────────────────────────────────────
    ("thank", "you"),
    ("thanks", "man"), ("thanks", "bro"), ("thanks", "dude"), ("thanks", "mate"),
    ("much", "appreciated"),

    # ── Farewell ─────────────────────────────────────────────────────────
    ("see", "ya"), ("take", "care"), ("peace", "out"),
    ("catch", "ya"), ("talk", "later"), ("gotta", "go"),

    # ── Legacy (from original, kept for backwards compat) ────────────────
    ("is", "that"),
}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                               TRIGRAMS                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

BACKCHANNEL_TRIGRAMS: set[tuple[str, str, str]] = {
    # ── Agreement ────────────────────────────────────────────────────────
    ("i", "know", "right"),
    ("that's", "so", "true"),
    ("that", "makes", "sense"),
    ("that's", "a", "fact"),
    ("you're", "so", "right"),
    ("you're", "not", "wrong"),
    ("couldn't", "agree", "more"),
    ("you", "said", "it"),
    ("say", "no", "more"),
    ("you", "know", "it"),
    ("ain't", "that", "right"),
    ("for", "real", "though"),
    ("on", "god", "bro"),
    ("i", "feel", "that"),

    # ── Acknowledgment ───────────────────────────────────────────────────
    ("oh", "i", "see"),
    ("i", "see", "yeah"),
    ("yeah", "i", "see"),
    ("yeah", "of", "course"),
    ("oh", "of", "course"),
    ("yeah", "for", "sure"),
    ("oh", "for", "sure"),
    ("yeah", "makes", "sense"),
    ("i", "got", "it"),
    ("i", "got", "you"),
    ("i", "gotcha", "yeah"),
    ("i", "hear", "you"),
    ("i", "hear", "ya"),
    ("yeah", "i", "know"),
    ("yeah", "that's", "right"),
    ("yeah", "that's", "true"),

    # ── Surprise ─────────────────────────────────────────────────────────
    ("oh", "my", "god"),
    ("oh", "my", "gosh"),
    ("oh", "my", "goodness"),
    ("oh", "my", "lord"),
    ("oh", "my", "word"),
    ("oh", "my", "days"),
    ("oh", "my", "lanta"),
    ("oh", "dear", "god"),
    ("oh", "dear", "lord"),
    ("oh", "for", "real"),
    ("is", "that", "right"),
    ("is", "that", "so"),
    ("is", "that", "true"),
    ("are", "you", "serious"),
    ("are", "you", "kidding"),
    ("are", "you", "joking"),
    ("are", "you", "sure"),
    ("you're", "not", "serious"),
    ("get", "out", "of"),
    ("shut", "the", "front"),
    ("what", "the", "heck"),
    ("what", "the", "hell"),
    ("what", "the", "fuck"),
    ("what", "on", "earth"),
    ("what", "in", "the"),
    ("how", "is", "that"),
    ("how", "about", "that"),
    ("well", "i'll", "be"),
    ("i'll", "be", "damned"),
    ("can", "you", "believe"),
    ("you", "don't", "say"),
    ("no", "way", "really"),
    ("no", "freaking", "way"),
    ("holy", "mother", "of"),
    ("son", "of", "a"),

    # ── Empathy / Sympathy ───────────────────────────────────────────────
    ("i'm", "so", "sorry"),
    ("that's", "so", "sad"),
    ("that's", "too", "bad"),
    ("oh", "that's", "terrible"),
    ("oh", "that's", "awful"),
    ("oh", "poor", "thing"),
    ("oh", "no", "way"),
    ("oh", "bless", "you"),
    ("bless", "your", "heart"),
    ("i", "feel", "you"),
    ("hang", "in", "there"),
    ("that", "really", "sucks"),
    ("oh", "come", "on"),
    ("been", "there", "done"),

    # ── Encouragement / Positive ─────────────────────────────────────────
    ("that's", "so", "cool"),
    ("that's", "so", "great"),
    ("that's", "so", "nice"),
    ("that's", "so", "awesome"),
    ("that's", "so", "sweet"),
    ("that's", "so", "funny"),
    ("that's", "so", "interesting"),
    ("that's", "so", "good"),
    ("that", "sounds", "great"),
    ("that", "sounds", "good"),
    ("that", "sounds", "fun"),
    ("that", "sounds", "perfect"),
    ("that", "sounds", "amazing"),
    ("good", "for", "you"),
    ("good", "on", "you"),
    ("way", "to", "go"),
    ("keep", "it", "up"),
    ("i", "love", "that"),
    ("i", "love", "it"),
    ("love", "to", "hear"),
    ("that's", "what's", "up"),
    ("you", "love", "to"),
    ("let's", "freaking", "go"),

    # ── Disapproval / Sarcasm ────────────────────────────────────────────
    ("yeah", "no", "thanks"),
    ("yeah", "right", "sure"),
    ("give", "me", "a"),
    ("you've", "gotta", "be"),
    ("i", "don't", "think"),
    ("not", "gonna", "lie"),
    ("i", "highly", "doubt"),
    ("hard", "to", "believe"),

    # ── Amusement ────────────────────────────────────────────────────────
    ("that's", "too", "funny"),
    ("i'm", "literally", "dead"),
    ("i'm", "literally", "dying"),
    ("i", "can't", "breathe"),
    ("i", "can't", "even"),
    ("stop", "i'm", "dead"),
    ("bro", "i'm", "dead"),
    ("lol", "that's", "hilarious"),
    ("dying", "of", "laughter"),

    # ── Discourse ────────────────────────────────────────────────────────
    ("i", "mean", "yeah"),
    ("i", "mean", "like"),
    ("you", "know", "what"),
    ("to", "be", "fair"),
    ("to", "be", "honest"),
    ("at", "the", "end"),
    ("if", "you", "say"),
    ("if", "i'm", "honest"),
    ("to", "be", "real"),
    ("real", "talk", "though"),
    ("low", "key", "though"),

    # ── Realisation ──────────────────────────────────────────────────────
    ("oh", "wait", "really"),
    ("oh", "that", "makes"),
    ("oh", "right", "right"),
    ("oh", "yeah", "right"),
    ("oh", "i", "forgot"),
    ("oh", "i", "remember"),
    ("now", "i", "see"),
    ("now", "i", "get"),
    ("now", "i", "understand"),
    ("oh", "snap", "yeah"),
    ("ohh", "that's", "why"),

    # ── Gratitude ────────────────────────────────────────────────────────
    ("thank", "you", "so"),
    ("thanks", "a", "lot"),
    ("thanks", "so", "much"),
    ("much", "appreciated", "thanks"),
    ("i", "appreciate", "it"),
    ("thanks", "for", "that"),

    # ── Farewell ─────────────────────────────────────────────────────────
    ("see", "you", "later"),
    ("see", "you", "soon"),
    ("talk", "to", "you"),
    ("have", "a", "good"),
    ("catch", "you", "later"),
    ("peace", "out", "bro"),
}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                          QUADGRAMS (4-grams)                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

BACKCHANNEL_QUADGRAMS: set[tuple[str, str, str, str]] = {
    # ── Agreement ────────────────────────────────────────────────────────
    ("i", "couldn't", "agree", "more"),
    ("you", "can", "say", "that"),
    ("that's", "what", "i'm", "saying"),
    ("that's", "what", "i", "said"),
    ("i", "know", "right", "like"),

    # ── Surprise ─────────────────────────────────────────────────────────
    ("oh", "my", "god", "really"),
    ("oh", "my", "god", "no"),
    ("oh", "my", "god", "yes"),
    ("oh", "my", "god", "wow"),
    ("are", "you", "for", "real"),
    ("are", "you", "kidding", "me"),
    ("you", "have", "got", "to"),
    ("you've", "got", "to", "be"),
    ("get", "out", "of", "here"),
    ("shut", "the", "front", "door"),
    ("what", "in", "the", "world"),
    ("how", "in", "the", "world"),
    ("well", "i'll", "be", "damned"),
    ("i", "can't", "believe", "it"),
    ("i", "can't", "believe", "that"),
    ("no", "way", "that's", "crazy"),
    ("what", "the", "heck", "man"),

    # ── Empathy ──────────────────────────────────────────────────────────
    ("i'm", "sorry", "to", "hear"),
    ("i'm", "sorry", "about", "that"),
    ("that's", "really", "too", "bad"),
    ("oh", "no", "i'm", "sorry"),
    ("been", "there", "done", "that"),
    ("i", "know", "the", "feeling"),
    ("i", "know", "how", "you"),
    ("that", "must", "be", "tough"),
    ("that", "must", "be", "hard"),

    # ── Encouragement ────────────────────────────────────────────────────
    ("that's", "really", "cool", "man"),
    ("i", "love", "to", "hear"),
    ("you", "love", "to", "see"),
    ("that's", "what", "i'm", "talking"),
    ("now", "that's", "what", "i"),
    ("keep", "up", "the", "good"),

    # ── Disapproval ──────────────────────────────────────────────────────
    ("give", "me", "a", "break"),
    ("you've", "gotta", "be", "kidding"),
    ("i", "don't", "buy", "it"),
    ("i", "don't", "think", "so"),
    ("i", "beg", "to", "differ"),
    ("yeah", "i", "don't", "think"),
    ("that's", "a", "bit", "much"),

    # ── Amusement ────────────────────────────────────────────────────────
    ("i'm", "literally", "crying", "laughing"),
    ("i", "can't", "stop", "laughing"),
    ("i", "can't", "even", "right"),
    ("that", "was", "so", "funny"),
    ("oh", "my", "god", "lol"),
    ("bro", "i'm", "so", "dead"),

    # ── Discourse ────────────────────────────────────────────────────────
    ("at", "the", "end", "of"),
    ("if", "you", "ask", "me"),
    ("as", "far", "as", "i"),
    ("if", "i'm", "being", "honest"),
    ("if", "you", "say", "so"),
    ("to", "be", "fair", "though"),
    ("for", "what", "it's", "worth"),
    ("not", "going", "to", "lie"),
    ("to", "be", "completely", "honest"),
    ("i", "mean", "to", "be"),

    # ── Realisation ──────────────────────────────────────────────────────
    ("oh", "that", "makes", "sense"),
    ("now", "that", "you", "mention"),
    ("oh", "i", "didn't", "know"),
    ("oh", "i", "see", "what"),
    ("now", "i", "get", "it"),
    ("i", "just", "realized", "that"),

    # ── Farewell ─────────────────────────────────────────────────────────
    ("talk", "to", "you", "later"),
    ("see", "you", "next", "time"),
    ("have", "a", "good", "one"),
    ("it", "was", "nice", "talking"),
    ("catch", "you", "on", "the"),
}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                          PENTAGRAMS (5-grams)                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

BACKCHANNEL_PENTAGRAMS: set[tuple[str, str, str, str, str]] = {
    ("you", "can", "say", "that", "again"),
    ("that's", "what", "i'm", "talking", "about"),
    ("now", "that's", "what", "i'm", "talking"),
    ("i", "know", "what", "you", "mean"),
    ("tell", "me", "about", "it", "man"),
    ("at", "the", "end", "of", "the"),
    ("if", "you", "know", "what", "i"),
    ("you", "don't", "have", "to", "tell"),
    ("i", "know", "exactly", "what", "you"),
    ("i'm", "sorry", "to", "hear", "that"),
    ("i", "can't", "believe", "you", "just"),
    ("oh", "my", "god", "i", "can't"),
    ("keep", "up", "the", "good", "work"),
    ("that's", "what", "i", "was", "thinking"),
    ("i", "was", "just", "about", "to"),
    ("for", "what", "it's", "worth", "though"),
    ("as", "far", "as", "i", "know"),
}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                     CATEGORY → TOKEN MAPPING                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

CATEGORY_MAP: dict[str, set[str]] = {
    "agreement":       AGREEMENT_TOKENS,
    "acknowledgment":  ACKNOWLEDGMENT_TOKENS,
    "surprise":        SURPRISE_TOKENS,
    "empathy":         EMPATHY_TOKENS,
    "encouragement":   ENCOURAGEMENT_TOKENS,
    "hesitation":      HESITATION_TOKENS,
    "disapproval":     DISAPPROVAL_TOKENS,
    "amusement":       AMUSEMENT_TOKENS,
    "discourse":       DISCOURSE_MARKER_TOKENS,
    "realisation":     REALISATION_TOKENS,
    "gratitude":       GRATITUDE_TOKENS,
    "farewell":        FAREWELL_TOKENS,
}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                        UTILITY HELPERS                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def classify_token(token: str) -> list[str]:
    """Return all categories a single token belongs to."""
    token = token.lower().strip()
    return [cat for cat, tokens in CATEGORY_MAP.items() if token in tokens]


def is_backchannel_unigram(token: str) -> bool:
    """Check whether a single token is in the master lexicon."""
    return token.lower().strip() in BACKCHANNEL_LEXICON


def is_backchannel_bigram(w1: str, w2: str) -> bool:
    return (w1.lower(), w2.lower()) in BACKCHANNEL_BIGRAMS


def is_backchannel_trigram(w1: str, w2: str, w3: str) -> bool:
    return (w1.lower(), w2.lower(), w3.lower()) in BACKCHANNEL_TRIGRAMS


def is_backchannel_quadgram(w1: str, w2: str, w3: str, w4: str) -> bool:
    return (w1.lower(), w2.lower(), w3.lower(), w4.lower()) in BACKCHANNEL_QUADGRAMS


def find_backchannels(tokens: list[str]) -> list[dict]:
    """
    Scan a token list and return all backchannel spans found.
    Greedy: scans longest n-grams first.

    Returns a list of dicts:
        {"span": (start, end), "text": str, "ngram": int}
    """
    tokens_lower = [t.lower() for t in tokens]
    n = len(tokens_lower)
    results: list[dict] = []

    for i in range(n):
        # 5-grams
        if i + 5 <= n:
            gram = tuple(tokens_lower[i:i + 5])
            if gram in BACKCHANNEL_PENTAGRAMS:
                results.append({"span": (i, i + 5), "text": " ".join(tokens[i:i + 5]), "ngram": 5})
                continue
        # 4-grams
        if i + 4 <= n:
            gram = tuple(tokens_lower[i:i + 4])
            if gram in BACKCHANNEL_QUADGRAMS:
                results.append({"span": (i, i + 4), "text": " ".join(tokens[i:i + 4]), "ngram": 4})
                continue
        # 3-grams
        if i + 3 <= n:
            gram = tuple(tokens_lower[i:i + 3])
            if gram in BACKCHANNEL_TRIGRAMS:
                results.append({"span": (i, i + 3), "text": " ".join(tokens[i:i + 3]), "ngram": 3})
                continue
        # 2-grams
        if i + 2 <= n:
            gram = tuple(tokens_lower[i:i + 2])
            if gram in BACKCHANNEL_BIGRAMS:
                results.append({"span": (i, i + 2), "text": " ".join(tokens[i:i + 2]), "ngram": 2})
                continue
        # 1-grams
        if tokens_lower[i] in BACKCHANNEL_LEXICON:
            results.append({"span": (i, i + 1), "text": tokens[i], "ngram": 1})

    return results


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              STATS                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    print("Backchannel Lexicon — Stats")
    print("=" * 50)
    print(f"  Unigrams  (master set) : {len(BACKCHANNEL_LEXICON):>5}")
    print(f"  Bigrams               : {len(BACKCHANNEL_BIGRAMS):>5}")
    print(f"  Trigrams              : {len(BACKCHANNEL_TRIGRAMS):>5}")
    print(f"  Quadgrams             : {len(BACKCHANNEL_QUADGRAMS):>5}")
    print(f"  Pentagrams            : {len(BACKCHANNEL_PENTAGRAMS):>5}")
    total = (
        len(BACKCHANNEL_LEXICON)
        + len(BACKCHANNEL_BIGRAMS)
        + len(BACKCHANNEL_TRIGRAMS)
        + len(BACKCHANNEL_QUADGRAMS)
        + len(BACKCHANNEL_PENTAGRAMS)
    )
    print(f"  ────────────────────────────")
    print(f"  Total entries         : {total:>5}")
    print()
    print("Category breakdown (unigrams):")
    for cat, toks in sorted(CATEGORY_MAP.items()):
        print(f"  {cat:<20s}: {len(toks):>4} tokens")
