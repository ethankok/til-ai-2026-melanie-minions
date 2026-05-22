"""Mine in-world slang terms from the NLP training corpus.

The hackathon notes that ASR transcripts share a world with the NLP dataset and
contain original slang/phrases. We extract tokens that appear repeatedly in the
NLP corpus but are rare in standard English, and write them as a single
space-separated string to ``slang_prompt.txt``. That file is loaded by
``ASRManager`` and passed as ``initial_prompt`` to faster-whisper so the decoder
prior is shifted toward the dataset's vocabulary.

Usage on the GCP Workbench instance::

    python training/asr/extract_slang.py \
        --nlp-dir /home/jupyter/novice/nlp \
        --out asr/models/slang_prompt.txt \
        --top-k 80

Whisper truncates ``initial_prompt`` to roughly the last 224 tokens, so 80
short words is a safe budget.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


_TOKEN_RE = re.compile(r"[A-Za-z]+")


def _iter_text_chunks(nlp_dir: Path):
    """Yield raw text chunks from anything that looks like an NLP corpus file.

    The NLP task ships documents in JSONL and/or plain text. Be permissive: try
    JSONL first (yielding any string-valued field), fall back to plain text.
    """
    for path in sorted(nlp_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".jsonl", ".json", ".txt", ".md"}:
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                if path.suffix.lower() in {".jsonl", ".json"}:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            yield line
                            continue
                        _yield_strings(obj, out=lambda s: None)
                        for s in _collect_strings(obj):
                            yield s
                else:
                    yield f.read()
        except (OSError, UnicodeDecodeError):
            continue


def _collect_strings(obj) -> list[str]:
    out: list[str] = []
    _yield_strings(obj, out=out.append)
    return out


def _yield_strings(obj, out) -> None:
    if isinstance(obj, str):
        out(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _yield_strings(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _yield_strings(v, out)


FALLBACK_WORDS = [
    "that", "this", "with", "from", "your", "have", "more", "will", "home", "about", "page", "search", "free", "other", "information",
    "time", "they", "site", "what", "which", "their", "news", "there", "only", "when", "contact", "here", "business", "also", "help",
    "view", "online", "first", "been", "would", "were", "services", "some", "these", "click", "like", "service", "than", "find", "price",
    "date", "back", "people", "list", "name", "just", "over", "state", "year", "into", "email", "health", "world", "next", "used",
    "work", "last", "most", "products", "music", "data", "make", "them", "should", "product", "system", "post", "city", "policy", "number",
    "such", "please", "available", "copyright", "support", "message", "after", "best", "software", "then", "good", "video", "well", "where", "info",
    "rights", "public", "books", "high", "school", "through", "each", "links", "review", "years", "order", "very", "privacy", "book", "items",
    "company", "read", "group", "need", "many", "user", "said", "does", "under", "general", "research", "university", "january", "mail", "full",
    "reviews", "program", "life", "know", "games", "days", "management", "part", "could", "great", "united", "hotel", "real", "item", "international",
    "center", "ebay", "must", "store", "travel", "comments", "made", "development", "report", "member", "details", "line", "terms", "before", "hotels",
    "send", "right", "type", "because", "local", "those", "using", "results", "office", "education", "national", "design", "take", "posted", "internet",
    "address", "community", "within", "states", "area", "want", "phone", "shipping", "reserved", "subject", "between", "forum", "family", "long", "based",
    "code", "show", "even", "black", "check", "special", "prices", "website", "index", "being", "women", "much", "sign", "file", "link",
    "open", "today", "technology", "south", "case", "project", "same", "pages", "version", "section", "found", "sports", "house", "related", "security",
    "both", "county", "american", "photo", "game", "members", "power", "while", "care", "network", "down", "computer", "systems", "three", "total",
    "place", "following", "download", "without", "access", "think", "north", "resources", "current", "posts", "media", "control", "water", "history", "pictures",
    "size", "personal", "since", "including", "guide", "shop", "directory", "board", "location", "change", "white", "text", "small", "rating", "rate",
    "government", "children", "during", "return", "students", "shopping", "account", "times", "sites", "level", "digital", "profile", "previous", "form", "events",
    "love", "john", "main", "call", "hours", "image", "department", "title", "description", "insurance", "another", "shall", "property", "class", "still",
    "money", "quality", "every", "listing", "content", "country", "private", "little", "visit", "save", "tools", "reply", "customer", "december", "compare",
    "movies", "include", "college", "value", "article", "york", "card", "jobs", "provide", "food", "source", "author", "different", "press", "learn",
    "sale", "around", "print", "course", "canada", "process", "teen", "room", "stock", "training", "credit", "point", "join", "science", "categories",
    "advanced", "west", "sales", "look", "english", "left", "team", "estate", "conditions", "select", "windows", "photos", "thread", "week", "category",
    "note", "live", "large", "gallery", "table", "register", "however", "june", "october", "november", "market", "library", "really", "action", "start",
    "series", "model", "features", "industry", "plan", "human", "provided", "required", "second", "accessories", "cost", "movie", "forums", "march", "september",
    "better", "questions", "july", "yahoo", "going", "medical", "test", "friend", "come", "server", "study", "application", "cart", "staff", "articles",
    "feedback", "again", "play", "looking", "issues", "april", "never", "users", "complete", "street", "topic", "comment", "financial", "things", "working",
    "against", "standard", "person", "below", "mobile", "less", "blog", "party", "payment", "equipment", "login", "student", "programs", "offers", "legal",
    "above", "recent", "park", "stores", "side", "problem", "give", "memory", "performance", "social", "august", "quote", "language", "story", "sell",
    "options", "experience", "rates", "create", "body", "young", "america", "important", "field", "east", "paper", "single", "activities", "club", "example",
    "girls", "additional", "password", "latest", "something", "road", "gift", "question", "changes", "night", "hard", "texas", "four", "poker", "status",
    "browse", "issue", "range", "building", "seller", "court", "february", "always", "result", "audio", "light", "write", "offer", "blue", "groups",
    "easy", "given", "files", "event", "release", "analysis", "request", "china", "making", "picture", "needs", "possible", "might", "professional", "month",
    "major", "star", "areas", "future", "space", "committee", "hand", "cards", "problems", "london", "washington", "meeting", "become", "interest", "child",
    "keep", "enter", "california", "share", "similar", "garden", "schools", "million", "added", "reference", "companies", "listed", "baby", "learning", "energy",
    "delivery", "popular", "term", "film", "stories", "computers", "journal", "reports", "welcome", "central", "images", "president", "notice", "original", "head",
    "radio", "until", "cell", "color", "self", "council", "away", "includes", "track", "australia", "discussion", "archive", "once", "others", "entertainment",
    "agreement", "format", "least", "society", "months", "safety", "friends", "sure", "trade", "edition", "cars", "messages", "marketing", "tell", "further",
    "updated", "association", "able", "having", "provides", "david", "already", "green", "studies", "close", "common", "drive", "specific", "several", "gold",
    "living", "collection", "called", "short", "arts", "display", "limited", "powered", "solutions", "means", "director", "daily", "beach", "past", "natural",
    "whether", "electronics", "five", "upon", "period", "planning", "database", "says", "official", "weather", "land", "average", "done", "technical", "window",
    "france", "region", "island", "record", "direct", "microsoft", "conference", "environment", "records", "district", "calendar", "costs", "style", "front", "statement",
    "update", "parts", "ever", "downloads", "early", "miles", "sound", "resource", "present", "applications", "either", "document", "word", "works", "material",
    "bill", "written", "talk", "federal", "hosting", "rules", "final", "adult", "tickets", "thing", "centre", "requirements", "cheap", "kids", "finance",
    "true", "minutes", "else", "mark", "third", "rock", "gifts", "europe", "reading", "topics", "individual", "tips", "plus", "auto", "cover",
    "usually", "edit", "together", "videos", "percent", "fast", "function", "fact", "unit", "getting", "global", "tech", "meet", "economic", "player",
    "projects", "lyrics", "often", "subscribe", "submit", "germany", "amount", "watch", "included", "feel", "though", "bank", "risk", "thanks", "everything",
    "deals", "various", "words", "linux", "production", "commercial", "james", "weight", "town", "heart", "advertising", "received", "choose", "treatment", "newsletter",
    "archives", "points", "knowledge", "magazine", "error", "camera", "girl", "currently", "construction", "toys", "registered", "clear", "golf", "receive", "domain",
    "methods", "chapter", "makes", "protection", "policies", "loan", "wide", "beauty", "manager", "india", "position", "taken", "sort", "listings", "models",
    "michael", "known", "half", "cases", "step", "engineering", "florida", "simple", "quick", "none", "wireless", "license", "paul", "friday", "lake",
    "whole", "annual", "published", "later", "basic", "sony", "shows", "corporate", "google", "church", "method", "purchase", "customers", "active", "response",
    "practice", "hardware", "figure", "materials", "fire", "holiday", "chat", "enough", "designed", "along", "among", "death", "writing", "speed", "html",
    "countries", "loss", "face", "brand", "discount", "higher", "effects", "created", "remember", "standards", "yellow", "political", "increase", "advertise", "kingdom",
    "base", "near", "environmental", "thought", "stuff", "french", "storage", "japan", "doing", "loans", "shoes", "entry", "stay", "nature", "orders",
    "availability", "africa", "summary", "turn", "mean", "growth", "notes", "agency", "king", "monday", "european", "activity", "copy", "although", "drug",
    "pics", "western", "income", "force", "cash", "employment", "overall", "river", "commission", "package", "contents", "seen", "players", "engine", "port",
    "album", "regional", "stop", "supplies", "started", "administration", "institute", "views", "plans", "double", "build", "screen", "exchange", "types", "soon",
    "sponsored", "lines", "electronic", "continue", "across", "benefits", "needed", "season", "apply", "someone", "held", "anything", "printer", "condition", "effective",
    "believe", "organization", "effect", "asked", "mind", "sunday", "selection", "casino", "lost", "tour", "menu", "volume", "cross", "anyone", "mortgage",
    "hope", "silver", "corporation", "wish", "inside", "solution", "mature", "role", "rather", "weeks", "addition", "came", "supply", "nothing", "certain",
    "executive", "running", "lower", "necessary", "union", "jewelry", "according", "clothing", "particular", "fine", "names", "robert", "homepage", "hour", "skills",
    "bush", "islands", "advice", "career", "military", "rental", "decision", "leave", "british", "teens", "huge", "woman", "facilities", "kind", "sellers",
    "middle", "move", "cable", "opportunities", "taking", "values", "division", "coming", "tuesday", "object", "lesbian", "appropriate", "machine", "logo", "length",
    "actually", "nice", "score", "statistics", "client", "returns", "capital", "follow", "sample", "investment", "sent", "shown", "saturday", "christmas", "england",
    "culture", "band", "flash", "lead", "george", "choice", "went", "starting", "registration", "thursday", "courses", "consumer", "airport", "foreign", "artist",
    "outside", "furniture", "levels", "channel", "letter", "mode", "phones", "ideas", "wednesday", "structure", "fund", "summer", "allow", "degree", "contract",
    "button", "releases", "homes", "super", "male", "matter", "custom", "virginia", "almost", "took", "located", "multiple", "asian", "distribution", "editor",
    "industrial", "cause", "potential", "song", "cnet", "focus", "late", "fall", "featured", "idea", "rooms", "female", "responsible", "communications", "associated",
    "thomas", "primary", "cancer", "numbers", "reason", "tool", "browser", "spring", "foundation", "answer", "voice", "friendly", "schedule", "documents", "communication",
    "purpose", "feature", "comes", "police", "everyone", "independent", "approach", "cameras", "brown", "physical", "operating", "hill", "maps", "medicine", "deal",
    "hold", "ratings", "chicago", "forms", "glass", "happy", "smith", "wanted", "developed", "thank", "safe", "unique", "survey", "prior", "telephone",
    "sport", "ready", "feed", "animal", "sources", "mexico", "population", "regular", "secure", "navigation", "operations", "therefore", "simply", "evidence", "station",
    "christian", "round", "paypal", "favorite", "understand", "option", "master", "valley", "recently", "probably", "rentals", "built", "publications", "blood", "worldwide",
    "improve", "connection", "publisher", "hall", "larger", "anti", "networks", "earth", "parents", "nokia", "impact", "transfer", "introduction", "kitchen", "strong",
    "carolina", "wedding", "properties", "hospital", "ground", "overview", "ship", "accommodation", "owners", "disease", "excellent", "paid", "italy", "perfect", "hair",
    "opportunity", "classic", "basis", "command", "cities", "william", "express", "award", "distance", "tree", "peter", "assessment", "ensure", "thus", "wall",
    "involved", "extra", "especially", "interface", "partners", "budget", "rated", "guides", "success", "maximum", "operation", "existing", "quite", "selected", "amazon",
    "patients", "restaurants", "beautiful", "warning", "wine", "locations", "horse", "vote", "forward", "flowers", "stars", "significant", "lists", "technologies", "owner",
    "retail", "animals", "useful", "directly", "manufacturer", "ways", "providing", "rule", "housing", "takes", "bring", "catalog", "searches", "trying", "mother",
    "authority", "considered", "told", "traffic", "programme", "joined", "input", "strategy", "feet", "agent", "valid", "modern", "senior", "ireland", "teaching",
    "door", "grand", "testing", "trial", "charge", "units", "instead", "canadian", "cool", "normal", "wrote", "enterprise", "ships", "entire", "educational",
    "leading", "metal", "positive", "fitness", "chinese", "opinion", "asia", "football", "abstract", "uses", "output", "funds", "greater", "likely", "develop",
    "employees", "artists", "alternative", "processing", "responsibility", "resolution", "java", "guest", "seems", "publication", "pass", "relations", "trust", "contains", "session",
    "multi", "photography", "republic", "fees", "components", "vacation", "century", "academic", "assistance", "completed", "skin", "graphics", "indian", "prev", "mary",
    "expected", "ring", "grade", "dating", "pacific", "mountain", "organizations", "filter", "mailing", "vehicle", "longer", "consider", "northern", "behind", "panel",
    "floor", "german", "buying", "match", "proposed", "default", "require", "iraq", "boys", "outdoor", "deep", "morning", "otherwise", "allows", "rest",
    "protein", "plant", "reported", "transportation", "pool", "mini", "politics", "partner", "disclaimer", "authors", "boards", "faculty", "parties", "fish", "membership",
    "mission", "string", "sense", "modified", "pack", "released", "stage", "internal", "goods", "recommended", "born", "unless", "richard", "failed", "detailed", "japanese",
    "race", "approved", "background", "target", "except", "character", "maintenance", "ability", "maybe", "functions", "moving", "brands", "places", "pretty", "trademarks",
    "phentermine", "spain", "southern", "yourself", "winter", "battery", "youth", "pressure", "submitted", "boston", "debt", "keywords", "medium", "television", "interested",
    "core", "break", "purposes", "throughout", "sets", "dance", "wood", "itself", "defined", "papers", "playing", "awards", "studio", "reader", "virtual",
    "device", "established", "answers", "rent", "remote", "dark", "programming", "external", "apple", "regarding", "instructions", "offered", "theory", "enjoy", "remove",
    "surface", "minimum", "visual", "host", "variety", "teachers", "isbn", "martin", "manual", "block", "subjects", "agents", "increased", "repair", "fair",
    "civil", "steel", "understanding", "songs", "fixed", "wrong", "beginning", "hands", "associates", "finally", "updates", "desktop", "classes", "paris", "ohio",
    "gets", "sector", "capacity", "requires", "jersey", "fully", "father", "electric", "instruments", "quotes", "officer", "driver", "businesses", "dead", "respect",
    "unknown", "specified", "restaurant", "mike", "trip", "worth", "procedures", "poor", "teacher", "eyes", "relationship", "workers", "farm", "georgia", "peace",
    "traditional", "campus", "showing", "creative", "coast", "benefit", "progress", "funding", "devices", "lord", "grant", "agree", "fiction", "hear", "sometimes",
    "watches", "careers", "beyond", "goes", "families", "museum", "themselves", "transport", "interesting", "blogs", "wife", "evaluation", "accepted", "former", "implementation",
    "hits", "zone", "complex", "galleries", "references", "presented", "jack", "flat", "flow", "agencies", "literature", "respective", "parent", "spanish", "michigan",
]


def _load_common_english() -> set[str]:
    """Return a lowercase set of common English words to subtract.

    Prefers wordfreq's top-N list; falls back to downloading the google-10000-english list;
    falls back to nltk's words corpus; falls back to the embedded FALLBACK_WORDS list.
    """
    try:
        from wordfreq import top_n_list

        return set(w.lower() for w in top_n_list("en", 50000))
    except Exception:
        pass

    try:
        import urllib.request
        url = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-no-swears.txt"
        with urllib.request.urlopen(url, timeout=5) as response:
            text = response.read().decode("utf-8")
            words_list = []
            for line in text.splitlines():
                w = line.strip().lower()
                if len(w) >= 4 and w.isalpha():
                    words_list.append(w)
            if len(words_list) > 100:
                return set(words_list)
    except Exception:
        pass

    try:
        from nltk.corpus import words  # type: ignore

        return set(w.lower() for w in words.words())
    except Exception:
        pass

    return set(FALLBACK_WORDS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--nlp-dir",
        type=Path,
        default=Path("/home/jupyter/novice/nlp"),
        help="Directory containing the NLP corpus files.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("asr/models/slang_prompt.txt"),
        help="Destination for the slang prompt text file.",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=200,
        help=(
            "Number of slang tokens to keep. Whisper truncates the "
            "initial_prompt to roughly the last 224 tokens, so ~200 short "
            "words is the safe budget. ERROR_ANALYSIS shows proper-noun "
            "substitutions (Sarento, Cyanite, Phyrexis, Mewan, etc.) drive a "
            "large share of remaining WER, so we want broad coverage."
        ),
    )
    ap.add_argument(
        "--min-count",
        type=int,
        default=2,
        help="Minimum occurrences in the NLP corpus.",
    )
    ap.add_argument(
        "--min-len",
        type=int,
        default=4,
        help="Minimum token length.",
    )
    args = ap.parse_args()

    if not args.nlp_dir.exists():
        raise SystemExit(f"NLP dir does not exist: {args.nlp_dir}")

    common = _load_common_english()
    counts: Counter[str] = Counter()
    for chunk in _iter_text_chunks(args.nlp_dir):
        for tok in _TOKEN_RE.findall(chunk):
            tok = tok.lower()
            if len(tok) < args.min_len:
                continue
            if tok in common:
                continue
            counts[tok] += 1

    ranked = [t for t, c in counts.most_common() if c >= args.min_count]
    slang = ranked[: args.top_k]

    # NOTE on ordering: faster-whisper truncates initial_prompt to the LAST
    # ~223 tokens (via previous_tokens[-(max_length // 2 - 1):]). With ~200
    # proper nouns the tokenized prompt overflows. Intuition would say to
    # reverse the list so high-frequency terms land at the end and survive
    # truncation -- BUT the vad-off-v2 experiment showed that ordering
    # produced WORSE local WER than the original highest-frequency-first
    # layout (0.060 vs 0.055 on the 1028-clip Workbench manifest). Likely
    # cause: putting the most common in-world nouns immediately before
    # decode-start over-primes the decoder and causes false-positive
    # hallucinations on unrelated clips. Keeping high-frequency first lets
    # those terms be in the prompt as background context but not in the
    # last-attended position. See training/asr/ERROR_ANALYSIS.md.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(" ".join(slang) + "\n", encoding="utf-8")

    print(f"Wrote {len(slang)} slang tokens to {args.out}")
    print("Top 20 (highest frequency, written first in the prompt):",
          " ".join(slang[:20]))


if __name__ == "__main__":
    main()
