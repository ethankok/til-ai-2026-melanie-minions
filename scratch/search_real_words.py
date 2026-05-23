import json
import re

with open("asr.jsonl", "r", encoding="utf-8") as f:
    gold = [json.loads(line)["transcript"] for line in f]

def check_word(word):
    print(f"=== Matches for '{word}' ===")
    matches = [g for g in gold if re.search(rf"\b{word}\b", g, re.I)]
    print(f"Total occurrences: {len(matches)}")
    for m in matches[:5]:
        print(f" - {m}")

check_word("reverent")
check_word("reverend")
check_word("bloc")
check_word("block")
check_word("dreamer")
check_word("streamer")
