"""Offline, keyword-based preview using entirely fictional records."""

import argparse
import re


RECORDS = {
    "Mumbai": (
        ("Fictional news brief", "Public transit delays prompt a trial of extra buses."),
        ("Fictional forum post", "A commuter suggests live alerts for transit delays."),
        ("Fictional news brief", "A neighborhood park adds shaded benches."),
    ),
    "Delhi": (
        ("Fictional news brief", "Public transit delays lead to revised train schedules."),
        ("Fictional forum post", "A rider asks for clearer updates during transit delays."),
        ("Fictional forum post", "Residents discuss adding trees near a park."),
    ),
}


def words(text):
    return set(re.findall(r"\w+", text.casefold()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default="public transit delays")
    query = words(parser.parse_args().query)
    if not query:
        parser.error("enter a query with at least one word")
    print("Synthetic demo — all records are fictional.")
    for city, records in RECORDS.items():
        print(f"\n{city}")
        ranked = sorted(records, key=lambda row: -len(query & words(row[1])))
        matches = [(source, snippet) for source, snippet in ranked if query & words(snippet)]
        for source, snippet in matches:
            print(f"  [{source}] {snippet}")
        if not matches:
            print("  No matching records.")


if __name__ == "__main__":
    main()
