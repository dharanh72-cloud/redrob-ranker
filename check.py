import csv

with open('submission.csv') as f:
    rows = list(csv.DictReader(f))

total = len(rows)
ranks = sorted([int(r["rank"]) for r in rows])
unique_ids = len(set(r["candidate_id"] for r in rows))
ranks_ok = ranks == list(range(1, 101))

print(f"Total rows     : {total}")
print(f"Ranks 1-100    : {ranks_ok}")
print(f"Unique IDs     : {unique_ids}")
print()

if total == 100 and ranks_ok and unique_ids == 100:
    print("Submission looks good. Ready to submit!")
else:
    if total != 100:
        print(f"ERROR: Expected 100 rows, got {total}")
    if not ranks_ok:
        print("ERROR: Ranks are not exactly 1-100")
    if unique_ids != 100:
        print(f"ERROR: Duplicate candidate IDs found ({unique_ids} unique)")