"""Upload generated Q-A pairs to a Google Sheet for manual accept/reject review.

The reviewed sheet is what gets exported back to ``data/qa_pairs/qa_review.json``,
which is the question bank the retrieval experiments draw on.

Requires two values that are specific to your own Google account, so both are
read from the environment rather than hardcoded:

    GSPREAD_SHEET_KEY    the target spreadsheet id (from its URL)
    GSPREAD_CREDENTIALS  path to your OAuth client-secret JSON

See https://docs.gspread.org/en/latest/oauth2.html for how to obtain them.
"""
import json, gspread, os, sys
from google.auth.exceptions import RefreshError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

FILE_JSON  = os.getenv("QA_PAIRS_FILE", os.path.join(paths.QA_PAIRS_DIR, "pairs_set1.jsonl"))
SHEET_KEY  = os.getenv("GSPREAD_SHEET_KEY")
HEADER     = ["question", "answer", "question_type", "difficulty", "source_file"]
CREDENTIALS_FILE = os.getenv("GSPREAD_CREDENTIALS", "client_secret.json")

def iter_rows(path):
    def to_row(qa):
        md = qa.get("metadata", {})
        return [
            qa.get("question", ""),
            qa.get("answer", ""),
            md.get("question_type", ""),
            md.get("difficulty", ""),
            qa.get("source_file", ""),
        ]

    # Read entire file first (to allow sniffing and fallback parsing)
    with open(path, encoding="utf-8-sig") as f:
        content = f.read()

    rows = []
    stripped = content.lstrip()

    # If it looks like full JSON (object or array), parse as JSON regardless of extension
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            data = json.loads(content)
            items = data if isinstance(data, list) else data.get("qa_pairs", [])
            for qa in items:
                rows.append(to_row(qa))
        except json.JSONDecodeError as e:
            print(f"❌ JSON parse error for {path}: {e}")
            return []
    else:
        # Treat as JSONL (one object per line), with some tolerance
        for i, raw_line in enumerate(content.splitlines(), 1):
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # Fallback: support single quotes / Python dicts via ast.literal_eval
                import ast
                try:
                    obj = ast.literal_eval(line)
                except Exception as e:
                    print(f"❌ Skipping malformed JSONL at line {i}: {e}\n  Line: {line[:200]}")
                    continue
            rows.append(to_row(obj))

    for r in rows:
        yield r

def main():
    if not SHEET_KEY:
        print("❌ GSPREAD_SHEET_KEY not set — put your spreadsheet id in .env")
        return
    try:
        gc = gspread.oauth(
            credentials_filename=CREDENTIALS_FILE,
            authorized_user_filename=".gspread_token.json",
        )
    except RefreshError:
        print("⚠️ OAuth refresh failed (invalid_grant). Delete the cached token and try again:\n  rm -f ~/.config/gspread/authorized_user.json\nThen re-run this script to complete a fresh login.")
        return
    sh = gc.open_by_key(SHEET_KEY).sheet1
    if sh.row_values(1) != HEADER:
        sh.insert_row(HEADER, 1)
    data = list(iter_rows(FILE_JSON))
    sh.append_rows(data, value_input_option="RAW")
    print(f"✅ uploaded {len(data)} rows")

if __name__ == "__main__":
    main()