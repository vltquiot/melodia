import sys

def count_lines(filename: str) -> int:
    with open(filename, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python count_jsonl.py <file.jsonl>")
        sys.exit(1)

    filename = sys.argv[1]
    try:
        count = count_lines(filename)
        print(f"{filename} contains {count} lines")
    except FileNotFoundError:
        print(f"File not found: {filename}")
