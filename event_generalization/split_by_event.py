from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "CFND_dataset"
OUTPUT_DIR = Path(__file__).resolve().parent / "data"

TRAIN_EVENTS = [2, 9, 5, 6]
VAL_EVENTS = [3]
TEST_EVENTS = [0, 1, 4, 7, 8]


def load_all_rows():
    frames = []
    for split in ("train", "val", "test"):
        path = SOURCE_DIR / f"{split}_data.csv"
        df = pd.read_csv(path)
        df["source_split"] = split
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    if "event" not in data.columns:
        raise ValueError("The source CSV files must contain an 'event' column.")
    return data


def write_split(data, name, events):
    split_df = data[data["event"].isin(events)].copy()
    split_df = split_df.drop(columns=["source_split"], errors="ignore")
    out_path = OUTPUT_DIR / f"{name}_data.csv"
    split_df.to_csv(out_path, index=False, encoding="utf-8")
    return out_path, split_df


def summarize(name, df):
    rows = []
    for event_id, event_df in df.groupby("event"):
        label_counts = event_df["label"].value_counts().to_dict()
        rows.append(
            {
                "split": name,
                "event": int(event_id),
                "total": len(event_df),
                "real_label_0": int(label_counts.get(0, 0)),
                "fake_label_1": int(label_counts.get(1, 0)),
            }
        )
    label_counts = df["label"].value_counts().to_dict()
    rows.append(
        {
            "split": name,
            "event": "ALL",
            "total": len(df),
            "real_label_0": int(label_counts.get(0, 0)),
            "fake_label_1": int(label_counts.get(1, 0)),
        }
    )
    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_all_rows()

    split_specs = {
        "train": TRAIN_EVENTS,
        "val": VAL_EVENTS,
        "test": TEST_EVENTS,
    }

    summary_rows = []
    for name, events in split_specs.items():
        path, split_df = write_split(data, name, events)
        summary_rows.extend(summarize(name, split_df))
        print(f"{name}: events={events} rows={len(split_df)} -> {path}")

    summary = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "split_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"summary -> {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
