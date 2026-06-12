"""Shared preprocessing for the Zigbee_UAD dataset.

Loads the three per-attack CSV files, converts every protocol field to a
numeric representation, derives causal traffic-context features and
returns a single labelled DataFrame.

Conventions
-----------
- Hexadecimal strings (0x...) are parsed as integers.
- Booleans are mapped to {0, 1}.
- Missing values are encoded with the sentinel -1 (a frame that lacks a
  layer simply does not carry that field).
- `frame.number` and `frame.time_epoch` encode the *position* of a frame
  inside the recording session.  Because attacks were executed in timed
  bursts, those fields reveal the labelling layout of the capture and
  must never be model inputs.  `frame.time_epoch` is used only to build
  the sliding windows of the context features.

Context features (all causal: computed from the current frame and the
frames that precede it within the same capture; only w5_dup counts
strictly earlier frames):
    w1_count   frames observed in the last 1 s
    w5_count   frames observed in the last 5 s
    w5_nsrc    distinct MAC short source addresses in the last 5 s
    w5_breq    Beacon Request commands (wpan.cmd = 0x07) in the last 5 s
    w5_areq    Association Request commands (wpan.cmd = 0x01) in the last 5 s
    w5_dup     earlier frames in the last 5 s with the same
               (src16, seq_no, length) triple as the current frame
"""

from collections import Counter, deque
from pathlib import Path

import numpy as np
import pandas as pd

DATA_FILES = {
    "association_flood": "zigbee_asociation_dataset.csv",
    "replay": "zigbee_replay_dataset.csv",
    "beacon_flood": "zigbee_beacon_dataset.csv",
}

LABEL_COL = "type"

# Capture-index artefacts: excluded from every model input.
LEAKY_COLS = ["frame.number", "frame.time_epoch"]

HEX_COLS = [
    "wpan.fcf", "wpan.frame_type", "wpan.dst_addr_mode", "wpan.src_addr_mode",
    "wpan.dst_pan", "wpan.dst16", "wpan.src16", "wpan.cmd", "wpan.fcs",
    "zbee_nwk.src", "zbee_nwk.dst", "zbee_nwk.src64", "zbee_nwk.dst64",
]

BOOL_COLS = [
    "wpan.security", "wpan.pending", "wpan.ack_request",
    "wpan.pan_id_compression", "wpan.fcs_ok", "zbee_nwk.security",
]

CONTEXT_COLS = ["w1_count", "w5_count", "w5_nsrc", "w5_breq", "w5_areq",
                "w5_dup"]


def _hex_to_int(value):
    if pd.isna(value):
        return np.nan
    try:
        return int(str(value), 16)
    except (ValueError, TypeError):
        return np.nan


def _add_context(part):
    """Causal sliding-window features for one capture (sorted by time)."""
    part = part.sort_values("frame.time_epoch").copy()
    t = part["frame.time_epoch"].to_numpy()
    src = part["wpan.src16"].to_numpy()
    seq = part["wpan.seq_no"].to_numpy()
    cmd = part["wpan.cmd"].to_numpy()
    ln = part["frame.len"].to_numpy()
    n = len(part)
    out = {k: np.zeros(n) for k in CONTEXT_COLS}
    win = deque()
    cnt_src, cnt_triple, cnt_cmd = Counter(), Counter(), Counter()
    start1 = 0
    for i in range(n):
        win.append(i)
        cnt_src[src[i]] += 1
        cnt_triple[(src[i], seq[i], ln[i])] += 1
        cnt_cmd[cmd[i]] += 1
        while t[i] - t[win[0]] > 5.0:
            j = win.popleft()
            cnt_src[src[j]] -= 1
            if cnt_src[src[j]] == 0:
                del cnt_src[src[j]]
            cnt_triple[(src[j], seq[j], ln[j])] -= 1
            cnt_cmd[cmd[j]] -= 1
        while t[i] - t[start1] > 1.0:
            start1 += 1
        out["w1_count"][i] = i - start1 + 1
        out["w5_count"][i] = len(win)
        out["w5_nsrc"][i] = len(cnt_src)
        out["w5_breq"][i] = cnt_cmd.get(7, 0)
        out["w5_areq"][i] = cnt_cmd.get(1, 0)
        out["w5_dup"][i] = cnt_triple[(src[i], seq[i], ln[i])] - 1
    for k, v in out.items():
        part[k] = v
    return part


def load_dataset(data_dir, context=True):
    """Return (df, protocol_cols, context_cols)."""
    data_dir = Path(data_dir)
    frames = []
    for scenario, fname in DATA_FILES.items():
        df = pd.read_csv(data_dir / fname, low_memory=False)
        df["scenario"] = scenario
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    df["label"] = (df[LABEL_COL] == "attack").astype(int)

    for col in HEX_COLS:
        df[col] = df[col].map(_hex_to_int)
    for col in BOOL_COLS:
        df[col] = df[col].map({True: 1, False: 0, "True": 1, "False": 0})
    df["frame.protocols"] = df["frame.protocols"].astype("category").cat.codes

    protocol_cols = [
        c for c in df.columns
        if c not in (LABEL_COL, "label", "scenario", *LEAKY_COLS)
    ]
    df[protocol_cols] = df[protocol_cols].apply(
        pd.to_numeric, errors="coerce").fillna(-1)

    context_cols = []
    if context:
        df = pd.concat([_add_context(p) for _, p in df.groupby("scenario")],
                       ignore_index=True)
        context_cols = CONTEXT_COLS
    return df, protocol_cols, context_cols


if __name__ == "__main__":
    import sys
    data = sys.argv[1] if len(sys.argv) > 1 else "../../../recursos/datasets_zigbee"
    df, proto, ctx = load_dataset(data)
    print(f"frames: {len(df)}  protocol features: {len(proto)}  context: {len(ctx)}")
    print(df.groupby("scenario")["label"].agg(["count", "sum"]))
    for c in proto + ctx:
        print(f"{c:28s} unique={df[c].nunique():6d} "
              f"coverage={(df[c] != -1).mean() * 100:5.1f}")
