"""Headline & Subhead Tone‑Shifter

**Graceful 429 handling** – automatically waits the number of seconds
  suggested in the API error (or a fallback) and retries, unlimited times.
**Checkpointing** – writes a temp Parquet every N rows so you can resume if
  the job is interrupted.
**CLI flags** for rate‑limit sleep floor/ceiling and checkpoint frequency.

Run example
-----------
```
python run.py \
  --infile articles.parquet \
  --outfile output/articles_with_variants.parquet \
  --tones Playful Sensationalist Exciting Sad Angry Calm \
  --checkpoint-every 100  # save progress every 100 rows
```

See `--help` for all options.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from openai import OpenAI, APIError, RateLimitError

# ──────────────────────────────────────────────────────────────────────────────
# Prompt template
# ──────────────────────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = (
    "You are an award‑winning news copy editor. Rewrite the headline and "
    "subhead in the requested tone.\n\n"
    "Tone: {tone}\n\n"
    "Rules:\n"
    "  • Preserve the facts.\n"
    "  • Keep the rewritten **headline** within ±10 % of {headline_chars} chars.\n"
    "  • Keep the rewritten **subhead** within ±10 % of {subhead_chars} chars.\n"
    "  • If tone is Calm, avoid sensationalism or click‑bait.\n\n"
    "Return exactly one JSON object – no markdown fences – \n"
    "with keys: headline, subhead."
)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

JSON_RE = re.compile(r"\{[\s\S]*?\}")


def _extract_json(text: str) -> Dict[str, str]:
    match = JSON_RE.search(text)
    if not match:
        raise ValueError("No JSON object found in model response")
    return json.loads(match.group(0))


def _parse_retry_secs(message: str) -> Optional[int]:
    """Return seconds to wait, if message contains 'try again in Xs'."""
    m = re.search(r"try again in ([0-9.]+)s", message)
    return int(float(m.group(1))) if m else None


def complete_variation(
    client: OpenAI,
    model: str,
    headline: str,
    subhead: str,
    tone: str,
    temp: float = 0.7,
    max_tokens: int = 256,
) -> Dict[str, str]:
    sys_prompt = "You are ChatGPT – a meticulous rewriting assistant."
    user_prompt = PROMPT_TEMPLATE.format(
        tone=tone,
        headline_chars=len(headline),
        subhead_chars=len(subhead),
    ) + f"\n\nOriginal headline: {headline}\nOriginal subhead: {subhead}"

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temp,
        max_tokens=max_tokens,
    )
    return _extract_json(resp.choices[0].message.content)


# ──────────────────────────────────────────────────────────────────────────────
# Core loop
# ──────────────────────────────────────────────────────────────────────────────

def generate_variants(
    df: pd.DataFrame,
    tones: List[str],
    model: str,
    checkpoint_every: int,
    outfile: Path,
    sleep_floor: int,
) -> pd.DataFrame:
    client = OpenAI()

    for tone in tones:
        df[f"{tone.lower()}_headline"] = ""
        df[f"{tone.lower()}_subhead"] = ""

    total = len(df)
    for i, row in df.iterrows():
        for tone in tones:
            while True:
                try:
                    out = complete_variation(
                        client, model, row["headline"], row["subhead"], tone
                    )
                    df.at[i, f"{tone.lower()}_headline"] = out["headline"]
                    df.at[i, f"{tone.lower()}_subhead"] = out["subhead"]
                    break  # success
                except RateLimitError as exc:
                    wait = _parse_retry_secs(str(exc)) or sleep_floor
                    print(f"⚠︎ 429 – sleeping {wait}s", file=sys.stderr)
                    time.sleep(wait)
                    continue  # retry same call
                except (APIError, ValueError) as exc:
                    # Let user know and bail – could downgrade to empty string instead.
                    raise RuntimeError(f"Row {i}, tone {tone}: {exc}")
        if (i + 1) % checkpoint_every == 0:
            tmp = outfile.with_suffix(".partial.parquet")
            df.to_parquet(tmp, index=False)
            print(f"⏩ checkpoint @ {i + 1}/{total} → {tmp}", file=sys.stderr)
    return df

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--infile", required=True)
    p.add_argument("--outfile", required=True)
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument(
        "--tones",
        nargs="+",
        default=["Playful", "Sensationalist", "Exciting", "Sad", "Angry", "Calm"],
    )
    p.add_argument("--checkpoint-every", type=int, default=100, help="Rows between temp saves")
    p.add_argument("--sleep-floor", type=int, default=15, help="Min seconds to sleep on 429 without hint")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if "OPENAI_API_KEY" not in os.environ:
        sys.exit("Error: OPENAI_API_KEY not set.")

    df = pd.read_parquet(args.infile)
    if not {"headline", "subhead"}.issubset(df.columns):
        sys.exit("Input must have 'headline' and 'subhead' columns.")

    out_path = Path(args.outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_out = generate_variants(
        df.copy(),
        args.tones,
        args.model,
        args.checkpoint_every,
        out_path,
        args.sleep_floor,
    )
    df_out.to_parquet(out_path, index=False)
    print(f"✓ All done – saved {out_path}")


if __name__ == "__main__":
    main()
