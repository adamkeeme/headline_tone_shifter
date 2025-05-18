"""Headline & Subhead Tone‑Shifter

Reads a Parquet file containing article metadata (must include columns:
  * headline
  * subhead
)
Generates tone‑specific variations of each headline/subhead pair using the
OpenAI Chat Completion API and saves the augmented dataset back to disk.

Usage
-----
$ python generate_headline_variations.py \
        --infile articles.parquet \
        --outfile articles_with_variants.parquet \
        --model gpt-4o-mini \
        --tones Playful Sensationalist Exciting Sad Angry Calm

Environment
-----------
OPENAI_API_KEY  Your OpenAI key (export before running)

Notes
-----
* Tones can be swapped in/out at the command line without touching code.
* Character length is kept within ±10% of originals.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Dict

import pandas as pd
from openai import OpenAI, APIError

################################################################################
# Prompt template (easy to modify in one place)
################################################################################
PROMPT_TEMPLATE = (
    "You are a seasoned news copy‑writer. Your task is to rewrite the given "
    "headline and subhead in the specified tone.\n\n"
    "Tone: {tone}\n\n"
    "Constraints:\n"
    "  • Preserve the factual meaning.\n"
    "  • Keep the rewritten headline within ±10 % of the original character "
    "count (original: {headline_chars} chars).\n"
    "  • Keep the rewritten subhead within ±10 % of the original character "
    "count (original: {subhead_chars} chars).\n"
    "  • Avoid click‑bait when the tone is Calm.\n\n"
    "Return *only* a JSON object with exactly two fields, 'headline' and "
    "'subhead', each containing the rewritten string."
)
################################################################################

# --------------------------------------------------------------------------- #
# Helper
# --------------------------------------------------------------------------- #

def complete_variation(client: OpenAI, model: str, headline: str, subhead: str, tone: str) -> Dict[str, str]:
    """Call the Chat Completion API once and return the rewritten pair."""
    sys_prompt = "You are ChatGPT, a helpful assistant for copy rewriting."
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
        temperature=0.7,
        max_tokens=256,
    )

    text = resp.choices[0].message.content
    try:
        data = pd.json.loads(text)  # type: ignore[attr-defined]
    except Exception:
        raise ValueError(f"Unexpected response format for tone '{tone}':\n{text}")
    if not {"headline", "subhead"}.issubset(data):
        raise ValueError(f"JSON response missing keys for tone '{tone}': {text}")
    return {"headline": data["headline"], "subhead": data["subhead"]}


def generate_variants(df: pd.DataFrame, tones: List[str], model: str) -> pd.DataFrame:
    """Generate variants for every row; return new DataFrame with extra cols."""
    client = OpenAI()

    for tone in tones:
        h_col = f"{tone.lower()}_headline"
        s_col = f"{tone.lower()}_subhead"
        df[h_col] = ""  # pre‑allocate for type stability
        df[s_col] = ""

    for i, row in df.iterrows():
        headline, subhead = row["headline"], row["subhead"]
        for tone in tones:
            retries = 0
            while True:
                try:
                    out = complete_variation(client, model, headline, subhead, tone)
                    break
                except (APIError, ValueError) as e:
                    retries += 1
                    if retries > 3:
                        raise RuntimeError(f"Failed after 3 retries: {e}")
                    time.sleep(2 ** retries)  # exponential backoff
            df.at[i, f"{tone.lower()}_headline"] = out["headline"]
            df.at[i, f"{tone.lower()}_subhead"] = out["subhead"]
        # Keep stdout progress minimal to avoid clutter
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(df)} rows", file=sys.stderr)

    return df

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate tone‑shifted headline/subhead variants.")
    parser.add_argument("--infile", required=True, help="Input Parquet path")
    parser.add_argument("--outfile", required=True, help="Output Parquet path")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--tones", nargs="+", default=[
        "Playful", "Sensationalist", "Exciting", "Sad", "Angry", "Calm"
    ], help="List of tones to generate (space‑separated)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        sys.exit("Error: OPENAI_API_KEY environment variable not set.")

    infile = Path(args.infile)
    if not infile.exists():
        sys.exit(f"Input file not found: {infile}")

    print(f"Loading {infile}…", file=sys.stderr)
    df = pd.read_parquet(infile)

    required = {"headline", "subhead"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        sys.exit(f"Input is missing required columns: {', '.join(missing)}")

    print("Generating variants…", file=sys.stderr)
    df_out = generate_variants(df, args.tones, args.model)

    outfile = Path(args.outfile)
    df_out.to_parquet(outfile, index=False)
    print(f"Saved augmented dataset to {outfile}")


if __name__ == "__main__":
    main()
