"""
data_loader.py - Load and preprocess candidate data
"""

import re
import pandas as pd


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\[.*?\]", "", text)        # remove placeholder tokens like [EMAIL]
    text = re.sub(r"http\S+", "", text)         # strip URLs
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)   # remove control chars
    text = re.sub(r"\s+", " ", text)            # collapse whitespace
    return text.strip()


def extract_section(text: str, keywords: list[str]) -> str:
    """Pull lines near a section header (e.g. SKILLS, EXPERIENCE)."""
    lines = text.split("\n")
    result, capture = [], False
    stop_words = {"education", "projects", "achievements", "certifications",
                  "awards", "languages", "references", "volunteer", "honors"}
    for line in lines:
        lower = line.lower().strip()
        if any(k in lower for k in keywords):
            capture = True
            continue
        if capture:
            if any(s in lower for s in stop_words) and lower not in keywords:
                break
            result.append(line)
    return " ".join(result[:15])        # cap to first 15 lines


def build_profile_text(row: pd.Series) -> str:
    """Combine key parts of a resume into a single searchable string."""
    resume = str(row.get("resume_text", ""))
    parts = [
        str(row.get("name", "")),
        str(row.get("title", "")),
        extract_section(resume, ["skills", "technical skills", "tech stack"]),
        extract_section(resume, ["experience", "work experience", "employment"]),
        extract_section(resume, ["education"]),
        extract_section(resume, ["projects", "key projects"]),
        extract_section(resume, ["summary", "profile", "objective", "about"]),
    ]
    return clean_text(" | ".join(p for p in parts if p.strip()))


def load_candidates(csv_path: str) -> pd.DataFrame:
    """Load CSV, fill missing values, and build profile_text column."""
    df = pd.read_csv(csv_path)

    for col in ["name", "title", "resume_text"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    df["profile_text"] = df.apply(build_profile_text, axis=1)

    # Drop rows with no usable text
    df = df[df["profile_text"].str.len() > 30].reset_index(drop=True)
    return df
