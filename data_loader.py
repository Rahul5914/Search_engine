"""
data_loader.py - Load and preprocess candidate data.
Works whether files are in root folder OR in data/ subfolder.
"""

import re
from pathlib import Path

import pandas as pd


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_section(text, keywords):
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
    return " ".join(result[:15])


def build_profile_text(row):
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


def _extract_name(text):
    if not text:
        return "Unknown"
    for line in str(text).strip().split("\n")[:3]:
        line = line.strip()
        if (line and len(line) < 60
                and not any(x in line.lower() for x in
                            ["email", "phone", "github", "linkedin", "[", "http", "@"])):
            return line
    return "Unknown"


def _extract_title(text):
    if not text:
        return ""
    title_keywords = {
        "engineer", "developer", "manager", "analyst", "scientist", "designer",
        "architect", "intern", "consultant", "lead", "head", "director",
        "specialist", "coordinator", "officer", "associate", "student",
    }
    for line in str(text).strip().split("\n")[1:6]:
        line = line.strip()
        if (line and len(line) < 100
                and not any(x in line.lower() for x in
                            ["[phone]", "[email]", "[github", "[linkedin", "http", "@", "•"])
                and any(k in line.lower() for k in title_keywords)):
            return line
    return ""


def _load_from_xlsx(xlsx_path):
    df1 = pd.read_excel(xlsx_path, sheet_name="candidate dump 1", engine="openpyxl")
    df2 = pd.read_excel(xlsx_path, sheet_name="candidate dump 2", engine="openpyxl")
    raw = pd.concat([df1, df2], ignore_index=True)
    records = []
    for _, row in raw.iterrows():
        text = str(row["Resume Text"]) if pd.notna(row.get("Resume Text")) else ""
        records.append({
            "name": _extract_name(text),
            "title": _extract_title(text),
            "resume_text": text,
        })
    df = pd.DataFrame(records)
    df = df[df["resume_text"].str.len() > 100].reset_index(drop=True)
    return df


def _finalise(df):
    for col in ["name", "title", "resume_text"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    df["profile_text"] = df.apply(build_profile_text, axis=1)
    df = df[df["profile_text"].str.len() > 30].reset_index(drop=True)
    return df


def load_candidates(csv_path):
    csv = Path(csv_path)

    # Try loading CSV from multiple locations
    csv_locations = [
        csv,
        Path(".") / csv.name,
        Path(".") / "data" / csv.name,
    ]
    for loc in csv_locations:
        if loc.exists():
            df = pd.read_csv(loc)
            return _finalise(df)

    # No CSV found — look for the xlsx file in root or data/ folder
    xlsx_names = ["Candidates_and_Jobs.xlsx", "candidates_and_jobs.xlsx"]
    search_dirs = [Path("."), Path(".") / "data", csv.parent]

    for d in search_dirs:
        for name in xlsx_names:
            candidate = d / name
            if candidate.exists():
                df = _load_from_xlsx(candidate)
                # Try to save CSV cache (best effort)
                try:
                    df.to_csv(csv, index=False)
                except Exception:
                    pass
                return _finalise(df)

    raise FileNotFoundError(
        "Could not find candidate data.\n"
        "Please make sure 'Candidates_and_Jobs.xlsx' is in your repository."
    )
