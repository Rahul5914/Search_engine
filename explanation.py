"""
explanation.py - Generate human-readable explanations for each search result
"""

import re

# ── Domain keyword banks ──────────────────────────────────────────────────────

SKILL_KEYWORDS = {
    # Languages
    "python", "java", "javascript", "typescript", "go", "golang", "rust",
    "c++", "c#", "ruby", "kotlin", "swift", "scala", "r", "sql",
    # Frameworks / libraries
    "react", "angular", "vue", "django", "flask", "fastapi", "spring",
    "node", "express", "tensorflow", "pytorch", "scikit-learn", "pandas",
    "spark", "kafka", "redis", "graphql", "grpc",
    # Cloud / infra
    "aws", "gcp", "azure", "docker", "kubernetes", "terraform", "ci/cd",
    "devops", "mlops", "linux",
    # Domains
    "machine learning", "deep learning", "nlp", "llm", "data science",
    "backend", "frontend", "full stack", "mobile", "ios", "android",
    "flutter", "microservices", "api", "database",
}

EXPERIENCE_KEYWORDS = {
    "senior", "lead", "principal", "staff", "architect", "head", "director",
    "vp", "years", "experience", "production", "deployed", "scaled",
    "startup", "enterprise", "founded", "intern",
}

DOMAIN_KEYWORDS = {
    "fintech", "healthtech", "edtech", "ecommerce", "saas", "b2b", "b2c",
    "banking", "finance", "healthcare", "retail", "logistics", "gaming",
    "security", "cybersecurity", "blockchain", "ai", "ml", "data",
}

LOCATION_KEYWORDS = {
    "remote", "india", "mumbai", "bangalore", "delhi", "hyderabad", "chennai",
    "pune", "usa", "uk", "canada", "singapore", "germany", "europe",
}


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b[a-z][a-z0-9+#./ ]*\b", text.lower()))


def _matched(tokens: set[str], bank: set[str]) -> list[str]:
    return sorted(tokens & bank)


def _extract_query_tokens(query: str) -> set[str]:
    return _tokenize(query)


def generate_explanation(query: str, candidate_row) -> str:
    """
    Return a concise, natural-language explanation for why a candidate
    matches the search query.
    """
    q_tokens = _extract_query_tokens(query)
    profile = str(candidate_row.get("profile_text", "")).lower()
    p_tokens = _tokenize(profile)

    skill_hits = _matched(q_tokens & p_tokens, SKILL_KEYWORDS)
    exp_hits = _matched(q_tokens & p_tokens, EXPERIENCE_KEYWORDS)
    domain_hits = _matched(q_tokens & p_tokens, DOMAIN_KEYWORDS)
    location_hits = _matched(q_tokens & p_tokens, LOCATION_KEYWORDS)

    # Also check what candidate has (even if not in query)
    candidate_skills = _matched(p_tokens, SKILL_KEYWORDS)
    candidate_exp = _matched(p_tokens, EXPERIENCE_KEYWORDS)
    candidate_domains = _matched(p_tokens, DOMAIN_KEYWORDS)

    parts = []

    # Skill match
    if skill_hits:
        parts.append(
            f"✅ **Skill match:** {', '.join(s.title() for s in skill_hits[:5])}"
        )
    elif candidate_skills:
        parts.append(
            f"🔧 **Key skills:** {', '.join(s.title() for s in candidate_skills[:5])}"
        )

    # Experience level
    if exp_hits:
        parts.append(
            f"💼 **Experience signals:** {', '.join(exp_hits[:3])}"
        )
    elif candidate_exp:
        parts.append(
            f"💼 **Experience level:** {', '.join(candidate_exp[:3])}"
        )

    # Domain match
    if domain_hits:
        parts.append(
            f"🏢 **Domain relevance:** {', '.join(d.title() for d in domain_hits[:3])}"
        )
    elif candidate_domains:
        parts.append(
            f"🏢 **Industry exposure:** {', '.join(d.title() for d in candidate_domains[:3])}"
        )

    # Title relevance
    title = str(candidate_row.get("title", "")).strip()
    if title:
        parts.append(f"🎯 **Role:** {title}")

    # Location match
    if location_hits:
        parts.append(
            f"📍 **Location match:** {', '.join(location_hits)}"
        )

    if not parts:
        parts.append(
            "📄 Candidate profile semantically aligns with the query context."
        )

    return "\n".join(parts)


def highlight_keywords(text: str, query: str) -> str:
    """
    Wrap query keywords found in `text` with a Streamlit-compatible
    HTML highlight span (used inside st.markdown).
    """
    if not query or not text:
        return text
    words = re.findall(r"\b\w+\b", query)
    for word in words:
        if len(word) < 3:
            continue
        pattern = re.compile(rf"\b({re.escape(word)})\b", re.IGNORECASE)
        text = pattern.sub(
            r'<mark style="background:#FDE68A;border-radius:3px;padding:1px 3px">\1</mark>',
            text,
        )
    return text
