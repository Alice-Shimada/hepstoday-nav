"""
notify.py – Daily notification script for HEPSToday
---------------------------------------------------

This script fetches the latest High‑Energy Physics papers from the
HEPSToday data repositories, enriches them using a language model, and
sends personalized email digests to subscribers based on their
interests.  Subscribers define keywords and preferred authors in a
JSON file; if a paper matches either a keyword in the title or
summary, or lists a preferred author, it will be included in their
daily email.

Requirements
------------
The script requires the following environment variables to be set:

```
# OpenAI‑compatible API credentials
OPENAI_API_KEY   – API key for the language model
OPENAI_BASE_URL  – Base URL of the API (e.g., https://api.openai.com)

# Email sending credentials
SMTP_HOST        – Hostname of your SMTP server
SMTP_PORT        – Port number (e.g., 465 for SSL, 587 for STARTTLS)
SMTP_USER        – Username/login email for the SMTP server
SMTP_PASSWORD    – Password or app token for the SMTP server
SENDER_EMAIL     – Address that appears in the “From” header

# Notification settings
SUBSCRIBERS_FILE – Path to subscribers.json describing interest lists
TARGET_DATE      – (optional) YYYY‑MM‑DD date to fetch. Defaults to today
TIMEZONE         – (optional) IANA timezone (default Europe/Berlin)
```

Subscribers JSON Format
-----------------------
The subscribers file should contain a list of objects with these keys:

* `email`: recipient email address
* `keywords`: list of keywords (strings) that will be matched against
  extracted keywords from each paper. Matching is case‑insensitive.
* `authors`: list of author names (strings) to match against the
  authors list of each paper. Matching is case‑insensitive and
  performs substring search.

Example:

```json
[
  {
    "email": "alice@example.com",
    "keywords": ["quantum computing", "supersymmetry"],
    "authors": ["Witten", "Zhang"]
  },
  {
    "email": "bob@example.com",
    "keywords": ["collider", "neutrino"],
    "authors": []
  }
]
```

Sending Email
-------------
The script uses Python’s built‑in `smtplib` library to send emails.
`smtplib` is a standard module for communicating with SMTP servers and
supports both SSL/TLS (port 465) and STARTTLS (port 587) connections
【633963091985529†L253-L259】.  The SMTP connection is secured using
`ssl.create_default_context()`, and the message is assembled with
`email.mime.text.MIMEText`.

Author: OpenAI’s ChatGPT
"""

import json
import os
import ssl
import datetime
from typing import Dict, List, Optional
import requests
import zoneinfo
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def get_target_date() -> str:
    """Return the date to process in YYYY‑MM‑DD format.

    Uses the TARGET_DATE environment variable if provided, otherwise
    returns today’s date in the configured timezone.
    """
    tzname = os.getenv("TIMEZONE", "Europe/Berlin")
    tz = zoneinfo.ZoneInfo(tzname)
    target = os.getenv("TARGET_DATE")
    if target:
        # Validate format
        try:
            datetime.datetime.strptime(target, "%Y-%m-%d")
            return target
        except ValueError:
            raise ValueError(f"TARGET_DATE must be YYYY‑MM‑DD, got {target}")
    # Default to current date in the specified timezone
    return datetime.datetime.now(tz).strftime("%Y-%m-%d")


def fetch_jsonl(url: str) -> List[Dict]:
    """Fetch a JSONL file from a URL and return list of JSON objects."""
    resp = requests.get(url)
    resp.raise_for_status()
    lines = resp.text.strip().split("\n")
    data = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Skipping malformed JSON line: {e}")
    return data


def call_llm(api_base: str, api_key: str, prompt: str, model: str = "qwen-plus-latest") -> str:
    """Call an OpenAI‑compatible chat completion endpoint and return the response text.

    This function assumes the API follows the OpenAI Chat API format.
    Raises HTTPError if the response indicates failure.
    """
    url = api_base.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a professor in High Energy Physics and Mathematical Physics. And you are a helpful assistant that extracts topics and keywords."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 64,
        "temperature": 0.1,
        "n": 1,
    }
    resp = requests.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected response format: {data}") from e


def extract_keywords(summary: str, api_base: str, api_key: str) -> List[str]:
    """Use LLM to extract a list of keywords from the provided summary.

    The returned keywords are normalized to lowercase and stripped.
    If the API call fails, returns an empty list.
    """
    prompt = (
        "Please extract keywords from the following paper summary. "
        "Return a concise list of significant terms and phrases (in English) "
        "separated by commas.\n\n"
        f"Summary: {summary}\n\nKeywords:"
    )
    try:
        response = call_llm(api_base, api_key, prompt)
    except Exception as e:
        print(f"Warning: failed to extract keywords: {e}")
        return []
    # Split on commas and normalize
    keywords = [kw.strip().lower() for kw in response.split(",") if kw.strip()]
    return keywords


def load_subscribers(path: str) -> List[Dict]:
    """Load subscriber definitions from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Subscribers file must contain a list of objects")
    return data


def match_paper_to_subscriber(paper: Dict, keywords: List[str], subscriber: Dict) -> bool:
    """Determine whether a paper matches the subscriber’s interests.

    A match occurs if any of the subscriber’s keywords are found in the
    extracted keywords list or if any of the subscriber’s authors are
    substrings of any author in the paper’s `authors` list.
    """
    sub_keywords = [kw.lower() for kw in subscriber.get("keywords", [])]
    sub_authors = [a.lower() for a in subscriber.get("authors", [])]
    # Keyword match: any intersection between subscriber keywords and extracted keywords
    for kw in sub_keywords:
        for extracted in keywords:
            if kw in extracted:
                return True
    # Author match: check if any subscriber author appears in any paper author string
    paper_authors = paper.get("authors", [])
    if isinstance(paper_authors, list):
        author_strings = [a.lower() for a in paper_authors]
    else:
        author_strings = [str(paper_authors).lower()]
    for sub_author in sub_authors:
        for auth in author_strings:
            if sub_author in auth:
                return True
    return False


def compose_email(date: str, matches: List[Dict]) -> str:
    """Compose the body of the email based on matched papers."""
    lines = [f"HEPSToday Daily Digest for {date}", ""]
    if not matches:
        lines.append("No articles matched your interests today.")
    else:
        for idx, paper in enumerate(matches, start=1):
            lines.append(f"{idx}. {paper['title']}")
            lines.append(f"   Authors: {', '.join(paper['authors'])}")
            if paper.get("categories"):
                lines.append(f"   Categories: {', '.join(paper['categories'])}")
            # Include summary or AI tldr if available
            summary = paper.get("summary") or paper.get("AI", {}).get("tldr")
            if summary:
                lines.append(f"   Summary: {summary}")
            # Provide link to PDF or details page
            pdf_link = paper.get("pdf")
            if pdf_link:
                lines.append(f"   PDF: {pdf_link}")
            lines.append("")
    return "\n".join(lines)


def send_email(smtp_host: str, smtp_port: int, smtp_user: str, smtp_password: str,
               sender: str, recipient: str, subject: str, body: str) -> None:
    """Send an email with the specified subject and body."""
    # Build MIME message
    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))
    # Establish secure connection using SMTP_SSL as recommended【633963091985529†L253-L259】
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_host, smtp_port, context=context) as server:
        if smtp_user and smtp_password:
            server.login(smtp_user, smtp_password)
        server.sendmail(sender, [recipient], msg.as_string())


def main() -> None:
    date_str = get_target_date()
    # Build URLs for English and Chinese AI‑enhanced data
    base_en = "https://raw.githubusercontent.com/Alice-Shimada/hepstoday-en/data/data"
    base_cn = "https://raw.githubusercontent.com/Alice-Shimada/hepstoday-cn/data/data"
    en_url = f"{base_en}/{date_str}_AI_enhanced_English.jsonl"
    cn_url = f"{base_cn}/{date_str}_AI_enhanced_Chinese.jsonl"
    # Fetch papers for both languages
    papers: List[Dict] = []
    for url in [en_url, cn_url]:
        try:
            papers += fetch_jsonl(url)
        except Exception as e:
            print(f"Warning: failed to fetch data from {url}: {e}")
    if not papers:
        print(f"No papers found for {date_str}.")
        return
    # Load subscribers
    subscribers_path = os.getenv("SUBSCRIBERS_FILE", "subscribers.json")
    subscribers = load_subscribers(subscribers_path)
    # LLM credentials
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_BASE_URL")
    if not api_key or not api_base:
        raise EnvironmentError("OPENAI_API_KEY and OPENAI_BASE_URL must be set.")
    # Email credentials
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port_str = os.getenv("SMTP_PORT") or "465"
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_password = os.getenv("SMTP_PASSWORD", "")
    sender_email = os.getenv("SENDER_EMAIL") or smtp_user
    try:
        smtp_port = int(smtp_port_str)
    except ValueError:
        raise ValueError("SMTP_PORT must be an integer")
    # Precompute keywords for each paper
    paper_keywords: Dict[int, List[str]] = {}
    for idx, paper in enumerate(papers):
        summary = paper.get("summary")
        if not summary:
            # Use AI tldr or abstract
            summary = paper.get("AI", {}).get("tldr") or paper.get("abs", "")
        # Extract keywords via LLM
        kws = extract_keywords(summary, api_base, api_key) if summary else []
        paper_keywords[idx] = kws
    # For each subscriber, collect matching papers
    for subscriber in subscribers:
        matches: List[Dict] = []
        for idx, paper in enumerate(papers):
            kws = paper_keywords.get(idx, [])
            if match_paper_to_subscriber(paper, kws, subscriber):
                matches.append(paper)
        # Compose and send email
        subject = f"HEPSToday Digest – {date_str}"
        body = compose_email(date_str, matches)
        recipient = subscriber.get("email")
        if not recipient:
            print(f"Skipping subscriber with missing email: {subscriber}")
            continue
        try:
            send_email(
                smtp_host=smtp_host,
                smtp_port=smtp_port,
                smtp_user=smtp_user,
                smtp_password=smtp_password,
                sender=sender_email,
                recipient=recipient,
                subject=subject,
                body=body,
            )
            print(f"Sent digest to {recipient}, {len(matches)} matches")
        except Exception as e:
            print(f"Failed to send email to {recipient}: {e}")


if __name__ == "__main__":
    main()