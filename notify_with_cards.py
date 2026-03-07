"""HEPSToday notifier that renders each paper as an image card.

This script extends the original notification workflow by drawing a
compact, visually appealing “card” for each paper.  Each card shows
the title, authors, categories and summary (with LaTeX formulas
rendered inline) in a single image.  The images are attached to the
email and referenced from the HTML body.  Clients that support HTML
will display the images; plain text clients will still receive a
fallback text summary.

The approach uses matplotlib’s text and math rendering capabilities
to format the content.  Each card is rendered on a separate figure
with dynamic height based on the amount of text.  The script
otherwise follows the original logic: fetch data, extract keywords,
match to subscribers, and send individualized emails.

"""

import json
import os
import ssl
import datetime
import io
import textwrap
from typing import Dict, List, Optional, Tuple
import requests
import zoneinfo
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from matplotlib import pyplot as plt


def get_target_date() -> str:
    """Return the date to process in YYYY‑MM‑DD format."""
    tzname = os.getenv("TIMEZONE", "Europe/Berlin")
    tz = zoneinfo.ZoneInfo(tzname)
    target = os.getenv("TARGET_DATE")
    if target:
        try:
            datetime.datetime.strptime(target, "%Y-%m-%d")
            return target
        except ValueError:
            raise ValueError(f"TARGET_DATE must be YYYY‑MM‑DD, got {target}")
    return datetime.datetime.now(tz).strftime("%Y-%m-%d")


def fetch_jsonl(url: str) -> List[Dict]:
    """Fetch a JSONL file from a URL and return list of JSON objects."""
    resp = requests.get(url)
    resp.raise_for_status()
    lines = resp.text.strip().split("\n")
    data: List[Dict] = []
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
    """Call an OpenAI‑compatible chat completion endpoint and return the response text."""
    url = api_base.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a professor in High Energy Physics, and also a helpful assistant that extracts passage topics and keywords."},
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
    """Use LLM to extract a list of keywords from the provided summary."""
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
    """Determine whether a paper matches the subscriber’s interests."""
    sub_keywords = [kw.lower() for kw in subscriber.get("keywords", [])]
    sub_authors = [a.lower() for a in subscriber.get("authors", [])]
    for kw in sub_keywords:
        for extracted in keywords:
            if kw in extracted:
                return True
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


def render_paper_card(paper: Dict) -> bytes:
    """Render a paper summary as a PNG image.

    The card contains the title (bold, larger font), authors, categories
    and the summary text.  Inline formulas delimited by $...$ are
    rendered using matplotlib’s mathtext engine.  The figure size is
    computed based on the number of lines after wrapping the summary.
    """
    title = paper.get("title", "")
    authors = paper.get("authors") or []
    author_line = f"Authors: {', '.join(authors)}" if authors else ""
    categories = paper.get("categories") or []
    category_line = f"Categories: {', '.join(categories)}" if categories else ""
    summary = paper.get("summary") or paper.get("AI", {}).get("tldr") or ""
    # Wrap the summary for a reasonable line length
    wrapped_summary = textwrap.fill(summary, width=80)
    # Assemble lines; empty strings are filtered out later
    lines: List[str] = []
    if title:
        lines.append(title)
        lines.append("")
    if author_line:
        lines.append(author_line)
    if category_line:
        lines.append(category_line)
    if author_line or category_line:
        lines.append("")
    if wrapped_summary:
        lines.append(wrapped_summary)
    # Remove trailing empty lines
    while lines and lines[-1] == "":
        lines.pop()
    # Determine figure size: 0.35 inch per line, min 2 inches height
    n_lines = sum(len(l.split("\n")) for l in lines)
    height = max(2, 0.35 * n_lines)
    width = 6  # inches
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    # Draw text with math rendering; join lines with newline
    text = "\n".join(lines)
    ax.text(0.01, 0.99, text, va='top', ha='left', fontsize=10, wrap=True)
    # Save to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    return buf.getvalue()


def compose_email_plain(date: str, matches: List[Dict]) -> str:
    """Compose the plain text body of the email."""
    lines = [f"HEPSToday Daily Digest for {date}", ""]
    if not matches:
        lines.append("No articles matched your interests today.")
    else:
        for idx, paper in enumerate(matches, start=1):
            lines.append(f"{idx}. {paper['title']}")
            authors = paper.get("authors") or []
            if authors:
                lines.append(f"   Authors: {', '.join(authors)}")
            categories = paper.get("categories") or []
            if categories:
                lines.append(f"   Categories: {', '.join(categories)}")
            summary = paper.get("summary") or paper.get("AI", {}).get("tldr")
            if summary:
                lines.append(f"   Summary: {summary}")
            pdf_link = paper.get("pdf")
            if pdf_link:
                lines.append(f"   PDF: {pdf_link}")
            lines.append("")
    return "\n".join(lines)


def compose_email_html(date: str, matches: List[Dict]) -> Tuple[str, List[Tuple[str, bytes]]]:
    """Compose the HTML body and list of image attachments for the email."""
    attachments: List[Tuple[str, bytes]] = []
    parts: List[str] = [f"<h2>HEPSToday Daily Digest for {date}</h2>"]
    if not matches:
        parts.append("<p>No articles matched your interests today.</p>")
    else:
        for idx, paper in enumerate(matches, start=1):
            try:
                img_data = render_paper_card(paper)
            except Exception as e:
                print(f"Warning: failed to render card for paper '{paper.get('title', '')}': {e}")
                continue
            cid = f"card{len(attachments)}"
            attachments.append((cid, img_data))
            parts.append(f"<p><img src=\"cid:{cid}\" alt=\"Paper {idx}\" style=\"max-width:100%; height:auto;\"></p>")
    html_body = "\n".join(parts)
    return html_body, attachments


def send_email(
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str,
    sender: str,
    recipient: str,
    subject: str,
    body_text: str,
    body_html: Optional[str] = None,
    attachments: Optional[List[Tuple[str, bytes]]] = None,
) -> None:
    """Send an email with plain text, HTML and image attachments."""
    msg = MIMEMultipart('related')
    msg['From'] = sender
    msg['To'] = recipient
    msg['Subject'] = subject
    alt = MIMEMultipart('alternative')
    alt.attach(MIMEText(body_text, 'plain', 'utf-8'))
    if body_html:
        alt.attach(MIMEText(body_html, 'html', 'utf-8'))
    msg.attach(alt)
    if attachments:
        for cid, data in attachments:
            img = MIMEImage(data, 'png')
            img.add_header('Content-ID', f'<{cid}>')
            img.add_header('Content-Disposition', 'inline', filename=f'{cid}.png')
            msg.attach(img)
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_host, smtp_port, context=context) as server:
        if smtp_user and smtp_password:
            server.login(smtp_user, smtp_password)
        server.sendmail(sender, [recipient], msg.as_string())


def main() -> None:
    date_str = get_target_date()
    base_en = "https://raw.githubusercontent.com/Alice-Shimada/hepstoday-en/data/data"
    en_url = f"{base_en}/{date_str}_AI_enhanced_English.jsonl"
    papers: List[Dict] = []
    for url in [en_url]:
        try:
            papers += fetch_jsonl(url)
        except Exception as e:
            print(f"Warning: failed to fetch data from {url}: {e}")
    if not papers:
        print(f"No papers found for {date_str}.")
        return
    print(f"Fetched {len(papers)} papers from data sources.")
    subscribers_path = os.getenv("SUBSCRIBERS_FILE", "subscribers.json")
    subscribers = load_subscribers(subscribers_path)
    print(f"Loaded {len(subscribers)} subscriber(s) from {subscribers_path}.")
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_BASE_URL")
    if not api_key or not api_base:
        raise EnvironmentError("OPENAI_API_KEY and OPENAI_BASE_URL must be set.")
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port_str = os.getenv("SMTP_PORT") or "465"
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_password = os.getenv("SMTP_PASSWORD", "")
    sender_email = os.getenv("SENDER_EMAIL") or smtp_user
    try:
        smtp_port = int(smtp_port_str)
    except ValueError:
        raise ValueError("SMTP_PORT must be an integer")
    paper_keywords: Dict[int, List[str]] = {}
    total_papers = len(papers)
    print(f"Processing {total_papers} papers for {date_str}...")
    for idx, paper in enumerate(papers):
        summary = paper.get("summary")
        if not summary:
            summary = paper.get("AI", {}).get("tldr") or paper.get("abs", "")
        print(f"  Extracting keywords for paper {idx + 1}/{total_papers}")
        kws = extract_keywords(summary, api_base, api_key) if summary else []
        paper_keywords[idx] = kws
    print("Keyword extraction complete.")
    total_subs = len(subscribers)
    for sub_idx, subscriber in enumerate(subscribers):
        recipient = subscriber.get("email") or "(unknown)"
        print(f"Processing subscriber {sub_idx + 1}/{total_subs}: {recipient}")
        matches: List[Dict] = []
        for idx, paper in enumerate(papers):
            kws = paper_keywords.get(idx, [])
            if match_paper_to_subscriber(paper, kws, subscriber):
                matches.append(paper)
        subject = f"HEPSToday Digest – {date_str}"
        plain_body = compose_email_plain(date_str, matches)
        html_body, attachments = compose_email_html(date_str, matches)
        if not subscriber.get("email"):
            print(f"Skipping subscriber with missing email: {subscriber}")
            continue
        try:
            send_email(
                smtp_host=smtp_host,
                smtp_port=smtp_port,
                smtp_user=smtp_user,
                smtp_password=smtp_password,
                sender=sender_email,
                recipient=subscriber["email"],
                subject=subject,
                body_text=plain_body,
                body_html=html_body,
                attachments=attachments,
            )
            print(f"  Sent digest to {subscriber['email']}, {len(matches)} matches")
        except Exception as e:
            print(f"  Failed to send email to {subscriber['email']}: {e}")
    print("Processing complete.")


if __name__ == "__main__":
    main()