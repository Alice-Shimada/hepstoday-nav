# HEPSToday Notifier

This repository contains a GitHub Actions workflow and accompanying
scripts for sending daily personalized email digests of the latest
High‑Energy Physics papers from the HEPSToday project.  The workflow
fetches the AI‑enhanced data from the existing `hepstoday-cn` and
`hepstoday-en` repositories, extracts topics and keywords using a
language model API, matches papers against subscriber interests, and
emails a summary to each subscriber.

## Features

- **Daily schedule** – runs automatically every morning (06:00 UTC) via
  GitHub Actions.
- **Subscriber matching** – matches papers based on user‑defined
  keywords and preferred authors.
- **LLM integration** – uses an OpenAI‑compatible API to extract
  keywords from paper summaries.
- **Secure email sending** – sends emails via a configurable SMTP
  server using TLS encryption as recommended in the Python
  documentation【633963091985529†L253-L259】.

## Files

- `notify.py` – main Python script that performs the data fetch,
  keyword extraction, matching and email sending.
- `subscribers.json` – example subscriber definitions.  Replace with
  your actual subscriber list.
- `.github/workflows/daily_email.yml` – GitHub Actions workflow to
  schedule the script daily.

## Setting up

1. **Create a new repository**.  Initialize a new repository on
   GitHub (private if your subscriber list is sensitive) and copy the
   contents of this folder into it.

2. **Configure secrets**.  Go to the repository’s *Settings →
   Secrets → Actions* and add the following repository secrets:

   | Secret name        | Description                                               |
   |--------------------|-----------------------------------------------------------|
   | `OPENAI_API_KEY`   | API key for your OpenAI‑compatible service               |
   | `OPENAI_BASE_URL`  | Base URL of the API endpoint (e.g. `https://api.openai.com`)|
   | `SMTP_HOST`        | Hostname of your SMTP server (e.g. `smtp.gmail.com`)     |
   | `SMTP_PORT`        | Port number (465 for SSL or 587 for STARTTLS)            |
   | `SMTP_USER`        | Username/login for the SMTP server                       |
   | `SMTP_PASSWORD`    | Password or app token for the SMTP server                |
   | `SENDER_EMAIL`     | Email address that appears in the “From” header          |

   These secrets are injected into the workflow environment and used
   by `notify.py` when sending emails.  For security, do **not**
   hardcode any credentials in your repository.

3. **Update subscribers**.  Edit `subscribers.json` with the email
   addresses, keywords and authors for each of your subscribers.  See
   the file for an example format.  You can rename the file and set
   the `SUBSCRIBERS_FILE` environment variable accordingly in the
   workflow if you prefer to keep subscriber data outside of your
   repository.

4. **Commit and push**.  Commit the files to your repository and push
   to GitHub.  The workflow will run automatically at the next
   scheduled time.  You can also trigger it manually from the Actions
   tab.

## How it works

When executed, `notify.py` performs the following steps:

1. **Determine the date** – uses the `TARGET_DATE` environment
   variable or defaults to today’s date in the specified timezone.
2. **Fetch data** – downloads the AI‑enhanced JSONL files for the date
   from both the English and Chinese HEPSToday repositories.
3. **Extract keywords** – for each paper, sends the summary to the
   language model API and receives a list of keywords.  The script
   normalizes these keywords to lowercase.
4. **Match interests** – compares the extracted keywords and author
   names against each subscriber’s interests.  A paper matches if any
   subscriber keyword appears within the extracted keywords or if any
   author name contains a subscriber’s preferred author substring.
5. **Compose emails** – for each subscriber, composes a plain text
   email summarizing the matching papers.  If there are no matches, the
   email indicates that nothing matched that day.
6. **Send emails** – uses `smtplib.SMTP_SSL` to connect securely to
   your SMTP server and sends the email.  The Real Python tutorial on
   sending emails recommends using `SMTP_SSL` for a secure connection
  【633963091985529†L253-L259】.

Feel free to adapt the script to your needs – for example, adjust the
prompt used for keyword extraction, change the email format to HTML, or
support additional matching criteria.

## Troubleshooting

- If the workflow fails, check the *Actions* tab for logs.  Common
  issues include missing secrets, invalid API credentials or SMTP
  connection errors.
- Make sure the GitHub Actions runner can reach your SMTP server.  Some
  providers (e.g., Gmail) require enabling “less secure apps” or using
  an app password.
- To test locally, install the dependencies (`pip install requests`)
  and run `python notify.py` with the appropriate environment
  variables set.

## License

This project is licensed under the MIT License.  See
`LICENSE` for details.