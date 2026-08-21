# STP PRACTICAL TEST -- Online Exam Portal

An auto-generated, self-contained online exam portal built with
[Streamlit](https://streamlit.io). Students log in, wait (if needed) for
their batch's access window, sit a timed multiple-choice test, and get an
instant score -- all with **no external services, no database, and no
internet API calls** once it's running.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [How the Exam Flow Works](#how-the-exam-flow-works)
- [Batch Schedule](#batch-schedule)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Deploying to Streamlit Community Cloud](#deploying-to-streamlit-community-cloud)
- [Data & Privacy](#data--privacy)
- [Security Notes](#security-notes)
- [Troubleshooting](#troubleshooting)
- [Regenerating This Exam](#regenerating-this-exam)
- [License](#license)

## Overview

| | |
|---|---|
| **Course** | STP PRACTICAL TEST |
| **Questions** | 20 (4 options each) |
| **Students** | 30 across 1 batch(es), max 40 per batch |
| **Test date** | 2026-08-21 |
| **Duration** | 5 minutes per student |
| **Generated** | 2026-08-21 18:56 from `STP 222 TEST QUESTION.docx` |
| **Question source** | AI-drafted with gpt-oss:20b, reviewed/edited by the instructor before export |

## Features

- **Self-contained** -- questions, options, and answers are embedded directly
  in `exam_app.py`; no Ollama, no internet connection, and no database needed
  to run the exam itself.
- **Batch scheduling** -- students are split into time-boxed batches (max
  40 students each) so a large cohort doesn't all sit
  the test simultaneously.
- **Live waiting room** -- if a student logs in before their batch's window
  opens, they see an auto-refreshing waiting screen (every 10 seconds)
  instead of a dead-end error or a page they have to keep manually reloading.
- **Live countdown timer** -- the active test screen refreshes every 10
  seconds in the background (via `st.fragment`) so the timer stays accurate
  without freezing or reloading the whole page.
- **One attempt per student** -- re-submission with the same Student ID is
  blocked, both before and after a successful submit.
- **Randomized presentation** -- each student sees questions and answer
  options in a different shuffled order.
- **Admin dashboard** -- a password-gated panel to monitor attempts, review
  per-batch performance, browse raw results, and download a CSV.

## Requirements

- Python 3.10+
- Packages listed in `requirements.txt` (Streamlit + pandas)

## Quick Start

```bash
pip install -r requirements.txt
streamlit run exam_app.py
```

Open the local URL Streamlit prints (usually http://localhost:8501).

## How the Exam Flow Works

1. **Login** -- the student enters their name, Student ID, and selects their
   assigned batch from a dropdown.
2. **Waiting room (if early)** -- if their batch's window hasn't opened yet,
   they land on an auto-refreshing waiting screen rather than an error; the
   test starts automatically the moment the window opens, with no action
   needed from the student.
3. **Timed test** -- once their window is open, the full question set loads
   (order and option order shuffled per student) with a live countdown.
4. **Auto-submit** -- if time runs out, the test is submitted automatically
   with whatever answers were recorded.
5. **One-time result** -- the student sees their score immediately; their
   Student ID is then locked out of re-attempting.

## Batch Schedule

| Batch | Capacity | Window |
|---|---|---|
| Batch 1 | 30 | 20:30 - 20:55 |

*(All times are in the `Africa/Lagos` timezone -- see [Configuration](#configuration) to change it.)*

## Project Structure

```
exam_app.py         # The full exam portal (Streamlit app) -- run this
question_bank.json  # Reference copy of the questions/options/answers
                     # (also embedded directly inside exam_app.py)
requirements.txt     # Python dependencies
README.md            # This file
exam_results.csv     # Created automatically the first time a student submits
```

## Configuration

A few values can be changed without touching the question logic:

- **Admin password** -- baked into `exam_app.py` as a default, but you can
  override it at launch without editing the file:
  ```bash
  export ADMIN_PASSWORD="a-stronger-password"   # Mac/Linux
  setx ADMIN_PASSWORD "a-stronger-password"      # Windows (new terminal after)
  streamlit run exam_app.py
  ```
- **Timezone** -- edit the `TIMEZONE_NAME` constant near the top of
  `exam_app.py` (uses standard [IANA timezone names](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones),
  e.g. `"Africa/Lagos"`, `"America/New_York"`).
- **Auto-refresh interval** -- `AUTO_REFRESH_SECONDS` (default 10) controls
  both the waiting-room and active-test refresh rate.

## Deploying to Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo,
   and set the main file path to `exam_app.py`.
3. In the app's **Settings -> Secrets**, you can optionally set
   `ADMIN_PASSWORD` there instead of as a plain environment variable --
   Streamlit Cloud exposes Secrets as environment variables automatically.
4. Deploy. No other configuration is required.

## Data & Privacy

- Student results (name, Student ID, batch, score, timing) are written to a
  local `exam_results.csv` file in the app's working directory. On Streamlit
  Community Cloud, this file persists only for the life of that deployment's
  filesystem -- **download the CSV from the admin Reports tab periodically**
  if you need a permanent copy.
- No data is sent anywhere outside the app itself; there are no external API
  calls in `exam_app.py`.

## Security Notes

- This portal is designed for classroom-scale, low-to-moderate-stakes
  testing. It does not include anti-cheating measures like tab-switch
  detection, webcam proctoring, or IP restrictions.
- The admin password, if left at its baked-in default, is visible to anyone
  who can read the source file (e.g. in a public GitHub repo). Override it
  via the `ADMIN_PASSWORD` environment variable (see
  [Configuration](#configuration)) before using this for anything sensitive.
- The `exam_results.csv` file is plain text and unencrypted.

## Troubleshooting

- **"This exam is scheduled for [date], not today."** -- the server's system
  date doesn't match `TEST_DATE`. Check the deployment environment's clock,
  or update `TEST_DATE` in `exam_app.py`.
- **A student can't log in / batch window already closed** -- double-check
  they were given the correct batch name, and that `TIMEZONE_NAME` matches
  the timezone you scheduled the test in.
- **Waiting room seems stuck** -- it refreshes every `AUTO_REFRESH_SECONDS`
  (default 10s) automatically; nothing needs to be clicked. If it truly
  isn't updating, confirm the Streamlit version supports `st.fragment`
  (Streamlit 1.33+; `requirements.txt` already pins a version that does).
- **Duplicate-attempt errors for a student who never actually took it** --
  Student IDs are matched case-insensitively; check for a typo'd ID already
  present in `exam_results.csv`.

## Regenerating This Exam

This exported folder is a standalone snapshot. To generate a new/different
exam (new material, question count, batch setup, etc.), go back to the quiz
builder tool (`app.py` in the *generator* project, not this one) -- it does
not need this exported folder to run.

## License

MIT -- use, modify, and redistribute freely for your own courses.
