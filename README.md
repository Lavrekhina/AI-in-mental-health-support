## AI in mental health support — data story (Mini-project #2)

This repository contains a reproducible Python analysis + visual data story based on
`AI_mental_health_interactions.csv` (anonymized interactions with an AI mental-health support assistant).

### Quickstart

- **Create a virtual environment** (recommended) and install dependencies:

```bash
pip install -r requirements.txt
```

- **Run the first analysis step** (loads the dataset, normalizes labels, writes a cleaned CSV):

```bash
python -m analysis.01_data_overview
```

- **Run step 2** (emotion prevalence + wordcloud + response-vs-followup plots):

```bash
python -m analysis.02_emotions_and_followup
```

- **Run step 3** (before/after sentiment, emotion transitions, age-group shifts, at-risk group table):

```bash
python -m analysis.03_shifts_transitions_risk
```

- **Build the story artifact** (writes `outputs/story.md`):

```bash
python -m analysis.04_build_story
```

- **Export to HTML** (writes `outputs/story.html`, embedding figures + interactive charts):

```bash
python -m analysis.05_export_html
```

- **Run everything end-to-end** (recommended for final regeneration):

```bash
python -m analysis.00_run_all
```

### Project structure

- `AI_mental_health_interactions.csv`: raw input dataset (provided externally).
- `src/aihms/`: reusable loading/cleaning utilities.
- `analysis/`: executable analysis steps (scripts). Later steps will generate plots and the narrative story.
- `outputs/`: generated, non-sensitive artifacts (cleaned data + figures).

### Notes

- The code is written to be **schema-tolerant**: if your dataset version includes extra columns
  for "after" emotion/sentiment (e.g., `Sentiment_score_after`), the loader keeps them and downstream
  steps will use them when available.
