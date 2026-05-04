## AI in mental health support data story mini project 2

This repo contains a reproducible Python analysis and a visual data story based on the file
AI_mental_health_interactions.csv which contains anonymized interactions with an AI mental health support assistant

### Quickstart

Install dependencies
pip install -r requirements.txt

Run step 1 data overview and cleaned data
python -m analysis.01_data_overview

Run step 2 emotions and follow up visuals
python -m analysis.02_emotions_and_followup

Run step 3 shifts transitions and risk tables
python -m analysis.03_shifts_transitions_risk

Build the story markdown and the plain text version
python -m analysis.04_build_story

Export the story to HTML
python -m analysis.05_export_html

Run everything end to end
python -m analysis.00_run_all

Export the full code appendix
python -m analysis.06_export_appendix

### Project structure

AI_mental_health_interactions.csv  input dataset
src/aihms  reusable loading and cleaning utilities
analysis  analysis steps that generate tables plots and narrative artifacts
outputs  generated cleaned data figures and story files

