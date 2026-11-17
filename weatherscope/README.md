# WeatherScope

An interactive Streamlit dashboard for exploring 25 years of daily weather in 10 Indian
cities, plus a machine-learning experiment: **forecasting tomorrow's maximum temperature**
from recent observations, evaluated honestly against simple baselines.

## What it does

|Page|Purpose|
|-|-|
|**Overview**|Dataset summary, what the model uses and why those inputs are fair, data-quality checks|
|**Explore**|Filter by city and date range; daily/monthly/yearly trends, distributions, month-of-year profile, summary statistics, per-decade trend|
|**City Compare**|Headline climate statistics, month × city heatmaps, seasonal comparison, auto-generated insights|
|**Model Lab**|Three models vs two baselines (MAE, RMSE, R²), error by city/month, permutation feature importance, and a demonstration of how evaluation choices inflate scores|
|**Predict**|Forecast the next day from a historical date, or from values you type (validated)|

## Quick start

Requires Python 3.10 or newer. From the project folder:

```bash
python -m venv .venv
# Windows (PowerShell):   .venv\\Scripts\\Activate.ps1
# Windows (cmd):          .venv\\Scripts\\activate.bat
# macOS / Linux:          source .venv/bin/activate
pip install -r requirements.txt

python train.py            # trains the models (about 3 minutes) -> models/ and reports/
streamlit run app.py       # opens http://localhost:8501
```

`train.py` must be run once before the **Model Lab** and **Predict** pages work. The
Overview, Explore and City Compare pages work without it.

The trained models are not included in this download on purpose: a model saved with one
scikit-learn version can misbehave when loaded with another, so you train on your own
machine. Re-run `python train.py` at any time to regenerate everything.

### Common problems

|Message / symptom|Fix|
|-|-|
|`ModuleNotFoundError: No module named ...`|The virtual environment is not active, or `pip install -r requirements.txt` was not run|
|"No training results found yet" in the app|Run `python train.py`, then refresh the page|
|"Trained model ... not found" / "could not be loaded"|Run `python train.py` again (usually a scikit-learn version change)|
|"Dataset not found"|`data/weather\_data.csv` must exist; open the project folder (the one containing `app.py`) as your working folder|
|PowerShell blocks `Activate.ps1`|Run `Set-ExecutionPolicy -Scope Process Bypass`, then activate again, or use `activate.bat` in cmd|

## Project structure

```
app.py        Streamlit interface (5 pages)
train.py      Offline training; writes models/ and reports/
data.py       Load + validate the CSV, with friendly error messages
features.py   Builds the "tomorrow's maximum" target and today's inputs
models.py     Ridge, Random Forest, Gradient Boosting; save/load helpers
evaluate.py   Chronological split, MAE/RMSE/R², baselines, importance, leakage demo
analysis.py   Statistics for the Explore and City Compare pages
plots.py      Plotly chart builders
paths.py      Project-root-relative file paths
data/weather\_data.csv
```

## How it works

**Task.** For every city and day D: using only information available at the end of day D,
predict the maximum temperature on day D+1.

**Inputs (11 numeric + city).** Today's max, yesterday's / 2- / 3-days-ago max, 7-day mean
of the max, today's min, today's rain, 7-day rain total, today's max wind speed, and the
day of the year encoded as sine/cosine (so 31 Dec and 1 Jan are neighbours). City is
one-hot encoded.

**Evaluation.** Chronological split by the date being predicted: train 2000–2019,
validation 2020–2021 (used to pick the best model), test 2022–2024 (final scores).
Each model is fitted on the training period and scored on validation; then re-fitted on
train + validation and scored once on the test period. Two baselines need no learning:
*tomorrow = today* and *the city's average maximum for that calendar month*.

### Results

Test period 2022–2024. Your numbers may differ slightly on another machine; the app always
shows the numbers from *your* latest `train.py` run.

|Method|MAE (°C)|RMSE (°C)|R²|
|-|-|-|-|
|City + month average (baseline)|1.84|2.42|0.761|
|Tomorrow = today (baseline)|0.93|1.30|0.931|
|Ridge regression|0.92|1.27|0.934|
|Random Forest|0.91|1.25|0.937|
|Gradient Boosting (best on validation)|0.90|1.24|0.938|

**Interpretation.** All models are much better than the monthly average, but only about
3% better (in MAE) than assuming tomorrow equals today. Daily maxima change slowly, so
persistence is a strong baseline. Today's maximum temperature is by far the most important
input. Errors are largest in May and for Delhi.

## Limitations

* Only past observations of one city are used; sudden changes (storms, fronts) cannot be anticipated.
* Skill is measured on one test window (2022–2024); other periods may differ.
* Only the next day is forecast; longer horizons are not attempted.
* Feature importance shows what the model *uses*, not what *causes* the weather, and shares credit between correlated inputs.
* The "typical error range" assumes future errors resemble the test-period errors.
* The data source is not documented in the project this one grew from (see below).

