# Smart Fake News Detection System

AIML with Python, Major Project

## What this project does

This project takes a news headline or article and predicts whether it looks
**real** or **fake**, based on patterns learned from a labelled dataset. It turns
the text into numbers with TF-IDF, trains and compares two classifiers (Logistic
Regression and Multinomial Naive Bayes), picks the better one, and serves it
through a simple **Streamlit** web app.

It is a text-classification tool, not a fact-checker. It learns how real and fake
articles tend to be written; it does not verify whether a claim is actually true.

## Objectives

- Load and clean a labelled real/fake news dataset.
- Explore it (label balance, article length) with charts.
- Convert the text to numeric features using TF-IDF, fitted on training data only.
- Train Logistic Regression and Multinomial Naive Bayes and compare them.
- Save the better model and the vectoriser, and serve them in a Streamlit app.

## Dataset

`fake_or_real_news.csv` — George McIntire's fake/real news dataset, **6,335 news
articles** labelled REAL or FAKE, each with a `title` and `text`. The classes are
almost perfectly balanced (3,171 real and 3,164 fake). It is mostly US political
news from around 2016.

- Download / source:
  https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv

To obtain the dataset, download it from the link above and keep it in the project
folder before running the notebook.

## Technology stack

Python, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, TF-IDF, Joblib and
Streamlit.

## Installing the dependencies

```bash
pip install -r requirements.txt
```

## Running the project

**1. Train the models (notebook).** Open `fake_news_analysis.ipynb` in Jupyter or
Google Colab, keep `fake_or_real_news.csv` in the same folder, and run all the
cells. This cleans the text, trains and compares the models, and saves two files:
`model.joblib` and `tfidf_vectorizer.joblib`.

**2. Launch the app.** With `model.joblib` and `tfidf_vectorizer.joblib` in the
folder, run:

```bash
streamlit run streamlit_app.py
```

Then paste a news headline or article into the box and press **Check this news**.
The app loads the saved model and vectoriser, so it predicts instantly and never
retrains.

## The models and evaluation

TF-IDF converts each article into weighted word counts, giving more weight to
words that are distinctive to an article. Two models are trained on these
features:

- **Logistic Regression**
- **Multinomial Naive Bayes**

Both are judged on a held-out 20% test set using accuracy, precision, recall,
F1-score and a confusion matrix. Precision, recall and F1 are measured for the
FAKE class, since catching fake news is the goal. The model with the higher F1 is
saved for the app.

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.920 | 0.901 | 0.943 | 0.921 |
| Multinomial Naive Bayes | 0.848 | 0.978 | 0.711 | 0.823 |

Logistic Regression was chosen. Naive Bayes is more precise (it rarely flags a
real article as fake) but misses many fake articles, while Logistic Regression
strikes a better balance, which the F1 score reflects.

## What is in the folder

| File | What it is |
|---|---|
| `fake_news_analysis.ipynb` | Main notebook: cleaning, EDA, TF-IDF, training, evaluation, saving |
| `streamlit_app.py` | The Streamlit web app |
| `predict_cli.py` | Command-line predictor (takes text input in the terminal) |
| `fake_or_real_news.csv` | The dataset |
| `model.joblib` | The saved trained model |
| `tfidf_vectorizer.joblib` | The saved fitted TF-IDF vectoriser |
| `Project_Report.pdf` | The project report |
| `requirements.txt` | Dependencies |
| `README.md` | This file |

Running the notebook also saves the charts (label balance, article length,
confusion matrices, model comparison) as `.png` files.

## Known limitations

- The model matches patterns; it does not fact-check. A confident prediction can
  still be wrong, especially for satire, breaking news, or topics outside the
  training data.
- The data is mostly 2016 US political news, so it will not transfer perfectly to
  other subjects or time periods.
- Very short inputs (a few words) are less reliable, since the model learned from
  full articles.
