# SMS Spam Detection: Legitimate Promotion-Word Messages vs Genuine Spam

NLP Course Project, ARTI501 (9MA1), Group 6, Option 1: Spam Detection
Team: Ali Fawzi Alshehri, Osama Saeed Al Haddaf, Hussain Ali Alabdurabalnabi

**Task.** Given an SMS message, return **Spam** or **Not spam**.

**Approved experimental question.** Which method distinguishes legitimate promotion-word SMS messages from
genuine spam more reliably: TF-IDF word features with Logistic Regression, or pretrained sentence
representations with a classifier?

## Status

| Stage | Notebook section | Lead | Status |
|---|---|---|---|
| Setup, data, cleaning, de-duplication, split | 1–2 | Ali (data), Hussain (challenge subset) | Done (first version, 22 Nov 2026) |
| Method 1: TF-IDF + Logistic Regression | 3 | Ali | Done: tuned on dev, final model trained (22 Nov 2026) |
| Method 2: Sentence embeddings + Logistic Regression | 4 | Osama | Done: tuned on dev, final model trained (22 Nov 2026) |
| Results on the test set | 5 | Hussain | Done: test set evaluated once (22 Nov 2026), errors explained, question answered |
| Try it | 6.1–6.3 | Hussain | Done (22 Nov 2026) |
| Original labelled examples | 6.4 | Every member wrote 2 | Done (23 Nov 2026): both models correct on all 6 |

## How to run

**Google Colab (main environment).** Open `SMS_Spam_Detection.ipynb` in Colab and choose
*Runtime → Run all*. No GPU is needed. If `data/` is missing, the notebook downloads the dataset from UCI
and verifies its checksum. The first run also downloads the MiniLM model (about 90 MB). A full run takes
under a minute once the model is cached. Once the repository is on GitHub, it can be opened directly at
`https://colab.research.google.com/github/<user>/<repo>/blob/main/SMS_Spam_Detection.ipynb`.

**Local Jupyter.**

```
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install notebook              # or open the notebook in VS Code
jupyter notebook SMS_Spam_Detection.ipynb
```

Then choose *Kernel → Restart & Run All*.

**Try it (Section 6.2).** After *Run all*, edit `MESSAGE` and run that cell, or set `ASK_FOR_INPUT = True` to
type a message when the cell runs. The prediction comes from the Try-it model (Method 2), with Method 1's
opinion alongside. Nothing is retrained. Empty or non-text input gets a helpful message instead of an error.
You can also call `classify_sms("your message")` from any new cell.

## Repository contents

| Path | What it is |
|---|---|
| `SMS_Spam_Detection.ipynb` | The main notebook: Setup, Data, Method 1, Method 2, Results, Try it, Sources |
| `data/smsspamcollection.zip` | UCI SMS Spam Collection v.1, unchanged (SHA-256 `1587ea43…9963ce3`) |
| `data/splits.csv` | Frozen train/dev/test assignment for every message kept (`msg_id` = line number in the UCI file) |
| `results/` | Tables the notebook writes for the report: class and split counts, both methods' tuning grids, the development-set comparison and development-set probabilities. Each run reports whether it reproduced them exactly. |
| `requirements.txt` | Package versions the notebook was tested with |

## Data preparation (Section 2 of the notebook)

* **Source:** UCI SMS Spam Collection v.1, 5,574 messages (4,827 ham, 747 spam).
* **Cleaning:** HTML entities decoded, Windows-1252 leftovers repaired, whitespace collapsed. Case,
  punctuation, digits and currency symbols are kept.
* **Exact duplicates removed** (ignoring case and spacing): 416 removed, leaving 5,158 unique messages
  (4,517 Not spam, 641 Spam).
* **Near-duplicates grouped** (same text once digits and punctuation are ignored), and whole groups are split
  so templated spam never crosses splits.
* **Challenge subset:** messages containing *free, win, wins, won, winner, winning, offer, offers, prize,
  prizes* as whole words: 356 messages (73 Not spam, 283 Spam).
* **Split:** 70/15/15 by template group, stratified by label × challenge flag, seed 42. The result is
  train 3,600 / dev 773 / test 785. Split fingerprint: **`a8889b52ce6f`**.

## Methods (Sections 3–4 of the notebook)

Both methods use the same classifier (logistic regression), so the comparison isolates the text representation.
All settings are chosen by macro F1 on the **development** set, and the final models are trained on the
training set only.

| | Method 1 | Method 2 |
|---|---|---|
| Representation | TF-IDF word features (lowercased, 1-character tokens kept, sublinear TF) | Frozen `sentence-transformers/all-MiniLM-L6-v2` embeddings (384 dimensions, L2-normalised), revision `1110a243` |
| Settings searched | n-grams × min_df × C (0.1–1000) × class weight: 40 | C (0.1–100,000) × class weight: 14 |
| Selected | word 1–2-grams, min_df 1, C = 100, balanced | C = 1000, no class weight |
| Dev accuracy / macro F1 | 0.9884 / 0.9724 | 0.9935 / 0.9848 |

Development-set scores were used to choose settings, so they are optimistic. **They are not the final results.**
Method 2 has the higher dev macro F1, so it is the Try-it model, as fixed in the evaluation plan.

## Test results (Section 5 of the notebook)

The test set (785 messages: 676 Not spam, 109 Spam) was used once, after Sections 2–4 were frozen.

| | Accuracy | Macro F1 | Challenge macro F1 | Legit challenge msgs flagged | Promo spam caught |
|---|---|---|---|---|---|
| Method 1: TF-IDF + LR | 0.9796 | 0.9571 | 0.9121 | 2 of 11 | 44 of 45 |
| Method 2: MiniLM embeddings + LR | 0.9847 | 0.9680 | 0.9707 | 1 of 11 | 45 of 45 |

McNemar's exact test:

| | Only Method 1 correct | Only Method 2 correct | p-value |
|---|---|---|---|
| Full test set | 6 | 10 | 0.45 |
| Challenge subset | 0 | 2 | 0.50 |

**Answer.** Method 2 handled the challenge subset better on the test set. However, the difference rests on two
messages and is not statistically significant, and the development set showed the opposite order on the
subset. The results are consistent with the hypothesis but do not establish it. Section 5.7 of the notebook
explains the errors. In short:
* Method 1's false alarms are mostly very short messages dominated by spam-associated words such as `call`
  and `text`.
* Method 2's false alarms are unusual messages with no close training neighbour.
* Both methods miss spam that is spam only because of its sender (subscription content, classified ads).

## Reproducibility

The notebook was run top to bottom from a fresh kernel. A second run reproduced every saved results file
exactly. It was also run from an empty folder with only the notebook, as on Colab, using older package
versions. That run downloaded the data and produced the same split, the same selected settings and identical
tuning, test-metric and error-analysis tables. The only differences were in the two probability files, where a
few Method 2 probabilities changed by 0.0001 (floating-point noise), with no prediction changed.

| Environment | Python | pandas | numpy | scikit-learn | torch | sentence-transformers |
|---|---|---|---|---|---|---|
| **Google Colab** (24 Nov 2026) | 3.13.18 | 2.2.3 | 2.4.6 | 1.6.1 | 2.14.1 (CPU) | 5.7.0 |
| Local Jupyter (Windows) | 3.13.9 | 3.1.1 | 2.5.5 | 1.9.1 | 2.15.0 (CPU) | 6.3.0 |
| Local Jupyter (Windows) | 3.13.9 | 2.3.3 | 2.2.6 | 1.6.1 | 2.15.0 (CPU) | 5.7.0 |

**The submitted notebook's outputs come from the Colab run.** That run started from a fresh session with no
`data/` or `results/` folder: it downloaded the dataset and produced the same split fingerprint
(`a8889b52ce6f`). Every table and printed result matches the local runs, with no errors or warnings.

## Credits

* **Data:** T. A. Almeida and J. M. Gómez Hidalgo, "SMS Spam Collection," UCI Machine Learning Repository, 2011,
  doi:10.24432/C5CC84. The corpus readme allows free use with citation of: T. A. Almeida, J. M. Gómez Hidalgo
  and A. Yamakami, "Contributions to the study of SMS spam filtering: new collection and results," ACM DocEng
  2011, doi:10.1145/2034691.2034742.
* **Pretrained model:** `sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0), Hugging Face. N. Reimers and
  I. Gurevych, "Sentence-BERT," EMNLP-IJCNLP 2019; W. Wang et al., "MiniLM," NeurIPS 2020.
* **Libraries:** pandas, NumPy, scikit-learn, PyTorch, sentence-transformers.
* **AI assistance:** parts of the data-preparation and model code and the notebook structure were drafted with help from
  Claude (Anthropic). The team reviewed, ran and edited all code and is responsible for it.
