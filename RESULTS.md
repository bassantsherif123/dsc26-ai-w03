# Results: Tawseel failed-delivery baseline
## Baseline table
Evaluation uses the last 20% of parcels by time from `make_parcels(4000, seed=0)`.

| Model | ROC AUC | PR-AUC | Brier | Base rate |
|---|---:|---:|---:|---:|
| Dummy (prior) | 0.5000 | 0.2200 | 0.1717 | 0.2200 |
| Rule| — | — | — | — |
| My model (Logistic Regression) | 0.6492 | 0.3837 | 0.1630 | 0.2200 |

PR-AUC lift over dummy: +0.1637

## Three sentences for Sherif
The model uses parcel information available at creation time and the customer's previous delivery history to estimate the likelihood of delivery failure. On the time-based test set, it achieved a PR-AUC of 0.3837 compared with 0.2200 for the Dummy baseline, while also achieving a lower Brier Score (0.1630 versus 0.1717). These results show that the model can identify at-risk parcels more effectively than the baseline while producing more accurate probability estimates, although its performance in real-world deliveries still needs to be evaluated.