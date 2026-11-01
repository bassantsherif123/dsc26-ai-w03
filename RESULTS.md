# Results: Tawseel failed-delivery baseline
## Baseline table

<!-- Last 20% of parcels by time, make_parcels(4000, seed=0). Add any rule you tried. -->

Evaluation uses the last 20% of parcels by time from `make_parcels(4000, seed=0)`.

| Model | ROC AUC | PR-AUC | Brier | Base rate |
|---|---:|---:|---:|---:|
| Dummy (prior) | 0.5000 | 0.2200 | 0.1717 | 0.2200 |
| Rule| — | — | — | — |
| My model (Logistic Regression) | 0.6458 | 0.3799 | 0.2100 | 0.2200 |

PR-AUC lift over dummy: +0.1599.

## Three sentences for Sherif

The model uses parcel information available at creation time and the customer's previous delivery history to estimate the likelihood of delivery failure. On the time-based test set, it achieved a PR-AUC of 0.3799 compared with 0.2200 for the Dummy baseline, showing that it can identify at-risk parcels better than a model that always predicts the overall failure rate. However, its predictions are not perfect, its Brier Score is higher than the dummy baseline's, and its performance on real-world deliveries or entirely new customers and merchants has not yet been established.