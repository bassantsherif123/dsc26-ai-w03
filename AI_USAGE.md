# AI usage
## Tools I used, and for what
I used ChatGPT to help design the structure of add_history_features, time_split, and the preprocessing pipeline in build_pipeline. I also used it to understand the requirements, review edge cases, debug failing tests, and compare different machine learning models.

## AI-suggested code I rejected or fixed in add_history_features or time_split

ChatGPT intially suggested Logistic Reegression as the model to be used. We then expreminted with three models, tuned their hyperparameters and compared their performance:

| Model | ROC-AUC | PR-AUC | Brier Score |
|---|---:|---:|---:|
| Logistic Regression | 0.6552 | 0.3150 | 0.2148 |
| Decision Tree | 0.5907 | 0.2696 | **0.1750** |
| Random Forest | 0.6198 | 0.2856 | 0.2473 |

Logistic Regression achieved the highest ROC-AUC and PR-AUC, while Decision Tree achieved the lowest Brier Score. I chose Logistic Regression as my final model because it performed best in distinguishing between parcels likely to fail and those likely to succeed.

## How I checked that my model has no leak
I excluded parcel_id because it is only an identifier, and courier_notes and delivered_at because they are only available after the delivery attempt. I also checked that customer history features use only strictly earlier parcels.