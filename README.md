# Regularization: Buying Bias to Sell Variance

## Objective
I compared OLS, LASSO and Ridge on a simulated design with far more predictors than training rows, to see how a penalty trades a little bias for a large drop in variance.

## Methodology
- Simulated 150 rows and 200 predictors, where only the first 5 predictors affect the outcome, and split the data 70/30 into 105 training rows and 45 test rows.
- Fitted OLS, then LASSO and Ridge, each in a pipeline with a StandardScaler and each choosing its penalty by 5-fold cross-validation.
- Compared training MSE, test MSE and the number of nonzero coefficients for all three models.
- Traced the LASSO coefficient path over 60 penalties to see when the true predictors and the first noise variable entered.
- Chose the LASSO penalty with GridSearchCV over 21 penalties from 0.01 to 1, using shuffled 5-fold CV with the scaler refitted inside every fold.
- Scored LASSO's selected variables against the known truth.
- Built a second design where two predictors are near-duplicates (correlation 0.9985) and both matter, and compared how LASSO and Ridge handled the pair.
- Wrote my own prompt asking an AI for an explicit cross-validation loop over the penalty, and checked its chosen penalty and CV MSE against GridSearchCV's.

## Key Findings
- OLS fit the training rows perfectly but did badly on new data, with a test MSE of 12.9587. Ridge brought that down to 10.1128 and LASSO to 1.4219, because the truth is sparse and LASSO set most noise coefficients to zero.
- On the LASSO path, all five true predictors were in from step 6, and the first noise variable did not enter until step 17.
- Cross-validation with the scaler fitted inside every fold chose lambda = 0.1995, with a CV MSE of 1.5502.
- LASSO found 5 of 5 true predictors, along with 11 false positives.
- With two near-duplicate predictors, LASSO put 4.462 on one and zero on the other, while Ridge split the weight 1.618 and 1.609.
- I went 2 rounds with the AI. Its loop matched my lambda and CV MSE in every digit, on all 21 penalties and on every second one.
