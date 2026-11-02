# House Price Regression — Ames, Iowa

Project 2 for the Kaggle challenge [Regression for house prices](https://www.kaggle.com/competitions/regression-for-house-prices): predict `SalePrice` for 365 houses in Ames, IA, scored by RMSE.

## Result

| Model | 5-fold CV RMSE |
|---|---|
| Linear regression (baseline) | $25,594 |
| Lasso, log target | $20,385 |
| **Final: 60% Lasso + 40% Gradient Boosting** | **$19,802** |

## Approach

1. **Log-transform the target.** Sale prices are right-skewed; modeling `log(SalePrice)` fits percentage effects and keeps expensive houses from dominating the loss.
2. **Remove two outliers.** Two unfinished 4,600+ sq ft houses sold far below value (`SaleCondition = Partial`).
3. **Treat `NA` by meaning.** Most missing values mean "no garage / pool / fence", so they are encoded as absent rather than imputed.
4. **Feature engineering.** Quality ratings (Ex/Gd/TA/Fa/Po) mapped to numbers; added `TotalSF`, `TotalBath`, `HouseAge`, `RemodAge`, `TotalPorchSF` and `QualxSF`.
5. **Model comparison.** Six scikit-learn models under the same 5-fold cross-validation; the final model is a weighted `VotingRegressor` of Lasso and gradient boosting.

The notebook walks through each step with charts and commentary.

## Repository

```
├── house_price_regression.ipynb   # full analysis, model and commentary
├── submission.csv                 # Kaggle submission (Id, SalePrice)
└── data/
    ├── train.csv
    └── test.csv
```

## Running it

Requires Python 3.10+ with `pandas`, `numpy`, `scikit-learn`, `matplotlib` and `seaborn`.

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook house_price_regression.ipynb
```

Run all cells; the last section writes `submission.csv`. A full run takes about two minutes.
