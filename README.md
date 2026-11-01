# House Price Regression — Ames, Iowa

Data Bootcamp, Project 2: a regression model for the Kaggle challenge [Regression for house prices](https://www.kaggle.com/competitions/regression-for-house-prices). The goal is to predict `SalePrice` for 260 houses in Ames, Iowa; submissions are scored by RMSE.

**Team:** Yuzhou Huang, Mingxi Xiao

## Final model

`Ridge` regression on `log(SalePrice)`, using 71 of the 79 features (categories one-hot encoded, numeric features standardized). Predictions are converted back to dollars with `np.exp()`.

## Results

Measured on the training data (holdout = 25% of training houses not used for fitting; CV = 5-fold cross-validation RMSE on log price).

| Model | Holdout RMSE | CV RMSE (log) |
|---|---|---|
| Baseline from class: `LinearRegression` on `OverallQual` | $50,801 | 0.2305 |
| `LinearRegression`, all features + 4 summed features | $21,088 | 0.1233 |
| `LinearRegression`, 8 redundant features removed | $20,125 | 0.1177 |
| **`Ridge` (alpha = 10), same features** | **$19,601** | **0.1147** |

## Process

The notebook starts from the class sample notebook and changes one thing at a time:

1. **Log target.** Sale prices are right-skewed, and price effects are roughly proportional, so the model is fit on `log(SalePrice)`.
2. **Outlier.** One unfinished 4,676 sq ft house sold for $184,750, far below similar houses. It is removed from training.
3. **Missing values.** For most columns `NA` means "no garage / basement / fireplace", so categorical gaps are filled with `'None'`; numeric gaps with the median.
4. **Feature selection.** Removed 8 features that repeat other columns or are almost always one value (e.g. `Exterior2nd`, `Condition2`). Each removal was kept only if it lowered cross-validated error, and the gain was rechecked on separate CV splits.
5. **Encoding.** Compared `OrdinalEncoder` and `OneHotEncoder` for quality ratings; one-hot had lower CV error.
6. **Model.** Compared `LinearRegression` with `Ridge` (alpha chosen by `GridSearchCV`); Ridge had lower CV error.
7. **Extrapolation check.** One test house is larger than any training house and got a ~$2.7M prediction. Predictions are capped at the training price range ($34,900–$755,000).

Things we tried that did not help: extra summed features (`TotalSF`, `TotalBath`, house age), log transforms of skewed features, squared terms, and `KNeighborsRegressor`.

## Files

```
└── house-price-regression/
    ├── house_price_regression.ipynb   # full workflow with comments
    ├── preds.csv                      # Kaggle submission (Id = test row number 0–259, SalePrice)
    └── data/
        ├── train.csv
        └── test.csv
```

## How to run

```bash
pip install pandas numpy scikit-learn matplotlib jupyter
cd house-price-regression
jupyter notebook house_price_regression.ipynb
```

Run all cells from inside the `house-price-regression` folder; the last cell writes `preds.csv`.
