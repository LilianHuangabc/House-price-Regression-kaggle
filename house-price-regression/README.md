# House Price Regression — Ames, Iowa

Data Bootcamp, Project 2: a regression model for the Kaggle challenge [Regression for house prices](https://www.kaggle.com/competitions/regression-for-house-prices). The goal is to predict `SalePrice` for 260 houses in Ames, Iowa; submissions are scored by RMSE.

**Team:** Yuzhou Huang, Mingxi Xiao

## Final model

A `Ridge` model that predicts `np.log(SalePrice)`, using 71 of the 79 columns. Category columns use `OneHotEncoder` and number columns use `StandardScaler`. Predictions are turned back into dollars with `np.exp()`.

## Results

RMSE on a `train_test_split` of the training data (25% of houses held back), plus `cross_val_score` error on log price.

| Model | Test RMSE | cross_val_score (log price) |
|---|---|---|
| Sample notebook from class: `LinearRegression` on `OverallQual` | $50,801 | 0.2305 |
| `LinearRegression`, all columns + 4 new columns | $21,088 | 0.1233 |
| `LinearRegression`, 8 columns dropped | $20,125 | 0.1177 |
| **`Ridge` (alpha = 10), same columns** | **$19,601** | **0.1147** |

## What we did

We started with the sample notebook from class and changed one thing at a time:

1. **Predict log price.** A few very expensive houses throw off the model, so we predict `np.log(SalePrice)` instead.
2. **Drop one odd house.** One huge house sold for only $184,750 because it wasn't finished yet. We dropped it from the training data.
3. **Missing values.** A missing value usually means the house doesn't have that thing (no garage, no pool), so we fill those with `'None'` instead of the most common value. Missing numbers get the median.
4. **Drop 8 columns.** Some columns repeat other columns or are almost always the same (like `Exterior2nd` and `Condition2`). We dropped one at a time and kept the drop only if `cross_val_score` got better.
5. **`OrdinalEncoder` vs. `OneHotEncoder`.** For the quality ratings, `OneHotEncoder` gave lower error.
6. **`LinearRegression` vs. `Ridge`.** `Ridge` keeps the coefficients from getting too big, which helps with many similar columns. We picked `alpha` with `GridSearchCV`. `Ridge` gave lower error.
7. **Keep predictions realistic.** One test house is bigger than every training house and got a $2.7M prediction, so we used `np.clip` to keep predictions between the lowest and highest training prices ($34,900–$755,000).

What didn't help: adding new columns (total area, total bathrooms, house age), taking the log of other columns, squared terms, and `KNeighborsRegressor`.

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
