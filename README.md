# House Price Regression

Data Bootcamp Project 2 — Kaggle challenge: [Regression for house prices](https://www.kaggle.com/competitions/regression-for-house-prices)

**Team:** Yuzhou Huang, Mingxi Xiao

We predict `SalePrice` for houses in Ames, Iowa. We started with the sample notebook from class and improved it step by step:

- predicted `np.log(SalePrice)` instead of `SalePrice`
- dropped one unfinished house with a very low price
- dropped 8 columns that didn't help and added the log of `LotArea`
- compared `OrdinalEncoder` and `OneHotEncoder` for the quality columns (kept `OrdinalEncoder`)
- compared `LinearRegression` and `Ridge`

For our final predictions we used `Ridge`, which is `LinearRegression` with a limit on how big the coefficients can get. It had a slightly lower error than plain `LinearRegression` on our data. Test RMSE went from $50,801 (sample notebook) to about $19,700.

## Files

- `house-price-regression/house_price_regression.ipynb` — our notebook
- `house-price-regression/preds.csv` — our Kaggle submission
- `house-price-regression/data/` — `train.csv` and `test.csv`

To run it, open the notebook in the `house-price-regression` folder and run all cells.
