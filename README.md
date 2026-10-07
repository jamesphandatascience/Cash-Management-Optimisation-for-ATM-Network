# Cash Management Optimisation for an ATM Network

Predicting daily cash demand for every ATM in a bank's network, so each machine can be stocked with the right amount of cash: enough to avoid running out, but not so much that idle cash sits unused.

**Result:** segmenting ATMs by location and day type, then fitting a separate OLS model to each segment, cut prediction error by **96%** compared with a single baseline linear model (test MSE **6.2 → 0.24**). That is a typical error of about **500 currency units per ATM per day**, down from about 2,500.

> UNSW Econometric Theory and Methods group project (Nov 2024). Full write-up: [`Cash Management Optimisation Report.pdf`](Cash%20Management%20Optimisation%20Report.pdf)

---

## The problem

Too much cash in an ATM ties up money the bank could lend out and increases security costs. Too little means machines run empty, frustrated customers and extra replenishment trips. Accurate demand prediction is what makes the trade-off manageable.

The dataset has **22,000 daily records**, each with the day's cash withdrawals (in thousands of currency units) and six features:

| Feature | Description |
|---|---|
| `Shops` | Number of shops and restaurants within walking distance |
| `ATMs` | Number of other ATMs within walking distance |
| `Downtown` | 1 if the ATM is downtown |
| `Workday` | 1 if the day is a workday, 0 if a holiday |
| `Center` | 1 if the ATM is in a shopping centre, airport, etc. |
| `High` | 1 if the ATM had high cash demand last month |

## Approach

1. **Exploratory analysis.** Scatterplots of withdrawals against shops and nearby ATMs showed **three distinct clusters**. Correlation heatmaps and variance inflation factors (VIF) revealed strong multicollinearity between `Downtown`, `Shops` and `ATMs`. It also exposed a misleading positive correlation between nearby ATMs and withdrawals at the aggregate level, which reversed to the expected negative relationship within each cluster.
2. **Segmentation.** The clusters were explained by location and day type:
   - **Cluster 1:** downtown, holiday, in a centre
   - **Cluster 2:** downtown on a workday, or downtown on a holiday outside a centre
   - **Cluster 3:** not downtown

   Splitting the data this way removed the multicollinearity and let each segment have its own relationships.
3. **Modelling.** We compared regularised and non-linear approaches, using cross-validated test MSE as the main metric:
   - Ridge, LASSO and Elastic Net with polynomial and interaction terms
   - LASSO fitted separately to K-Means clusters (4 clusters, chosen by silhouette score)
   - **OLS fitted separately to each segment**, with features chosen by best subset selection (base features) and forward stepwise selection (an enhanced set with polynomial, log and interaction terms)
   - Generalised Additive Models, for interpretation

## Results

| Model | Test MSE | R² |
|---|---|---|
| Baseline linear model (no interactions) | ~6.2 | – |
| Elastic Net | 0.281 | 0.9996 |
| Ridge (degree-3 polynomial + interactions) | 0.249 | 0.9996 |
| Polynomial regression | 0.248 | 0.9996 |
| LASSO + K-Means clustering | 0.243 (avg) | 0.94–0.998 |
| **Segmented OLS (final model)** | **0.238 (avg)** | **0.94–0.98** |

The segmented OLS model was chosen because it had the lowest error and is still interpretable segment by segment:

- **More nearby ATMs reduce demand per machine**, so adding ATMs to busy areas may not be worthwhile.
- **Holidays drive higher withdrawals**, especially in shopping centres, so those ATMs should be restocked ahead of holidays.
- **Demand rises with shop density but with diminishing returns**, which the log and polynomial terms capture.

## Repository contents

| File | What it does |
|---|---|
| `Clustered OLS.ipynb` | Segmentation, best subset and forward stepwise selection, final segmented OLS models |
| `Ridge.ipynb` | Ridge regression with polynomial and interaction features |
| `Lasso-with-K-means-clustering.ipynb` | K-Means segmentation followed by LASSO in each cluster |
| `ElasticNet.ipynb` | Elastic Net with cross-validated alpha and L1 ratio |
| `GAMs.ipynb` | Generalised Additive Models for interpreting each feature's effect |
| `Cash Management Optimisation Report.pdf` | Full report: EDA, methodology, results and limitations |

## Running the notebooks

Requires Python 3 with `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `matplotlib`, `seaborn` and `pygam`:

```bash
pip install pandas numpy scikit-learn statsmodels matplotlib seaborn pygam
```

The notebooks read `ATM_sample.csv` (and `ATM_test.csv` for the clustered OLS model) from the repository root. The course dataset is not included in this repository.

## Limitations and next steps

- Cluster 1 has only a few hundred observations, so its model may not generalise. Hierarchical clustering could reveal finer segments.
- The model minimises squared error, but running out of cash usually costs a bank more than holding too much. An **asymmetric loss function** that penalises under-prediction more heavily would align the model with real costs.
- External factors such as local events, weather and economic conditions are not included.

## Team

James Phan, Nancy Le, Lily Edwards, Hye Jun Jee and Wanda Kuai.
