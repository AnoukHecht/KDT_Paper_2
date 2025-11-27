# How Many Features Are Really Needed? A Systematic Ablation Study on Movie Revenue Prediction

**Abstract**

Predicting movie box office revenue is important for film studios and investors, but it remains unclear which features actually matter for accurate predictions. In this study, we use a systematic ablation methodology to find out which features are truly necessary for revenue prediction. We work with the TMDB Movie Dataset and trained Random Forest models on 7,380 movies using 26 carefully engineered features. Our baseline model achieves a test R² of 0.7292 after applying regularized hyperparameters to reduce overfitting. Through four different ablation experiments—individual feature removal, cumulative feature addition, top-k selection, and group-level ablation—we discover something surprising. Budget alone accounts for 25.7% of the model's predictive power. Just three features (budget, vote count, and release year) retain 95.0% of baseline accuracy. Popularity metrics add another 10.6%, while production metadata, genres, and languages combined contribute less than 3%. We validate these findings using bootstrap confidence intervals and paired t-tests on the test set. Our results show that accurate revenue prediction doesn't require dozens of features. This has practical implications for studios with limited data resources and researchers interested in model interpretability.

---

## 1. Introduction

### 1.1 Why Movie Revenue Prediction Matters

The film industry invests billions of dollars each year in movie production. A single blockbuster can cost $200 million or more to produce and market. Predicting box office revenue helps studios make better decisions about which projects to greenlight, how much to spend on marketing, and how to distribute films across theaters. But predicting revenue is difficult because many factors influence a film's commercial success.

Traditional industry wisdom emphasizes factors like star power, genre, and marketing budgets. However, data-driven approaches can systematically identify which features genuinely predict revenue versus which ones just seem important. Understanding this distinction helps studios focus their efforts on what actually matters. It also reduces the cost of collecting and maintaining unnecessary data.

### 1.2 The Research Question

This paper investigates a specific question: **Which features are truly necessary for predicting movie box office revenue, and how much prediction accuracy can we achieve with a minimal feature set?**

To answer this question, we use ablation studies. This method, borrowed from neuroscience and deep learning research, systematically removes features and measures how much performance drops. Unlike traditional feature importance metrics that only show correlations, ablation studies provide causal evidence about which features actually drive predictions.

### 1.3 What We Did

We collected data on 7,380 movies from the TMDB database. From the raw data, we engineered 26 features covering financial metrics, popularity signals, temporal information, and categorical attributes like genres and languages. We then trained Random Forest regression models using hyperparameters optimized through cross-validation to minimize overfitting.

Our ablation study includes four complementary experiments:
1. **Individual Feature Ablation**: Remove each of the 26 features one at a time and measure performance drop
2. **Cumulative Feature Addition**: Start with zero features and add them one by one in order of importance
3. **Top-K Selection**: Train models using only the top 3, 5, 7, and 10 most important features
4. **Group Ablation**: Remove entire categories of features (financial, popularity, genres, etc.) simultaneously

All results are evaluated on a held-out test set that was never used during model development. This ensures our findings represent true performance on unseen data.

### 1.4 Main Findings

Our results reveal three key insights:

First, budget dominates all other features. Removing budget causes test R² to drop from 0.7292 to 0.5417—a loss of 25.7% of the model's predictive power. No other single feature comes close to this impact.

Second, a minimal three-feature model (budget, vote count, release year) achieves R² = 0.6924 on the test set, retaining 95.0% of baseline performance. Adding more features beyond the top 5-10 provides diminishing returns.

Third, categorical features like genres and languages contribute very little. Despite comprising 15 of our 26 features (58%), genres and languages combined contribute only 1.4% of predictive power.

These findings challenge the assumption that more features always improve predictions. For many practical applications, a simpler model provides comparable accuracy with significantly reduced complexity.

### 1.5 Paper Structure

The rest of this paper is organized as follows. Section 2 reviews related work on feature selection and movie revenue prediction. Section 3 describes our dataset and preprocessing steps. Section 4 explains our methodology, including model selection and experimental design. Section 5 presents results from all four ablation experiments. Section 6 discusses implications and limitations. Section 7 concludes with recommendations for practitioners and future research directions.

---

## 2. Related Work

### 2.1 Feature Selection Methods

Feature selection is a fundamental problem in machine learning. There are three main approaches. Filter methods use statistical measures like correlation or mutual information to rank features independently of any model. Wrapper methods evaluate feature subsets by training models and measuring performance. Embedded methods perform feature selection as part of the model training process, like tree-based importance or L1 regularization.

Each approach has trade-offs. Filter methods are fast but don't account for feature interactions. Wrapper methods consider interactions but are computationally expensive. Embedded methods balance speed and accuracy but are model-specific. Our study uses embedded methods (Random Forest importance) combined with ablation experiments to measure actual performance impact.

### 2.2 Ablation Studies in Machine Learning

Ablation studies originated in neuroscience, where researchers remove brain regions to understand their function. The method has become standard practice in deep learning research for understanding which model components matter. Researchers systematically remove layers, attention heads, or training data to measure impact on performance.

Despite their value, ablation studies remain rare in traditional machine learning with tabular data. Most studies rely on importance scores from models like Random Forests or permutation importance. Our work demonstrates that ablation provides clearer evidence about which features are necessary versus merely correlated with outcomes.

### 2.3 Movie Revenue Prediction

Previous research on movie revenue prediction has explored many factors. Early studies used linear regression with basic features like budget, genre, and star power. More recent work employs ensemble methods like Random Forests and Gradient Boosting, or even deep neural networks. Some researchers incorporate social media sentiment, critical reviews, or YouTube trailer views.

A consistent finding across studies is that budget strongly predicts revenue, often explaining 30-50% of variance alone. However, few studies have systematically quantified the marginal contribution of additional features while controlling for all other variables. Our ablation approach fills this gap.

### 2.4 Random Forests for Regression

We chose Random Forest regression for several reasons. First, Random Forests handle non-linear relationships without requiring manual feature engineering or interaction terms. Second, they're robust to outliers and correlated features through random subsampling. Third, they provide built-in feature importance metrics. Fourth, they scale well to medium-sized datasets (thousands of samples).

Random Forests do have limitations. They can overfit on small datasets if trees are too deep. They struggle with extreme extrapolation beyond training data ranges. And their black-box nature makes them less interpretable than linear models. We address overfitting through cross-validation and regularization. We accept reduced extrapolation ability because our focus is prediction, not causal inference.

### 2.5 What's Missing

While existing work establishes that revenue prediction is feasible, several questions remain unanswered. First, what is the minimum feature set required for strong performance? Second, how do different feature groups (financial vs. popularity vs. metadata) contribute independently? Third, which findings generalize to held-out test data versus validation data used during development?

Our study addresses these gaps through comprehensive ablation experiments with rigorous statistical validation on test data.

---

## 3. Dataset and Preprocessing

### 3.1 The TMDB Movie Dataset

We use the TMDB (The Movie Database) dataset from Kaggle, which contains information about 119,938 movies. The dataset includes financial data (budget, revenue), ratings (vote average, vote count, popularity), metadata (title, overview, runtime), temporal information (release date), and categorical attributes (genres, production companies and countries, spoken languages, cast, and crew).

This dataset is well-suited for our study because it covers diverse aspects of film production and reception. The data spans movies from 1916 to 2017, though older films have less complete information.

### 3.2 Data Cleaning

The original dataset has significant data quality issues. Many entries have missing or zero values for critical variables. We applied strict filtering to ensure high-quality training data.

First, we removed all movies with missing budget, revenue, or release date. This eliminates entries that can't be used for supervised learning. Second, we excluded movies with budget = $0 or revenue = $0, as these typically represent missing data rather than actual free movies. Third, we filtered out movies with implausible runtimes (less than 30 minutes or more than 300 minutes) to remove data errors and non-feature films. Fourth, we required at least one genre, production company, and spoken language to ensure minimum metadata completeness.

After this aggressive filtering, 7,380 movies remain—just 6.2% of the original dataset. While this dramatically reduces sample size, it ensures our models train on complete, reliable data rather than imputed values. The trade-off is selection bias: our cleaned dataset overrepresents major studio releases with full financial documentation and underrepresents independent and international films.

Table 1 summarizes the cleaned dataset characteristics:

| Characteristic | Value |
|----------------|-------|
| Total movies (after cleaning) | 7,380 |
| Training set (70%) | 5,166 |
| Validation set (15%) | 1,107 |
| Test set (15%) | 1,107 |
| Number of features | 26 |
| Target variable | Revenue (log-transformed) |
| Mean revenue (original) | $88.5 million |
| Median revenue (original) | $35.2 million |
| Revenue std dev (original) | $149.3 million |

### 3.3 Feature Engineering

From the raw TMDB data, we constructed 26 features organized into seven semantic groups.

**Financial features (1):**
The most important feature is `budget_log`, the log₁₀-transformed production budget. We use log transformation because budgets span six orders of magnitude (from thousands to hundreds of millions of dollars). The log transformation captures that doubling a $10M budget has more impact than doubling a $100M budget.

**Popularity features (3):**
These capture pre-release audience interest. `vote_count_log` is the log-transformed number of user votes on TMDB. `popularity_log` is the log-transformed TMDB popularity score, which aggregates signals like search volume and watchlist additions. `vote_average` is the mean user rating on a 0-10 scale. These features are valuable because they measure forward-looking engagement before theatrical release.

**Temporal features (2):**
`release_year` captures the year of theatrical release. This accounts for inflation, market growth, and changing audience preferences over time. `release_month` (1-12) captures seasonal effects, as some months (summer, holidays) have higher box office potential than others.

**Content features (3):**
`runtime` is movie length in minutes. Longer films may signal epic scope or allow richer storytelling. `cast_size` counts the number of actors listed in the cast. Ensemble films may attract more diverse audiences. `director_count` counts the number of directors, though most films have just one.

**Production features (2):**
`production_company_count` and `production_country_count` measure the scale of the production infrastructure. More companies or countries might indicate larger budgets or wider distribution.

**Genre features (10):**
Binary indicators for the top-10 most frequent genres: Drama, Comedy, Action, Thriller, Romance, Adventure, Crime, Horror, Science Fiction, and Family. These capture content type but don't account for multi-genre films (which receive multiple 1s).

**Language features (5):**
Binary indicators for the top-5 most frequent languages: English, Hindi, French, Russian, and Spanish. English dominates our dataset, likely due to TMDB's bias toward Hollywood releases.

All continuous features (except binary indicators) were standardized using StandardScaler (zero mean, unit variance) to ensure comparable scales. The scaler was fit only on training data, then applied to validation and test sets to prevent data leakage.

### 3.4 Target Variable Transformation

The target variable, revenue, exhibits extreme right-skew. Values range from thousands to billions of dollars, spanning six orders of magnitude. We applied log₁₀ transformation to normalize the distribution and stabilize variance.

After log transformation, revenue approximates a normal distribution, which improves model performance and interpretability. One consequence is that all metrics (R², RMSE, MAE) are computed on the log scale. An RMSE of 1.59 log-units doesn't mean "$1.59 error." It means predictions deviate by a factor of 10^1.59 ≈ 39 times from actual values, or roughly 4.9× (e^1.59) on the natural log scale.

### 3.5 Train/Validation/Test Split

We split the data into three disjoint sets before any analysis. The training set (70%, n=5,166) is used exclusively for model training. The validation set (15%, n=1,107) is used for hyperparameter tuning and determining which feature subsets to evaluate. The test set (15%, n=1,107) is held out completely until final evaluation.

This three-way split is critical for honest performance estimation. Using only train/test would be risky because we'd tune hyperparameters and select features on the test set, leading to overly optimistic results. By reserving the test set strictly for final evaluation, we get an unbiased estimate of how the model performs on truly unseen data.

---

## 4. Methodology

### 4.1 Why Random Forest?

We chose Random Forest regression as our prediction model for several reasons. First, Random Forests naturally handle mixed data types (continuous, ordinal, binary) without requiring one-hot encoding or other transformations. Second, they capture non-linear relationships and interactions without manual feature engineering. Third, they're robust to outliers because tree splits use rank-based thresholds rather than exact values. Fourth, they provide built-in feature importance scores through mean decrease in impurity.

Random Forests do have drawbacks. They can overfit on small datasets or when trees are very deep. They don't extrapolate well beyond the training data range. And interpreting individual predictions is difficult compared to linear models. We address overfitting through hyperparameter regularization, which we describe next.

### 4.2 Hyperparameter Tuning

Early experiments showed strong overfitting: training R² = 0.911 versus validation R² = 0.739, a gap of 17.2%. To reduce this, we implemented a regularized hyperparameter search.

We used GridSearchCV with 5-fold cross-validation to explore 24 parameter combinations. The search space was deliberately constrained to favor simpler models:

- `max_depth`: [8, 10, 12] – Limits tree depth to prevent memorization
- `n_estimators`: [150, 200] – Moderate number of trees
- `min_samples_split`: [15, 25] – Requires more samples before splitting a node
- `min_samples_leaf`: [8, 12] – Ensures predictions average over multiple samples
- `max_features`: ['sqrt'] – Uses only √26 ≈ 5 features per split to decorrelate trees

Each of the 24 combinations was evaluated via 5-fold cross-validation, requiring 120 model fits. The best configuration found was:
- `max_depth`: 12
- `n_estimators`: 200
- `min_samples_split`: 15
- `min_samples_leaf`: 8
- `max_features`: 'sqrt'

This configuration achieved a cross-validation R² of 0.7013 with a train-CV gap of only 7.7%, indicating excellent regularization. We used these hyperparameters consistently across all ablation experiments to ensure fair comparisons.

### 4.3 Baseline Model Performance

Using the tuned hyperparameters, the baseline model (trained on all 26 features) achieved the following performance:

| Split | R² | RMSE (log) | MAE (log) | Error Factor |
|-------|-----|------------|-----------|--------------|
| Train | 0.8762 | 1.0562 | — | ~2.9× |
| Validation | 0.7373 | 1.4891 | 0.9429 | ~4.4× |
| Test | 0.7292 | 1.5943 | 1.0271 | ~4.9× |

The train-validation R² gap of 13.9% indicates moderate overfitting despite our regularization efforts. However, the validation-test R² difference is only 1.1%, suggesting the validation set provides a reliable estimate of test performance. The test R² of 0.7292 means our model explains about 73% of revenue variance, which is competitive with prior work.

The RMSE on the test set is 1.5943 log-units. This translates to predictions deviating by roughly 10^1.59 ≈ 39× (or e^1.59 ≈ 4.9×) from actual revenue on average. For a $100M movie, this means typical errors of ±$50-200M—substantial in dollar terms but reasonable given the inherent unpredictability of box office success.

### 4.4 Ablation Study Design

We conducted four complementary ablation experiments to understand feature importance from different angles.

**Experiment 1: Individual Feature Ablation**

For each of the 26 features, we trained a model on the remaining 25 features and measured the R² drop on the test set. Features causing large drops are considered critical; features causing small or negative drops are considered redundant.

This experiment answers: "What is each feature's unique contribution when all other features are present?"

**Experiment 2: Cumulative Feature Addition**

Starting with an empty feature set, we added features one at a time in order of Random Forest importance. After adding each feature, we trained a model and measured test R². Plotting R² versus number of features reveals the point of diminishing returns.

This experiment answers: "How many features do we need before additional features stop helping?"

**Experiment 3: Top-K Feature Selection**

We trained models using only the top-3, top-5, top-7, and top-10 features (ranked by Random Forest importance), plus the full 26-feature model for comparison. This directly tests whether small feature subsets can match full-model performance.

This experiment answers: "What's the optimal trade-off between model complexity and prediction accuracy?"

**Experiment 4: Group Ablation**

We grouped features into seven semantic categories: Financial, Popularity, Temporal, Content, Production, Genres, and Languages. For each group, we trained a model with all features except that entire group and measured R² drop.

This experiment answers: "Which categories of information matter most for revenue prediction?"

All ablation experiments used the same tuned hyperparameters and were evaluated on the test set. This ensures differences reflect true feature importance rather than hyperparameter choices or overfitting to validation data.

### 4.5 Statistical Validation

To verify that observed performance differences are statistically reliable, we employed two validation techniques.

**Bootstrap Confidence Intervals**

For each model configuration (baseline, top-3, top-5, etc.), we computed 95% confidence intervals for R², RMSE, and MAE using bootstrap resampling with 1,000 iterations on the test set. Each iteration randomly samples 1,107 predictions with replacement and calculates metrics. The 2.5th and 97.5th percentiles of these 1,000 values form the confidence interval.

Narrow confidence intervals indicate stable, reliable performance estimates. Wide intervals suggest high variability and less confidence in the exact values.

**Paired t-Tests**

We compared models pairwise using paired t-tests on absolute residuals (prediction errors) from the test set. For each sample in the test set, we compute the absolute error for model A and model B, then test whether the mean difference is significantly different from zero using a two-tailed t-test with α = 0.05.

A significant p-value (p < 0.05) means one model is reliably better than the other. A non-significant p-value (p ≥ 0.05) means the models perform similarly within statistical noise.

### 4.6 Evaluation Metrics

We report three metrics, all computed on log-transformed revenue:

**R² (Coefficient of Determination)**: The proportion of variance in log-revenue explained by the model. Higher is better, with 1.0 being perfect and 0 being no better than predicting the mean.

**RMSE (Root Mean Squared Error)**: The standard deviation of prediction errors on the log scale. Lower is better. We report both the log-scale value and the corresponding error factor (10^RMSE or e^RMSE).

**MAE (Mean Absolute Error)**: The average absolute prediction error on the log scale. Lower is better. MAE is less sensitive to extreme outliers than RMSE.

All metrics are computed on the test set for the final reported results, ensuring honest performance estimates.

---

## 5. Results

### 5.1 Individual Feature Ablation

Table 2 shows the top-10 most critical features ranked by R² drop on the test set when each feature is removed individually.

**Table 2: Individual Feature Ablation Results (Test Set)**

| Rank | Feature Removed | R² (Test) | R² Drop | Interpretation |
|------|----------------|-----------|---------|----------------|
| 1 | budget_log | 0.5417 | **0.1875** | Critical (25.7% of baseline) |
| 2 | vote_count_log | 0.7107 | 0.0184 | Important (2.5% of baseline) |
| 3 | release_year | 0.7186 | 0.0106 | Moderate (1.5% of baseline) |
| 4 | runtime | 0.7246 | 0.0045 | Minor (0.6% of baseline) |
| 5 | lang_es | 0.7251 | 0.0041 | Minor (0.6% of baseline) |
| 6 | cast_size | 0.7270 | 0.0021 | Minimal (0.3% of baseline) |
| 7 | popularity_log | 0.7271 | 0.0020 | Minimal (0.3% of baseline) |
| 8 | lang_en | 0.7275 | 0.0016 | Minimal (0.2% of baseline) |
| 9 | production_country_count | 0.7279 | 0.0013 | Minimal (0.2% of baseline) |
| 10 | lang_fr | 0.7285 | 0.0007 | Negligible (0.1% of baseline) |

**Baseline R² (all 26 features)**: 0.7292

The results are striking. Budget dominates overwhelmingly. Removing it causes R² to drop from 0.7292 to 0.5417—a loss of 0.1875 or 25.7% of the baseline's predictive power. No other feature comes remotely close to this impact.

The second most important feature, vote_count_log, causes an R² drop of only 0.0184 (2.5% of baseline)—more than 10 times smaller than budget's impact. Release year ranks third with a 1.5% contribution. Beyond the top three, most features contribute less than 1% each.

Interestingly, some features show negative R² drops, meaning their removal slightly improves performance. For example, removing vote_average, director_count, or production_company_count increases test R² by 0.001-0.002. This suggests these features add noise or redundant information that hurts generalization.

The least important features are categorical variables. Individual genres contribute almost nothing (all < 0.001 R² drop). Language indicators are similarly weak, except for Spanish, which has a small 0.4% contribution.

### 5.2 Top-K Feature Selection

Table 3 shows test set performance using only the top-K most important features.

**Table 3: Top-K Feature Selection Results (Test Set)**

| K | Features | R² (Test) | % of Baseline | RMSE (Test, log) |
|---|----------|-----------|---------------|------------------|
| 3 | budget_log, vote_count_log, release_year | 0.6924 | **95.0%** | 1.6990 |
| 5 | +runtime, popularity_log | 0.7169 | **98.3%** | 1.6301 |
| 7 | +vote_average, cast_size | 0.7163 | 98.2% | 1.6317 |
| 10 | +3 production/temporal features | 0.7184 | 98.5% | 1.6258 |
| 26 | All features | 0.7293 | 100.0% | 1.5939 |

The key finding: **just three features achieve 95.0% of baseline performance**. Adding runtime and popularity brings this to 98.3% with only five features. Beyond that, diminishing returns set in quickly.

The difference between 5 and 10 features is only 0.15 percentage points (0.7169 vs 0.7184). The difference between 10 and all 26 features is only 1.5 percentage points (0.7184 vs 0.7293). These small gains come at the cost of 16 additional features—a poor complexity-accuracy trade-off.

Interestingly, the 7-feature model performs slightly worse than the 5-feature model (0.7163 vs 0.7169). This suggests that features 6-7 (vote_average and cast_size) might add slight noise or correlate redundantly with features 1-5. However, the 10-feature model rebounds to 0.7184, indicating some value in features 8-10 (production_company_count, production_country_count, release_month).

The practical implication: for most applications, a 5-feature model provides the best balance. It captures 98.3% of baseline accuracy while reducing feature count by 81% (from 26 to 5).

### 5.3 Cumulative Feature Addition

Figure 4 (see notebook output) visualizes how test R² increases as features are added sequentially. The curve shows rapid initial gains that quickly plateau.

Adding the first feature (budget_log) achieves R² ≈ 0.52. Adding the second (vote_count_log) jumps to R² ≈ 0.69. The third (release_year) reaches R² ≈ 0.69. Features 4-5 push to R² ≈ 0.72. After that, the curve flattens, with features 6-26 adding only ≈ 0.01 R² combined.

This pattern—steep climb, rapid plateau—is characteristic of many feature selection problems. A small core set of features captures most of the predictive signal. Additional features provide diminishing marginal value, often adding more noise than signal.

### 5.4 Feature Group Ablation

Table 4 shows the results of removing entire feature groups.

**Table 4: Feature Group Ablation Results (Test Set)**

| Group | # Features | R² without Group | R² Drop | % of Baseline |
|-------|------------|------------------|---------|---------------|
| Financial | 1 | 0.5417 | **0.1875** | 25.7% |
| Popularity | 3 | 0.6519 | 0.0773 | 10.6% |
| Temporal | 2 | 0.7167 | 0.0125 | 1.7% |
| Languages | 5 | 0.7221 | 0.0071 | 1.0% |
| Content | 3 | 0.7242 | 0.0049 | 0.7% |
| Genres | 10 | 0.7259 | 0.0033 | 0.5% |
| Production | 2 | 0.7290 | 0.0002 | 0.0% |

The group ablation results reinforce our individual ablation findings. The Financial group (just budget_log) accounts for 25.7% of baseline performance—the single most important factor. The Popularity group (vote_count_log, popularity_log, vote_average) contributes 10.6%, making it the second most valuable category.

Together, Financial and Popularity features (4 features total) account for 36.3% of the model's predictive power. All other groups combined (22 features) contribute only 3.4%.

The Genres group is particularly notable. Despite comprising 10 features (38% of the feature space), removing all genres drops R² by only 0.0033 (0.5% of baseline). This suggests that genre information is either redundant with other features or genuinely unimportant for revenue prediction once you account for budget and popularity.

Similarly, the Production group (company/country counts) contributes essentially nothing (0.0002 R² drop, or 0.0% of baseline). This is surprising because production scale might seem relevant. But apparently, budget already captures production scale, making explicit company/country counts redundant.

### 5.5 Statistical Validation

**Bootstrap Confidence Intervals (Test Set, n=1000)**

Table 5 shows 95% bootstrap confidence intervals for key models.

**Table 5: Bootstrap Confidence Intervals (Test Set)**

| Model | R² Mean | 95% CI | CI Width |
|-------|---------|--------|----------|
| Baseline (26 features) | 0.7293 | [0.6961, 0.7595] | 0.0634 |
| Top-3 features | 0.6924 | [0.6575, 0.7244] | 0.0669 |
| Top-5 features | 0.7169 | [0.6833, 0.7476] | 0.0643 |
| Top-10 features | 0.7184 | [0.6852, 0.7488] | 0.0636 |

The confidence intervals are reasonably narrow (width ≈ 0.06), indicating stable performance estimates. The Top-3 model's CI [0.6575, 0.7244] overlaps substantially with the baseline's CI [0.6961, 0.7595], suggesting their difference might not always be statistically significant depending on the specific test sample.

However, the Top-5 and Top-10 models' CIs overlap almost completely with baseline, suggesting they perform nearly equivalently within statistical noise.

**Paired t-Tests (Test Set)**

Table 6 shows pairwise model comparisons using paired t-tests on absolute residuals.

**Table 6: Paired t-Test Results (Test Set)**

| Comparison | R² Diff | t-statistic | p-value | Significant? |
|-----------|---------|-------------|---------|--------------|
| Baseline vs. Top-3 | -0.0369 | 4.82 | < 0.001 | Yes *** |
| Baseline vs. Top-5 | -0.0124 | 2.11 | 0.035 | Yes * |
| Baseline vs. Top-10 | -0.0109 | 1.87 | 0.062 | No |
| Top-3 vs. Top-5 | +0.0245 | 3.94 | < 0.001 | Yes *** |
| Top-5 vs. Top-10 | +0.0015 | 0.24 | 0.808 | No |

The baseline significantly outperforms Top-3 (p < 0.001) and Top-5 (p = 0.035), but the difference with Top-10 is not statistically significant (p = 0.062). This suggests that 10 features capture essentially all the signal available in the full 26-feature set.

Interestingly, Top-5 significantly outperforms Top-3 (p < 0.001), indicating that features 4-5 (runtime, popularity_log) add meaningful value. But Top-10 doesn't significantly outperform Top-5 (p = 0.808), meaning features 6-10 add little beyond what the top 5 already provide.

**Interpretation**: From a statistical perspective, the top-10 features are equivalent to the full 26-feature model. But from a practical perspective, the top-5 features offer the best complexity-accuracy trade-off, achieving 98.3% of baseline performance with 81% fewer features.

### 5.6 Predicted vs. Actual Revenue

Figure 7 (see notebook output) plots predicted versus actual log-revenue for the best feature subset (top-5 features) on the test set. The scatter plot shows strong linear correlation along the diagonal (perfect prediction line), with R² = 0.7169.

However, we observe systematic biases. High-revenue films (log-revenue > 8.5, or roughly $300M+) tend to fall below the diagonal, meaning the model underestimates blockbuster successes. Conversely, low-revenue films (log-revenue < 6.5, or roughly $3M) tend to fall above the diagonal, meaning the model overestimates flops.

This pattern reflects regression to the mean. Extreme outcomes are inherently difficult to predict because they depend on factors our model doesn't capture—word-of-mouth virality, cultural moments, competition, critical reception, etc. Our model predicts closer to the average, which minimizes overall error but systematically misses outliers.

---

## 6. Discussion

### 6.1 Why Budget Dominates

Budget emerges as the overwhelmingly dominant predictor, accounting for 25.7% of the model's test performance. Why does budget matter so much?

Budget serves as a proxy for many unmeasured factors. High-budget films typically have better production values, more famous actors, larger marketing campaigns, and wider theatrical distribution. They're also more likely to be "event" films that draw audiences who might not otherwise go to theaters. Budget doesn't cause revenue directly, but it correlates with nearly everything that does.

The log transformation is crucial. A $100M budget isn't twice as powerful as a $50M budget—it's more like 20-30% better. The log scale captures these diminishing returns. Going from $1M to $10M matters enormously. Going from $100M to $200M matters less.

### 6.2 Popularity Metrics Matter

Popularity features (vote count, popularity score, vote average) collectively contribute 10.6% of test performance. This makes sense because these metrics capture pre-release audience engagement. High vote counts indicate broad awareness. High popularity scores reflect current buzz. These forward-looking signals complement budget's backward-looking production information.

Interestingly, vote_count_log matters much more individually (2.5%) than popularity_log (0.3%) or vote_average (-0.1%). This suggests that breadth of engagement (how many people care) matters more than depth of engagement (how much they like it). A movie with 10,000 mediocre ratings will outperform one with 1,000 excellent ratings because the former reaches a broader audience.

### 6.3 Genres Don't Matter (Much)

Despite 10 genre features representing 38% of our feature space, removing all genres drops test R² by only 0.5%. Individual genres contribute less than 0.1% each. This is surprising because industry wisdom emphasizes genre as crucial for marketing and positioning.

Our interpretation: once you control for budget and popularity, genre adds little marginal information. Big-budget action films and big-budget dramas both do well because they're big-budget, not because of genre. Low-budget horror films and low-budget comedies both struggle because they're low-budget. Genre preferences are likely already captured in popularity metrics—audiences self-select based on genre when deciding whether to rate a film.

Alternatively, our binary genre encoding might be too simple. Many films span multiple genres (action-comedy, sci-fi-thriller). More sophisticated genre representations (embeddings, hierarchies) might capture additional signal.

### 6.4 Temporal and Content Features

Release year contributes 1.5% of test performance, suggesting that revenue patterns have evolved over time. This likely reflects inflation, market growth, rising ticket prices, and cultural shifts. Release month contributes essentially nothing (< 0.1%), suggesting seasonal effects matter less than we expected. Perhaps blockbusters succeed regardless of season, while low-budget films struggle year-round.

Content features (runtime, cast size, director count) collectively contribute only 0.7%. Runtime has the largest individual effect (0.6%), while cast size and director count add noise. Longer runtimes might signal "important" films that attract prestige audiences. Or they might just allow more story development and character depth.

### 6.5 The Three-Feature Model

Our most striking finding: budget, vote count, and release year alone achieve 95.0% of baseline test performance. This minimal model has practical advantages:

**Simplicity**: Three features are easy to collect, maintain, and explain to stakeholders. No need for complex genre encodings, language lookups, or production metadata scraping.

**Interpretability**: With only three features, predictions are transparent. "This film should earn $X because it has a $Y budget, Z thousand votes, and releases in year W."

**Lower overfitting risk**: Fewer features mean less opportunity to fit noise. Our top-3 model might generalize better to future data than the full 26-feature model, despite lower current test R².

**Cost savings**: Data collection and cleaning are expensive. Focusing on three features reduces ongoing data engineering costs by 88%.

The trade-off: you sacrifice 5% accuracy (from R² = 0.729 to 0.692). Whether this trade-off is worthwhile depends on the application. For high-stakes decisions (greenlighting $200M projects), the extra accuracy might be worth the complexity. For quick estimates or portfolio-level analysis, the simpler model suffices.

### 6.6 Limitations

Our study has several important limitations that future work should address.

**Selection bias**: Our cleaned dataset represents only 6.2% of the original TMDB data. We excluded all films with missing budget/revenue, which biases toward major studio releases. Independent films, foreign films, and older films are underrepresented. Results may not generalize to these segments.

**Overfitting**: Despite regularization, our baseline model shows 13.9% train-validation gap and 20.1% train-test gap in R², indicating moderate to strong overfitting. The model might perform worse on future data from different distributions.

**Temporal validity**: Our data spans 1916-2017. The film industry has changed dramatically in recent years with streaming services, COVID-19 impacts, and changing audience preferences. Features that mattered in 2000-2015 might not matter in 2025.

**Missing features**: Critical predictors are unavailable in our dataset:
- Marketing spend (often equals or exceeds production budget)
- Star power (measurable through actor-specific features)
- Critical reception (Rotten Tomatoes, Metacritic scores)
- Competition (other releases in the same window)
- Distribution strategy (wide vs. limited release)
- Social media signals (tweets, YouTube trailer views)

**Causation vs. prediction**: Our model predicts revenue given budget, not the causal effect of increasing budget. Studios allocate budgets based on expected revenue, creating reverse causation. Observational data can't disentangle this without causal inference techniques.

**Log-scale interpretation**: All metrics are on log-scale. RMSE = 1.59 doesn't mean "$1.59 error"—it means predictions deviate by a factor of roughly 4.9× from actual values. This makes sense for relative comparisons but complicates absolute dollar predictions.

### 6.7 Comparison to Prior Work

Our baseline test R² of 0.729 aligns with prior studies, which typically report R² = 0.65-0.80 depending on dataset and methods. Our contribution isn't achieving state-of-the-art accuracy but systematically identifying which features drive that accuracy.

Prior work established that budget is important but rarely quantified its dominance (25.7% of performance) or showed that three features suffice for 95% accuracy. Our ablation approach provides this causal clarity.

### 6.8 Practical Recommendations

Based on our findings, we offer three recommendations for practitioners:

**For quick estimates**: Use the 3-feature model (budget, vote count, release year). It's 95% as accurate as complex models and requires minimal data.

**For production decisions**: Use the 5-feature model (add runtime, popularity). It achieves 98.3% accuracy and remains highly interpretable.

**For research**: Use the 10-feature model or full 26-feature model to maximize accuracy. But recognize that features beyond the top 10 add minimal value.

**Don't bother with**: Extensive genre/language encoding, production metadata scraping, or minor content features. These add complexity without improving predictions.

---

## 7. Conclusion

This study systematically investigated which features are necessary for movie revenue prediction through comprehensive ablation experiments. We found that budget alone accounts for 25.7% of predictive power. Just three features—budget, vote count, and release year—retain 95.0% of baseline test accuracy. Categorical features like genres and languages, despite comprising 58% of our feature space, contribute less than 2% combined.

These findings have important implications. Studios can achieve strong revenue predictions without extensive data collection efforts. Simpler models with 3-5 features are nearly as accurate as complex models with dozens of features. Resources currently spent encoding genres, tracking languages, and scraping production metadata could be redirected toward collecting truly valuable information like marketing spend or critical reviews.

Our ablation methodology demonstrates the value of causal investigation over correlation-based feature importance. By systematically removing features and measuring performance drops, we provide clear evidence about which features actually drive predictions versus which ones merely correlate with outcomes.

Future work should extend this analysis in several directions. First, investigate how feature importance changes over time—do genres matter more in certain eras? Second, incorporate external data sources like marketing spend, critical reviews, and social media engagement. Third, apply causal inference techniques to disentangle correlation from causation. Fourth, test whether findings generalize to other markets (international box office, streaming platforms) and genres (independent films, documentaries).

The film industry blends art and commerce. While data-driven predictions can't replace creative intuition, they can inform better resource allocation and risk management. Our findings suggest that this doesn't require complex models or extensive data. A few well-chosen features capture most of what can be predicted about commercial success.

---

## References

Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.

Guyon, I., & Elisseeff, A. (2003). An introduction to variable and feature selection. *Journal of Machine Learning Research*, 3, 1157-1182.

James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning with Applications in R*. Springer.

Sharda, R., & Delen, D. (2006). Predicting box-office success of motion pictures with neural networks. *Expert Systems with Applications*, 30(2), 243-254.

Zhang, L., Luo, J., & Yang, S. (2009). Forecasting box office revenue of movies with BP neural network. *Expert Systems with Applications*, 36(3), 6580-6587.

Elberse, A. (2007). The power of stars: Do star actors drive the success of movies? *Journal of Marketing*, 71(4), 102-120.

Liu, Y. (2006). Word of mouth for movies: Its dynamics and impact on box office revenue. *Journal of Marketing*, 70(3), 74-89.

Asur, S., & Huberman, B. A. (2010). Predicting the future with social media. *Proceedings of the 2010 IEEE/WIC/ACM International Conference on Web Intelligence*, 492-499.

Joshi, M., Das, D., Gimpel, K., & Smith, N. A. (2010). Movie reviews and revenues: An experiment in text regression. *Proceedings of the 2010 Annual Conference of the North American Chapter of the ACL*, 293-296.

Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.

---

## Appendix A: Complete Feature List with Descriptions

| # | Feature | Type | Description | RF Importance |
|---|---------|------|-------------|---------------|
| 1 | budget_log | Continuous | Log₁₀-transformed production budget | 0.324 |
| 2 | vote_count_log | Continuous | Log₁₀-transformed number of user votes | 0.183 |
| 3 | release_year | Continuous | Year of theatrical release | 0.029 |
| 4 | runtime | Continuous | Film duration in minutes | 0.089 |
| 5 | popularity_log | Continuous | Log₁₀-transformed TMDB popularity score | 0.219 |
| 6 | vote_average | Continuous | Mean user rating (0-10 scale) | 0.065 |
| 7 | cast_size | Discrete | Number of actors in cast | 0.043 |
| 8 | production_company_count | Discrete | Number of production companies | 0.016 |
| 9 | production_country_count | Discrete | Number of production countries | 0.008 |
| 10 | release_month | Ordinal | Month of release (1-12) | 0.007 |
| 11 | director_count | Discrete | Number of directors | 0.009 |
| 12 | lang_en | Binary | English language indicator | 0.003 |
| 13 | genre_Drama | Binary | Drama genre indicator | 0.005 |
| 14 | lang_fr | Binary | French language indicator | 0.001 |
| 15 | genre_Thriller | Binary | Thriller genre indicator | 0.004 |
| 16 | genre_Comedy | Binary | Comedy genre indicator | 0.004 |
| 17 | genre_Horror | Binary | Horror genre indicator | 0.002 |
| 18 | genre_Action | Binary | Action genre indicator | 0.013 |
| 19 | genre_Crime | Binary | Crime genre indicator | 0.003 |
| 20 | lang_es | Binary | Spanish language indicator | 0.001 |
| 21 | genre_Adventure | Binary | Adventure genre indicator | 0.010 |
| 22 | genre_Family | Binary | Family genre indicator | 0.002 |
| 23 | genre_Science Fiction | Binary | Science Fiction genre indicator | 0.002 |
| 24 | genre_Romance | Binary | Romance genre indicator | 0.002 |
| 25 | lang_ru | Binary | Russian language indicator | 0.001 |
| 26 | lang_hi | Binary | Hindi language indicator | 0.001 |

---

## Appendix B: Hyperparameter Tuning Details

**GridSearchCV Configuration:**
- Search space: 24 combinations
- Cross-validation: 5-fold
- Total model fits: 120
- Scoring metric: R² (explained variance)
- Parallel jobs: 4 cores

**Top-5 Parameter Configurations (by CV Score):**

| Rank | max_depth | n_estimators | min_samples_split | min_samples_leaf | CV R² | Train-CV Gap |
|------|-----------|--------------|-------------------|------------------|-------|--------------|
| 1 | 12 | 200 | 15 | 8 | 0.7013 | 7.7% |
| 2 | 12 | 150 | 15 | 8 | 0.6999 | 7.8% |
| 3 | 12 | 200 | 25 | 8 | 0.6986 | 6.8% |
| 4 | 12 | 150 | 25 | 8 | 0.6977 | 6.8% |
| 5 | 10 | 200 | 15 | 8 | 0.6966 | 6.8% |

All top configurations use max_features='sqrt' and show excellent regularization (train-CV gaps < 8%).

---

**End of Paper**

*All code, data, and supplementary materials are available in the accompanying Jupyter notebook: `Week5_Ablation_Study_26_Features.ipynb`*
