# What Makes a Movie Sell? A Systematic Ablation Study on Box Office Revenue Prediction

**Authors:** [Your Names]
**Course:** Applied AI I - Winter 2025
**Instructor:** Prof. Dr. Sigurd Schacht
**Date:** November 2025

---

## Abstract

This study investigates which features contribute most to predicting movie box office revenue through a systematic ablation study on the TMDB Movie Dataset. Using a Random Forest regression model with hyperparameter optimization via cross-validation, we evaluate 26 engineered features including financial metrics, popularity indicators, temporal information, and categorical encodings for genres and languages. Our baseline model achieves R² = 0.739 on validation data. Through individual feature ablation, cumulative feature addition, and top-k feature selection experiments, we identify that log-transformed budget, popularity, and vote count emerge as the most critical predictors. Remarkably, a reduced model using only 7 features achieves 99.4% of baseline performance (R² = 0.735), demonstrating that high prediction accuracy can be maintained with 73% fewer features. Statistical validation via bootstrap confidence intervals and paired t-tests confirms the robustness of these findings. Our results have practical implications for production studios and investors, suggesting that comprehensive data collection efforts can be significantly reduced without sacrificing predictive accuracy.

**Keywords:** Feature Selection, Ablation Study, Random Forest, Movie Revenue Prediction, Machine Learning

---

## 1. Introduction

Predicting box office revenue is a critical challenge for the entertainment industry. Production studios invest hundreds of millions of dollars in film projects, yet commercial success remains highly uncertain. While traditional industry wisdom emphasizes factors like star power and marketing budgets, a data-driven approach can systematically identify which features genuinely predict revenue.

### 1.1 Motivation

Previous research has explored various factors influencing box office performance, but often with limited feature engineering or without systematically evaluating feature importance. The rise of machine learning enables us to analyze comprehensive datasets and quantify the contribution of individual features. Understanding which features matter most has three key benefits: (1) reducing data collection costs, (2) improving model interpretability for decision-makers, and (3) focusing resources on the factors that truly drive commercial success.

### 1.2 Research Question

**"Which features matter most for predicting movie box office revenue, and can we achieve similar performance with fewer features?"**

### 1.3 Contributions

This paper makes the following contributions:

1. **Comprehensive Feature Engineering:** We construct 26 features from raw TMDB data, including log-transformed financial metrics, one-hot encoded genres and languages, and temporal features.

2. **Systematic Ablation Study:** We conduct four types of ablation experiments—individual feature removal, cumulative addition, feature group analysis, and top-k selection—to rigorously evaluate feature importance.

3. **Statistical Validation:** We employ bootstrap confidence intervals (n=1000 iterations) and paired t-tests to ensure our findings are statistically robust rather than artifacts of random variation.

4. **Practical Insights:** We demonstrate that 7 carefully selected features can match 99.4% of the performance achieved with all 26 features, providing actionable guidance for practitioners.

### 1.4 Paper Organization

The remainder of this paper is structured as follows: Section 2 reviews related work on feature selection and movie revenue prediction. Section 3 describes the TMDB dataset and preprocessing steps. Section 4 details our methodology including model selection, feature selection methods, and experimental protocol. Section 5 presents experimental results. Section 6 discusses implications and limitations. Section 7 concludes with recommendations for future work.

---

## 2. Related Work

Feature selection is a fundamental problem in machine learning with three main categories of approaches: filter methods (correlation, mutual information), wrapper methods (recursive feature elimination), and embedded methods (tree-based importance, L1 regularization) (Guyon & Elisseeff, 2003; Chandrashekar & Sahin, 2014).

Ablation studies have proven valuable for understanding neural network components (Merity et al., 2017), but their application to traditional machine learning models and feature analysis remains less common. Our work applies ablation methodology systematically to feature importance analysis.

In the domain of movie revenue prediction, researchers have explored various factors including social media sentiment (Asur & Huberman, 2010), cast characteristics (Elberse, 2007), and genre classifications. However, few studies have conducted rigorous ablation experiments to quantify the marginal contribution of individual features while controlling for other variables.

---

## 3. Dataset and Preprocessing

### 3.1 Dataset Description

This study uses the TMDB Movie Dataset, which is publicly available on Kaggle. The dataset contains comprehensive information about movies from The Movie Database (TMDB). It is particularly well-suited for an ablation study because it offers a variety of interpretable features that cover different aspects of film production.

The original dataset comprises 119,938 movie entries with 27 attributes. The target variable is movie revenue, measured in US dollars. This metric was chosen because it objectively quantifies a film's commercial success. Predicting movie revenue is of considerable practical relevance for production studios and investors.

### 3.2 Feature Overview

Table 1 shows the features used in the model after feature engineering.

**Table 1: Feature Groups**

| Feature Group | Count | Description |
|---------------|-------|-------------|
| Numerical (original) | 8 | runtime, vote_average, director_count, production_company_count, production_country_count, release_year, release_month, cast_size |
| Log-transformed | 3 | budget_log, popularity_log, vote_count_log |
| Genre (One-Hot) | 11 | Binary indicators for film genres (Action, Comedy, Drama, etc.) |
| Language (One-Hot) | 5 | Binary indicators for the most common original languages |
| **Total** | **26** | |

The numerical features capture basic film characteristics. Budget represents production costs, while *popularity* and *vote_count* reflect public attention. The variable *vote_average* indicates the mean user rating. Additionally, derived features were extracted from the raw data: the number of directors, production companies, production countries, and cast size.

### 3.3 Preprocessing

#### Missing Value Handling

The dataset exhibits considerable gaps. Particularly affected are the columns *belongs_to_collection* (92.1% missing) and *homepage* (83.2% missing). These columns were not used for the analysis. For the core variables *budget* and *revenue*, less than 1% of values are missing.

#### Outlier Treatment

Data cleaning was performed in several steps. First, all movies without documented budget or revenue were excluded. This affects 87.5% of entries with budget equal to zero and 89.9% with revenue equal to zero. This filtering is necessary because missing financial information would substantially impair prediction quality.

Subsequently, movies with unrealistic runtime were removed. The criterion is: 30 ≤ runtime ≤ 300 minutes. Two extreme outliers were thereby eliminated. After cleaning, 7,380 movies remain, corresponding to 6.2% of the original dataset.

This substantial reduction warrants mention. The cleaned dataset primarily contains commercially released films with complete financial documentation. These are predominantly major studio productions. This selection bias must be considered when interpreting the results.

#### Feature Scaling

All features were normalized using StandardScaler. The scaler was fitted exclusively on the training data and subsequently applied to validation and test data. This procedure prevents data leakage.

#### Log-Transformation

The variables *budget*, *popularity*, and *vote_count* exhibit strongly right-skewed distributions. To normalize these, a logarithmic transformation was applied. The target variable *revenue* was also log-transformed, as movie revenues typically vary across several orders of magnitude.

#### Train/Validation/Test Split

The data was divided into three disjoint subsets prior to any feature analysis. The training set contains 5,166 samples, representing 70% of the data, and is used exclusively for model training. The validation set comprises 1,107 samples (15%) and serves for conducting the ablation study as well as hyperparameter optimization. Finally, the test set contains another 1,107 samples (15%) and remains untouched until final evaluation to ensure an unbiased performance estimate.

This three-way split strategy was deliberately chosen over a simple train-test split. By separating validation from test data, we avoid optimistic bias that could arise from tuning model parameters on the same data used for final evaluation.

---

## 4. Methodology

### 4.1 Model Selection

We employ Random Forest Regression as our predictive model for the following reasons:

**Advantages for this task:**
- Handles mixed feature types (numerical, categorical) naturally
- Provides built-in feature importance estimates
- Robust to outliers and non-linear relationships
- No strong assumptions about data distributions

**Hyperparameter Optimization:**

To mitigate overfitting observed in preliminary experiments (training R² = 0.911 vs. validation R² = 0.739), we implement regularization through hyperparameter tuning. A grid search with 5-fold cross-validation explores the following parameter space:

**Table 2: Hyperparameter Search Space**

| Parameter | Values Tested | Purpose |
|-----------|---------------|---------|
| `max_depth` | [8, 10, 12] | Limit tree depth to prevent overfitting |
| `n_estimators` | [150, 200] | Number of trees in the forest |
| `min_samples_split` | [15, 25] | Minimum samples required to split a node |
| `min_samples_leaf` | [8, 12] | Minimum samples required in leaf nodes |
| `max_features` | ['sqrt'] | Use only √26 ≈ 5 features per split |

The grid search evaluates 24 combinations (120 fits with 5-fold CV), optimizing for R² score. The best configuration is used consistently across all ablation experiments to ensure fair comparison.

### 4.2 Feature Selection Methods

We implement two feature selection approaches:

**Method 1: Correlation Analysis (Filter Method)**

Compute Pearson correlation between each feature and the log-transformed target variable on training data. Features are ranked by absolute correlation coefficient.

**Method 2: Random Forest Feature Importance (Embedded Method)**

After training the baseline model, extract feature importance scores based on mean decrease in impurity (Gini importance). This captures non-linear relationships and feature interactions.

### 4.3 Ablation Study Protocol

We conduct four complementary ablation experiments:

**Experiment 1: Individual Feature Ablation**

For each of the 26 features:
1. Remove the feature from the training set
2. Train a Random Forest model with remaining 25 features
3. Evaluate on validation set
4. Record the drop in R² score: Δ R² = R²_baseline - R²_ablated

Features causing the largest performance drops are deemed most important.

**Experiment 2: Cumulative Feature Addition**

Starting with an empty feature set:
1. Add the most important feature (by Random Forest importance)
2. Train model and evaluate
3. Add the next most important feature
4. Repeat until all features are included

This reveals the marginal contribution of each additional feature and identifies the point of diminishing returns.

**Experiment 3: Feature Group Ablation**

We group features by semantic category:
- Financial: budget_log
- Popularity: popularity_log, vote_count_log, vote_average
- Temporal: release_year, release_month
- Production: cast_size, director_count, production_company_count, production_country_count
- Genres: 11 one-hot encoded genre features
- Languages: 5 one-hot encoded language features

Each group is removed entirely, and the performance drop indicates the group's collective importance.

**Experiment 4: Top-K Feature Selection**

We train models using only the top-3, top-5, top-7, and top-10 features (by Random Forest importance). This identifies the optimal feature subset balancing performance and complexity.

### 4.4 Statistical Validation

**Bootstrap Confidence Intervals**

For each model configuration, we compute 95% confidence intervals for R², RMSE, and MAE using bootstrap resampling (n=1000 iterations). This quantifies uncertainty in our performance estimates.

**Paired T-Tests**

We conduct paired t-tests comparing absolute residuals between model pairs (e.g., Baseline vs. Top-7). This determines whether performance differences are statistically significant (α = 0.05).

### 4.5 Evaluation Metrics

**Primary Metric:** R² (Coefficient of Determination) on validation set. Measures the proportion of variance explained by the model.

**Secondary Metrics:**
- RMSE (Root Mean Squared Error) on log-scale
- MAE (Mean Absolute Error) on log-scale
- Percentage of baseline performance: (R²_reduced / R²_baseline) × 100%

All metrics are computed on log-transformed revenue to account for the skewed distribution of movie earnings.

---

## 5. Experiments and Results

### 5.1 Baseline Model Performance

After hyperparameter optimization via grid search, the best configuration achieves:

**Best Hyperparameters (5-Fold CV):**
- max_depth: 12
- n_estimators: 200
- min_samples_split: 15
- min_samples_leaf: 8
- max_features: 'sqrt'
- CV Score (R²): 0.7327

**Performance on Hold-Out Sets:**

**Table 3: Baseline Model Performance**

| Split | R² | RMSE (log) | MAE (log) | Error Factor |
|-------|-----|------------|-----------|--------------|
| Training | 0.8650 | 0.5214 | 0.3892 | 1.7x |
| Validation | 0.7394 | 0.8092 | 0.5834 | 2.2x |
| Test | 0.7301 | 0.8231 | 0.5912 | 2.3x |

**Overfitting Analysis:**

The training-validation gap (R² difference: 0.1256) indicates moderate overfitting despite regularization efforts. However, the validation and test scores are consistent (R² difference: 0.0093), confirming reliable generalization to unseen data.

### 5.2 Feature Importance Rankings

**Table 4: Top-10 Features by Importance Method**

| Rank | Random Forest Importance | Score | Correlation with Target | Score |
|------|-------------------------|-------|------------------------|-------|
| 1 | budget_log | 0.3245 | budget_log | 0.6821 |
| 2 | popularity_log | 0.2187 | popularity_log | 0.5934 |
| 3 | vote_count_log | 0.1834 | vote_count_log | 0.5612 |
| 4 | runtime | 0.0892 | vote_average | 0.3145 |
| 5 | vote_average | 0.0645 | runtime | 0.2876 |
| 6 | cast_size | 0.0432 | cast_size | 0.2134 |
| 7 | release_year | 0.0289 | release_year | 0.1987 |
| 8 | production_company_count | 0.0156 | genre_Action | 0.1543 |
| 9 | genre_Action | 0.0134 | production_company_count | 0.1298 |
| 10 | genre_Adventure | 0.0098 | genre_Adventure | 0.1102 |

**Key Observations:**
- **Strong agreement:** Both methods identify budget_log, popularity_log, and vote_count_log as the top-3 features
- **Log transformation crucial:** The log-transformed financial/popularity metrics dominate importance rankings
- **Genre features:** Action and Adventure genres show modest predictive power
- **Temporal features:** release_year appears in top-10, suggesting recent films have different revenue patterns

### 5.3 Individual Feature Ablation Results

Figure 3 displays the performance drop (Δ R²) when each feature is individually removed from the full model.

**Table 5: Individual Feature Ablation - Top-10 Critical Features**

| Rank | Feature Removed | R² (Ablated) | Δ R² | Impact |
|------|----------------|--------------|------|--------|
| 1 | budget_log | 0.6142 | -0.1252 | Critical |
| 2 | popularity_log | 0.6589 | -0.0805 | Critical |
| 3 | vote_count_log | 0.6734 | -0.0660 | Critical |
| 4 | runtime | 0.7198 | -0.0196 | Moderate |
| 5 | vote_average | 0.7256 | -0.0138 | Moderate |
| 6 | cast_size | 0.7301 | -0.0093 | Minor |
| 7 | release_year | 0.7329 | -0.0065 | Minor |
| 8 | genre_Action | 0.7362 | -0.0032 | Minor |
| 9 | production_company_count | 0.7378 | -0.0016 | Minimal |
| 10 | genre_Adventure | 0.7385 | -0.0009 | Minimal |

**Key Findings:**
- **Three critical features:** Removing budget_log, popularity_log, or vote_count_log causes substantial performance degradation (Δ R² > 0.05)
- **Long tail of minimal impact:** 15 features (58%) cause performance drops of less than 0.005 when removed
- **Redundancy observed:** Many genre and language features show near-zero individual impact, suggesting they provide overlapping information

### 5.4 Cumulative Feature Addition Results

Figure 4 shows the performance curve as features are added incrementally.

**Table 6: Cumulative Feature Addition - Selected Milestones**

| # Features | Features Included | R² | % of Baseline |
|-----------|-------------------|-----|---------------|
| 1 | budget_log | 0.4652 | 62.9% |
| 2 | +popularity_log | 0.6234 | 84.3% |
| 3 | +vote_count_log | 0.6987 | 94.5% |
| 5 | +runtime, vote_average | 0.7189 | 97.2% |
| 7 | +cast_size, release_year | 0.7350 | 99.4% |
| 10 | +genre_Action, prod_company_count, genre_Adventure | 0.7378 | 99.8% |
| 26 | All features | 0.7394 | 100.0% |

**Key Findings:**
- **Rapid initial gains:** The first 3 features capture 94.5% of baseline performance
- **Diminishing returns:** Adding features beyond 7 yields marginal improvements (<0.5%)
- **Optimal subset:** 7 features achieve 99.4% of baseline performance while reducing feature count by 73%
- **Inflection point:** The performance curve plateaus around 5-7 features

### 5.5 Feature Group Ablation Results

**Table 7: Feature Group Ablation**

| Group | # Features | R² (Group Removed) | Δ R² | Interpretation |
|-------|-----------|-------------------|------|----------------|
| Financial | 1 | 0.6142 | -0.1252 | Critical - single most important feature |
| Popularity | 3 | 0.5987 | -0.1407 | Critical - collectively essential |
| Production | 4 | 0.7123 | -0.0271 | Moderate - provides context |
| Temporal | 2 | 0.7298 | -0.0096 | Minor - some predictive value |
| Genres | 11 | 0.7334 | -0.0060 | Minor - modest collective impact |
| Languages | 5 | 0.7381 | -0.0013 | Minimal - largely redundant |

**Key Findings:**
- **Popularity metrics dominate:** Removing the 3 popularity features (popularity_log, vote_count_log, vote_average) causes the largest performance drop
- **Genre paradox:** Despite 11 genre features, their collective removal only drops R² by 0.006
- **Language features redundant:** The 5 language indicators contribute virtually nothing (Δ R² = 0.0013)

### 5.6 Top-K Feature Selection Results

**Table 8: Top-K Feature Selection Performance**

| Model | Features | R² | 95% CI | % of Baseline | RMSE (log) |
|-------|----------|-----|--------|---------------|-----------|
| Baseline | All 26 | 0.7394 | [0.7312, 0.7476] | 100.0% | 0.8092 |
| Top-3 | budget_log, popularity_log, vote_count_log | 0.6987 | [0.6898, 0.7076] | 94.5% | 0.8654 |
| Top-5 | +runtime, vote_average | 0.7189 | [0.7102, 0.7276] | 97.2% | 0.8389 |
| Top-7 | +cast_size, release_year | 0.7350 | [0.7265, 0.7435] | 99.4% | 0.8145 |
| Top-10 | +genre_Action, prod_company_count, genre_Adventure | 0.7378 | [0.7294, 0.7462] | 99.8% | 0.8112 |

**Key Findings:**
- **Top-7 optimal:** Achieves 99.4% of baseline with 73% fewer features
- **Confidence intervals:** All models show stable performance with narrow CIs (width ≈ 0.016)
- **Practical recommendation:** Top-7 model offers the best performance-complexity trade-off

Figure 5 visualizes the performance-feature count relationship, clearly showing diminishing returns beyond 7 features.

### 5.7 Statistical Validation

**Bootstrap Confidence Intervals (n=1000 iterations):**

The narrow confidence intervals in Table 8 confirm that our performance estimates are robust. The R² CI widths range from 0.0164 to 0.0178, representing approximately ±1.1% uncertainty.

**Paired T-Tests (α = 0.05):**

**Table 9: Statistical Significance of Model Comparisons**

| Comparison | R² Difference | t-statistic | p-value | Significant? |
|-----------|---------------|-------------|---------|--------------|
| Baseline vs. Top-3 | -0.0407 | 12.45 | <0.001 | Yes *** |
| Baseline vs. Top-5 | -0.0205 | 7.89 | <0.001 | Yes *** |
| Baseline vs. Top-7 | -0.0044 | 1.34 | 0.180 | No (n.s.) |
| Baseline vs. Top-10 | -0.0016 | 0.48 | 0.632 | No (n.s.) |
| Top-3 vs. Top-5 | +0.0202 | 7.76 | <0.001 | Yes *** |
| Top-5 vs. Top-7 | +0.0161 | 6.21 | <0.001 | Yes *** |
| Top-7 vs. Top-10 | +0.0028 | 0.85 | 0.395 | No (n.s.) |

**Key Findings:**
- **Top-7 statistically equivalent to baseline:** p = 0.180 (not significant)
- **Top-3 and Top-5 significantly worse:** p < 0.001
- **Top-10 vs. Top-7:** No significant difference (p = 0.395), confirming features beyond 7 add no value

This statistical validation strongly supports our recommendation of the 7-feature model.

---

## 6. Discussion

### 6.1 Which Features Matter Most?

Our ablation study consistently identifies three features as critically important across all experimental approaches:

**1. Budget (log-transformed):** The single most important predictor (RF importance: 0.3245, correlation: 0.6821). High-budget films have more resources for production quality, marketing, and wide distribution. The log transformation captures the diminishing returns of budget increases—doubling a $10M budget has more impact than doubling a $100M budget.

**2. Popularity (log-transformed):** Second most important (RF importance: 0.2187, correlation: 0.5934). This metric from TMDB reflects pre-release buzz and anticipation. It aggregates signals like search volume, social media mentions, and watchlist additions. Films generating high pre-release interest tend to succeed commercially.

**3. Vote Count (log-transformed):** Third most important (RF importance: 0.1834, correlation: 0.5612). This represents audience engagement breadth. More votes indicate wider audience reach, even if average ratings vary. The log transformation addresses the extreme range (from hundreds to tens of thousands of votes).

**Secondary Features (Moderate Impact):**
- **Runtime:** Longer films may signal epic scope or allow for richer storytelling
- **Vote Average:** Quality indicator, though less predictive than vote count
- **Cast Size:** Ensemble films may attract diverse audiences

**Tertiary Features (Minimal Impact):**
- Genre indicators show weak individual effects but modest collective value
- Temporal features (release year/month) capture industry trends
- Production metrics (company count, country count) provide limited information

### 6.2 Why Do These Features Matter?

**Budget as a Proxy:** Budget correlates with many unmeasured factors—star power, visual effects quality, marketing spend, and distribution reach. It serves as a comprehensive proxy for production value.

**Popularity as Forward-Looking:** Unlike historical data (past box office), popularity measures real-time audience interest. This makes it particularly valuable for prediction, as it captures momentum before theatrical release.

**Vote Count as Engagement:** Unlike vote average (which can be manipulated by small, devoted fanbases), vote count requires broad engagement. It distinguishes niche films from mainstream appeal.

**Log Transformation Crucial:** The log transformation of budget, popularity, and vote count is essential. These variables span multiple orders of magnitude (budgets: $1M to $300M+). Log transformation normalizes distributions and captures percentage changes rather than absolute differences.

### 6.3 Genre and Language Findings

**Genre Paradox:** Despite encoding 11 genres, their collective impact is minimal (Δ R² = 0.006). This suggests:
- Genre preferences may be captured by popularity metrics (audiences self-select based on genre)
- Within-genre variation is larger than between-genre variation
- Financial and popularity features dominate regardless of genre

**Language Features Redundant:** The 5 language indicators contribute almost nothing (Δ R² = 0.0013). English-language films dominate the dataset (likely due to TMDB's bias toward Hollywood releases), providing little discriminative power.

### 6.4 Performance vs. Complexity Trade-Off

**The 7-Feature Model:**

Our recommended configuration uses only:
1. budget_log
2. popularity_log
3. vote_count_log
4. runtime
5. vote_average
6. cast_size
7. release_year

This achieves 99.4% of baseline performance (R² = 0.7350 vs. 0.7394), a difference that is statistically insignificant (p = 0.180). The practical benefits are substantial:
- **73% reduction in features:** Simpler data collection and maintenance
- **Improved interpretability:** Easier to explain predictions to stakeholders
- **Reduced overfitting risk:** Fewer features mean less model complexity
- **Lower computational cost:** Faster training and inference

### 6.5 Comparison to Prior Work

Previous movie revenue prediction studies report R² values ranging from 0.65 to 0.80, depending on dataset size, features, and methodology. Our baseline performance (R² = 0.74) falls within this range and is competitive given that:
- We use publicly available data (no proprietary marketing spend or distribution data)
- We focus on feature engineering and selection rather than complex ensemble methods
- We prioritize interpretability over maximizing predictive accuracy

Our contribution lies not in achieving the highest possible R², but in systematically identifying which features drive predictions and quantifying their marginal contributions.

### 6.6 Practical Implications

**For Production Studios:**
- Focus data collection efforts on budget planning, generating pre-release buzz (captured by popularity metrics), and broad audience engagement
- Genre and language considerations appear less critical for revenue prediction than commonly assumed
- The 7-feature model provides a practical tool for quick revenue estimates during greenlighting decisions

**For Investors:**
- Budget and popularity metrics provide 95% of predictive power with just 3 features
- Early engagement signals (vote count) offer valuable forward-looking information
- Overly complex models with dozens of features provide minimal additional insight

**For Researchers:**
- Ablation studies reveal insights that aggregate feature importance scores cannot
- Log transformation of financial and engagement metrics is essential
- Beware of selection bias—this dataset overrepresents films with complete financial documentation

### 6.7 Limitations

**Selection Bias:** Our cleaned dataset (7,380 films) represents only 6.2% of the original TMDB data. Films without reported budgets/revenues are excluded, biasing toward major studio releases. Independent and international films are underrepresented. Results may not generalize to the broader film market.

**Moderate Overfitting:** Despite regularization, training R² (0.865) exceeds validation R² (0.739) by 0.126. This indicates the model captures some training-specific patterns. However, consistent validation/test performance (R² difference: 0.009) suggests reliable generalization to unseen data.

**Temporal Validity:** Our dataset includes films through 2024. Industry dynamics change—streaming services, pandemic impacts, and social media's role have evolved. The relative importance of features may shift over time.

**Unmeasured Factors:** Critical predictors are unavailable in our dataset:
- Marketing spend (often equals or exceeds production budget)
- Distribution strategy (wide vs. limited release)
- Star power (measured imperfectly through cast size)
- Critical reviews (Rotten Tomatoes scores, Metacritic ratings)
- Competition (releases by other studios in the same window)

**Correlation vs. Causation:** Our study identifies predictive features, not causal drivers. High popularity predicts revenue, but intervening to increase popularity (e.g., via marketing) doesn't guarantee proportional revenue increases due to confounding factors.

**Log-Scale Interpretation:** All metrics are computed on log-transformed revenue. An RMSE of 0.81 log-units means predictions deviate by a factor of e^0.81 ≈ 2.2x from actual revenue on average. This is substantial in dollar terms for blockbuster films.

---

## 7. Conclusion

This paper systematically investigates feature importance for movie revenue prediction through a rigorous ablation study. Our key findings are:

**Main Result:** Three log-transformed features—budget, popularity, and vote count—dominate predictive performance. These capture production scale, pre-release buzz, and audience engagement breadth.

**Optimal Feature Subset:** A model using 7 carefully selected features achieves 99.4% of baseline performance (R² = 0.735 vs. 0.739) while reducing feature count by 73%. This difference is statistically insignificant (p = 0.180), making it our recommended configuration for practical deployment.

**Methodology Contribution:** We demonstrate the value of ablation studies for understanding feature importance. Our four-pronged approach—individual ablation, cumulative addition, group ablation, and top-k selection—provides complementary perspectives that converge on consistent conclusions.

**Statistical Rigor:** Bootstrap confidence intervals and paired t-tests validate our findings, ensuring they are not artifacts of random sampling variation.

### 7.1 Recommendations for Practice

1. **Prioritize data quality for the Top-7 features:** Budget, popularity, vote count, runtime, vote average, cast size, and release year provide maximum predictive value per data collection effort.

2. **Invest in pre-release engagement:** Popularity and vote count capture audience interest before theatrical release. Marketing strategies should focus on generating measurable buzz.

3. **Question conventional wisdom on genres:** Genre classifications show minimal predictive power once financial and popularity metrics are accounted for. Studios should focus on making high-quality, well-marketed films regardless of genre.

4. **Use the 7-feature model for rapid estimates:** During greenlighting decisions, a simple model can provide quick revenue forecasts without requiring comprehensive data collection.

### 7.2 Future Work

**Temporal Analysis:** Investigate how feature importance changes over time. Has social media made popularity metrics more predictive? Do genres matter more in certain eras?

**Causal Inference:** Apply causal inference techniques (instrumental variables, regression discontinuity) to identify which features are not merely predictive but causally drive revenue.

**Expanded Features:** Incorporate external data sources—marketing spend, critical reviews, social media sentiment, and competitive releases in the same window.

**Domain Adaptation:** Extend the analysis to streaming platforms where "success" is measured differently (viewing hours, subscriber retention) rather than box office revenue.

**Interaction Effects:** Systematically study feature interactions. Does budget matter more for certain genres? Do popularity thresholds create non-linear effects?

**Global Markets:** Analyze international box office separately from domestic. Feature importance may differ across markets (e.g., star power in China vs. US).

### 7.3 Closing Remarks

The entertainment industry operates on a blend of art and commerce. While data-driven approaches cannot replace creative intuition, they can inform better decision-making. Our findings suggest that commercial success, while multifaceted, is largely predicted by a small set of quantifiable factors. Production scale (budget), audience anticipation (popularity), and engagement breadth (vote count) emerge as the dominant signals.

For an industry that routinely invests hundreds of millions of dollars per project, understanding these patterns can reduce financial risk and improve capital allocation. Our 7-feature model provides a practical tool that balances predictive accuracy, interpretability, and data collection efficiency.

The code, data, and supplementary materials for this study are available in the accompanying Jupyter notebook.

---

## References

Asur, S., & Huberman, B. A. (2010). Predicting the future with social media. *IEEE/WIC/ACM International Conference on Web Intelligence and Intelligent Agent Technology*, 1, 492-499.

Chandrashekar, G., & Sahin, F. (2014). A survey on feature selection methods. *Computers & Electrical Engineering*, 40(1), 16-28.

Elberse, A. (2007). The power of stars: Do star actors drive the success of movies? *Journal of Marketing*, 71(4), 102-120.

Guyon, I., & Elisseeff, A. (2003). An introduction to variable and feature selection. *Journal of Machine Learning Research*, 3, 1157-1182.

Merity, S., Keskar, N. S., & Socher, R. (2017). Regularizing and optimizing LSTM language models. *arXiv preprint arXiv:1708.02182*.

---

## Appendix A: Complete Feature List

**Table A1: All 26 Features with Descriptions**

| # | Feature Name | Type | Description | Importance |
|---|--------------|------|-------------|------------|
| 1 | budget_log | Numerical (log) | Log-transformed production budget | 0.3245 |
| 2 | popularity_log | Numerical (log) | Log-transformed TMDB popularity score | 0.2187 |
| 3 | vote_count_log | Numerical (log) | Log-transformed number of user votes | 0.1834 |
| 4 | runtime | Numerical | Film duration in minutes | 0.0892 |
| 5 | vote_average | Numerical | Mean user rating (0-10 scale) | 0.0645 |
| 6 | cast_size | Numerical | Number of actors in cast | 0.0432 |
| 7 | release_year | Numerical | Year of theatrical release | 0.0289 |
| 8 | production_company_count | Numerical | Number of production companies | 0.0156 |
| 9 | genre_Action | Binary | Action genre indicator | 0.0134 |
| 10 | genre_Adventure | Binary | Adventure genre indicator | 0.0098 |
| 11 | director_count | Numerical | Number of directors | 0.0087 |
| 12 | production_country_count | Numerical | Number of production countries | 0.0076 |
| 13 | release_month | Numerical | Month of release (1-12) | 0.0065 |
| 14 | genre_Drama | Binary | Drama genre indicator | 0.0054 |
| 15 | genre_Comedy | Binary | Comedy genre indicator | 0.0043 |
| 16 | genre_Thriller | Binary | Thriller genre indicator | 0.0038 |
| 17 | lang_en | Binary | English language indicator | 0.0032 |
| 18 | genre_Crime | Binary | Crime genre indicator | 0.0029 |
| 19 | genre_Romance | Binary | Romance genre indicator | 0.0024 |
| 20 | genre_Science Fiction | Binary | Science Fiction genre indicator | 0.0021 |
| 21 | genre_Horror | Binary | Horror genre indicator | 0.0018 |
| 22 | genre_Fantasy | Binary | Fantasy genre indicator | 0.0015 |
| 23 | lang_fr | Binary | French language indicator | 0.0012 |
| 24 | lang_es | Binary | Spanish language indicator | 0.0009 |
| 25 | lang_de | Binary | German language indicator | 0.0007 |
| 26 | lang_ja | Binary | Japanese language indicator | 0.0005 |

---

## Appendix B: Hyperparameter Tuning Results

**Table B1: GridSearchCV Results - Top 10 Configurations**

| Rank | max_depth | n_estimators | min_samples_split | min_samples_leaf | CV R² | Std Dev |
|------|-----------|--------------|-------------------|------------------|-------|---------|
| 1 | 12 | 200 | 15 | 8 | 0.7327 | 0.0332 |
| 2 | 12 | 200 | 15 | 12 | 0.7325 | 0.0334 |
| 3 | 10 | 200 | 15 | 8 | 0.7319 | 0.0329 |
| 4 | 12 | 150 | 15 | 8 | 0.7315 | 0.0331 |
| 5 | 10 | 200 | 15 | 12 | 0.7312 | 0.0327 |
| 6 | 8 | 200 | 15 | 8 | 0.7305 | 0.0325 |
| 7 | 12 | 200 | 25 | 8 | 0.7301 | 0.0336 |
| 8 | 12 | 150 | 15 | 12 | 0.7298 | 0.0328 |
| 9 | 10 | 150 | 15 | 8 | 0.7294 | 0.0324 |
| 10 | 8 | 200 | 15 | 12 | 0.7289 | 0.0322 |

**Observations:**
- Best configuration: max_depth=12, n_estimators=200, min_samples_split=15, min_samples_leaf=8
- Top configurations show similar performance (R² range: 0.7289 - 0.7327)
- Deeper trees (max_depth=12) slightly outperform shallower trees
- Larger forests (n_estimators=200) consistently perform better than 150
- Regularization parameters (min_samples_split, min_samples_leaf) successfully mitigate overfitting

---

## Appendix C: Figure Descriptions

**Figure 1:** Feature Correlation Matrix (before log-transformation) - Shows high correlation between budget and revenue (0.72)

**Figure 1b:** Feature Correlation Matrix (after log-transformation) - Displays correlation structure of final 26 features with target

**Figure 2:** Feature Importance Comparison - Bar chart comparing Random Forest importance vs. correlation-based rankings

**Figure 3:** Individual Feature Ablation - Horizontal bar chart showing R² drop when each feature is removed

**Figure 4:** Cumulative Feature Addition Curve - Two-panel plot showing (left) R² growth and (right) percentage of baseline as features are added

**Figure 5:** Performance vs. Number of Features - Scatter plot with trend line illustrating diminishing returns

**Figure 6:** Feature Group Contributions - Bar chart displaying collective impact of each feature group

**Figure 7:** Predicted vs. Actual Revenue (Best Model) - Scatter plot with regression line and confidence bands

**Figure 8:** SHAP Summary Plot - Shows feature contributions for individual predictions

**Figure 9:** Statistical Significance Matrix - Heatmap of p-values from paired t-tests between all model pairs

**Figure S6:** Ablation Study Performance Comparison - Bar chart showing R² differences from baseline for Top-K models

---

**End of Paper**