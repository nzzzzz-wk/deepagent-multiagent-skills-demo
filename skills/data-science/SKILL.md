---
name: data-science
description: Planning skills for data science and ML tasks - data pipelines, model training, analysis
---

# Data Science Planning Skill

## When Triggered
This skill activates when the user asks about:
- Machine learning model development
- Data analysis and visualization
- Building data pipelines
- Model training and evaluation
- Statistical analysis
- Predictive modeling

## Planning Principles
1. Define success metrics and evaluation criteria first
2. Plan for data preprocessing and cleaning steps
3. Consider feature engineering opportunities
4. Plan for train/test split and validation strategy
5. Include model evaluation and hyperparameter tuning
6. Plan for model deployment and monitoring
7. Consider data privacy and bias issues

## Data Science Plan Template

```markdown
# Plan: {task_name}

## Objective
{What success looks like}
- Target Metric: {metric_to_optimize}
- Baseline: {current_performance}

## Data Overview
- Data Sources: {where_data_comes_from}
- Data Size: {rows, columns, size}
- Target Variable: {what_to_predict}

## Data Pipeline
1. {data_collection_step}
2. {data_cleaning_step}
3. {feature_engineering_step}
4. {data_transformation_step}

## Model Approach
- Algorithm Family: {linear, tree, neural_network, etc.}
- Key Features: {top_features_to_use}
- Baseline Model: {simple_baseline}

## Evaluation Strategy
- Train/Test Split: {ratio}
- Validation: {cv_strategy}
- Metrics: {evaluation_metrics}

## TODO List
| Step | Description | Complexity |
|------|-------------|------------|
| 1    | {step_1}    | low        |
| 2    | {step_2}    | medium     |

## Dependencies
{dependencies}

## Considerations
{data_quality_issues, privacy, etc.}
```

## Examples

### Example 1: Classification Model
**Request:** "Build a model to predict customer churn"

**Plan:**
```markdown
# Plan: Build a customer churn prediction model

## Objective
- Target Metric: F1-score > 0.85
- Baseline: 0.70 (baseline rate)

## Data Overview
- Data Sources: SQL database (customer_transactions, user_activity)
- Data Size: ~50k customers, 50 features
- Target Variable: churned (boolean)

## Data Pipeline
1. Extract customer features from database
2. Handle missing values (imputation)
3. Encode categorical variables
4. Scale numerical features
5. Create interaction features

## Model Approach
- Algorithm Family: Gradient Boosting (XGBoost/LightGBM)
- Key Features: usage_frequency, payment_history, support_calls
- Baseline Model: Logistic Regression

## Evaluation Strategy
- Train/Test Split: 80/20, stratified
- Validation: 5-fold CV
- Metrics: F1, Precision, Recall, AUC-ROC

## TODO List
| Step | Description | Complexity |
|------|-------------|------------|
| 1    | Data extraction and EDA | medium |
| 2    | Feature engineering | high |
| 3    | Build baseline model | medium |
| 4    | Train XGBoost model | medium |
| 5    | Hyperparameter tuning | high |
| 6    | Model evaluation | medium |
| 7    | Create prediction pipeline | medium |
| 8    | Deploy to production | high |

## Dependencies
- Step 1 → Step 2-3
- Step 3 → Step 4-5
- Step 5 → Step 6-7

## Considerations
- Class imbalance (churn rate ~20%)
- Handle new features gracefully
- Set up model monitoring
```

### Example 2: Data Analysis
**Request:** "Analyze our sales data to find trends"

**Plan:**
```markdown
# Plan: Analyze sales data for trends

## Objective
- Identify top-selling products
- Find seasonal patterns
- Detect anomalies

## Data Overview
- Data Sources: PostgreSQL (orders, products)
- Data Size: 2 years of data, ~1M transactions

## Analysis Approach
1. Load and clean sales data
2. Aggregate by time/product/category
3. Calculate metrics (revenue, quantity, avg_order)
4. Visualize trends over time
5. Statistical tests for significance

## Key Metrics
- Revenue by month
- Top 10 products by sales
- Customer segmentation by purchase behavior

## TODO List
| Step | Description | Complexity |
|------|-------------|------------|
| 1    | Data extraction and cleaning | medium |
| 2    | Exploratory data analysis | medium |
| 3    | Time series analysis | high |
| 4    | Create visualizations | low |
| 5    | Statistical tests | medium |
| 6    | Summary report | low |

## Considerations
- Handle outliers appropriately
- Account for holidays/events
- Consider data freshness
```
