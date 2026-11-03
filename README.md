#  Beauty Product Performance Analysis
### A Six-Week Data Science with Python Project
##  Project Overview
This project focuses on analyzing beauty and cosmetic product performance using Data Science and Machine Learning techniques.
The project investigates how product attributes such as price, brand, category, ingredients, skin type, cruelty-free status, and customer engagement relate to product ratings.
The primary objective is to understand product performance, identify meaningful patterns in customer ratings, develop predictive models, and generate data-driven business insights that can support pricing strategies, product positioning, marketing decisions, and portfolio management in the Beauty & Wellness industry.
This project follows a complete Data Science workflow, beginning with business problem identification and data sourcing, followed by data cleaning, exploratory data analysis, feature engineering, predictive modeling, and model evaluation.

## Project Objectives
- Understand the application of Data Science in the Beauty & Wellness industry.
- Identify business problems related to beauty product performance.
- Explore relationships between product attributes and customer ratings.
- Perform data cleaning and preprocessing.
- Conduct Exploratory Data Analysis (EDA).
- Apply statistical techniques to identify meaningful patterns.
- Engineer relevant features for machine learning.
- Develop regression and classification models.
- Evaluate model performance using appropriate evaluation metrics.
- Translate analytical findings into meaningful business recommendations.

## Dataset Information

Dataset: Most Used Beauty and Cosmetic Products 2024
Source: Kaggle
Dataset Link: [Most Used Beauty and Cosmetic Products 2024](https://www.kaggle.com/datasets/aashitanoah/most-used-beauty-and-cosmetic-products-2024)
Dataset Size: Approximately 15,000 records and 14 columns.
The dataset contains product-level information, including:
- Product Name
- Brand
- Product Category
- Price (USD)
- Product Rating
- Number of Reviews
- Skin Type
- Ingredients
- Cruelty-Free Status
The dataset was used throughout the six-week project to perform data analysis, preprocessing, feature engineering, and machine learning experiments.

### Week 1: Data Science Orientation & Problem Definition

**Objective:** Identify a real-world business problem in the Beauty & Wellness industry.
Activities:
- Explored the application of Data Science in the beauty and cosmetics industry.
- Identified product performance analysis as the business problem.
- Defined project objectives and research questions.
- Formulated hypotheses related to product pricing, ratings, brands, and customer engagement.
- Identified relevant dataset attributes and potential data sources.
- Proposed an analytical methodology.
Key Research Questions:
1. Which product categories have the highest average ratings?
2. Which brands have the highest average ratings?
3. Is product price associated with product ratings?
4. Is the number of reviews associated with product ratings?
5. Do average ratings differ across target skin types?
6. Are certain ingredients associated with higher ratings?
7. Do cruelty-free and non-cruelty-free products differ in price or ratings?
Deliverable: Business Problem Definition and Project Proposal.

### Week 2: Data Sourcing & Analysis Planning

**Objective:** Understand the data requirements and establish a structured analytical plan.
Activities:
- Explored potential data sources for beauty and cosmetic product information.
- Compared dataset suitability based on relevance, availability, and required attributes.
- Examined the selected dataset and its available variables.
- Identified important numerical and categorical features.
- Planned data validation and preprocessing strategies.
- Identified appropriate statistical techniques for investigating product attributes and ratings.
- Prepared a structured analysis plan for subsequent project stages.
Deliverables: Data Sourcing Notebook and Analysis Planning Report.

### Week 3: Data Cleaning & Preprocessing

**Objective:** Prepare the dataset for reliable analysis and machine learning.
Activities:
- Loaded and inspected the dataset using Pandas.
- Examined dataset dimensions, column names, and data types.
- Checked for missing values and duplicate records.
- Implemented missing-value handling techniques.
- Identified and examined invalid or inconsistent values.
- Checked numerical columns for potential outliers.
- Applied appropriate data preprocessing and validation techniques.
- Prepared the dataset for exploratory analysis and subsequent modeling.
Important Observation: The selected dataset contained no missing values and no exact duplicate records during inspection. Therefore, missing-value treatment and duplicate-removal procedures were implemented and evaluated, but no actual missing-value imputation or duplicate deletion was required.
Deliverables: Data Cleaning and Preprocessing Notebook and Report.

### Week 4: Exploratory Data Analysis & Feature Engineering

**Objective:** Discover meaningful patterns and relationships within the beauty product dataset.
Exploratory Data Analysis:
- Univariate analysis of numerical and categorical variables.
- Bivariate analysis of product attributes and ratings.
- Multivariate analysis to understand interactions between features.
- Analysis of product prices, ratings, review counts, brands, and categories.
- Pearson and Spearman correlation analysis.
- Statistical hypothesis testing using t-tests and ANOVA.
- Data visualization using Matplotlib and Seaborn.
Feature Engineering:
Created additional features to improve the dataset for machine learning:
- Price_Category
- Product_Popularity
- Log_Reviews
- Brand_Frequency
- Price_Review_Interaction
Additional preprocessing:
- Price normalization and standardization.
- One-hot encoding of categorical variables.
- Preparation of feature-engineered datasets.
Deliverables:
- Exploratory Data Analysis Notebook.
- Feature-Engineered Dataset.
- Encoded Dataset.
- EDA and Feature Engineering Report.

### Week 5: Predictive Modeling & Algorithm Selection

**Objective:** Develop machine learning regression models to predict beauty product ratings using product attributes.
Problem Type: Supervised Machine Learning – Regression.
Target Variable: Rating
Input Features:
- Price_USD
- Number_of_Reviews
- Product_Type
- Category
- Skin_Type
- Brand
- Cruelty_Free
  
Machine Learning Algorithms Implemented:
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
Model Development Process:
- Selected relevant input features and target variable.
- Applied preprocessing techniques for numerical and categorical features.
- Used median and most-frequent imputation strategies in the modeling pipeline.
- Applied feature scaling and one-hot encoding.
- Split the dataset into training and testing sets using an 80:20 ratio.
- Trained and compared three regression algorithms.
- Performed five-fold cross-validation.
Evaluation Metrics:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)
Visualizations:
- Model performance comparison using RMSE and R².
- Actual vs. predicted ratings.
- Feature importance analysis.
- Training and testing performance comparison.
The regression models were used to investigate how product characteristics could help estimate numerical product ratings.
Deliverable: Predictive Modeling Notebook and Model Comparison Report.

### Week 6: Model Evaluation & Business Impact Communication

**Objective:** Evaluate classification models and translate their performance into meaningful business insights.
Problem Type: Supervised Machine Learning – Classification.
Target Variable: High_Rating
Products were classified into two categories:
- High Rating: Rating ≥ 4.0
- Lower Rating: Rating < 4.0
Classification Algorithms Implemented:
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
Model Evaluation Process:
- Prepared the target variable using rating thresholds.
- Applied preprocessing and categorical encoding.
- Used an 80:20 stratified train-test split.
- Trained and evaluated classification algorithms.
- Compared classification performance.
- Performed five-fold cross-validation using F1-score.
- Examined feature importance to understand influential product attributes.
Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
Visualizations and Evaluation:
- Confusion matrices.
- ROC curves.
- Model performance comparison charts.
- Feature importance visualization.
- Classification reports.
- Cross-validation performance comparison.
Business Impact:
The classification analysis provides a way to identify product characteristics associated with highly rated products. Such insights can help businesses investigate product positioning, understand customer preferences, prioritize product categories, and make more informed marketing and portfolio decisions.
Deliverables:
- Model Evaluation Notebook.
- Model Performance Comparison CSV.
- Cross-Validation Results CSV.
- Feature Importance CSV.
- Final Project Report.

##  Technologies & Libraries Used
```text

 Technology                       -                      Application 
_______________________________________________________________________________________
 Python                           -                 Core programming language 
 NumPy                            -                 Numerical operations 
 Pandas                           -                 Data manipulation and preprocessing 
 Matplotlib                       -                 Data visualization 
 Seaborn                          -                 Statistical visualization 
 Scikit-learn                     -                 Machine learning and model evaluation 
 Jupyter Notebook & VSCode        -                 Interactive analysis and experimentation 
 Git & GitHub                     -                 Version control and project management 
```

## Repository Structure
```text
Beauty-Product-Performance-Analysis/
│
├── README.md
│
├── most_used_beauty_cosmetics_products_extended.csv
│
├── week1.ipynb
│
├── week2_modified.ipynb
├── week2_doc.docx
│
├── week3.ipynb
├── week3_doc.docx
│
├── week4.ipynb
├── week4_doc.docx
│
├── week5.ipynb
│
├── week6_model_evaluation.csv
├── week6_cross_validation.csv
├── week6_feature_importance.csv
│
└── Final_Project_Report.docx
```
Note: The structure highlights the main project files. Update filenames if your uploaded files use different names.*

## Overall Data Science Workflow
```text
Business Problem Definition
          ↓
Data Sourcing & Understanding
          ↓
Data Cleaning & Preprocessing
          ↓
Exploratory Data Analysis
          ↓
Feature Engineering
          ↓
Regression Modeling
          ↓
Classification Modeling
          ↓
Model Evaluation
          ↓
Business Insights & Recommendations
```

## Expected Business Applications
The analytical workflow can help beauty and cosmetics businesses:
- Understand the relationship between product attributes and ratings.
- Identify product categories and brands with stronger customer ratings.
- Investigate the association between pricing and customer satisfaction.
- Understand the role of customer engagement and review counts.
- Identify product characteristics associated with high ratings.
- Support product positioning and marketing strategies.
- Make data-informed decisions about product portfolios.

## Project Limitations
- The dataset represents a limited collection of beauty and cosmetic products.
- Product ratings may not fully represent overall customer satisfaction.
- Observational relationships do not establish causation.
- Some product characteristics may not be available or consistently represented.
- Machine learning predictions depend on the quality and representativeness of the available dataset.
- The high-rating classification threshold of 4.0 is a project-defined criterion.

##  Conclusion
Beauty Product Performance Analysis demonstrates an end-to-end Data Science workflow through a practical business problem in the Beauty & Wellness industry.
Across six weeks, the project progressed from business problem identification and data sourcing to data cleaning, exploratory data analysis, feature engineering, regression modeling, classification, and model evaluation. By applying Python, Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn, the project explored relationships between beauty product characteristics and customer ratings while developing predictive models to support data-driven decision-making. This project provided practical experience in data analysis, statistical techniques, data preprocessing, feature engineering, supervised machine learning, model evaluation, and communicating analytical findings in a business context.

Project Domain: Data Science | Machine Learning | Beauty & Wellness
Programming Language: Python
Project Duration: 6 Weeks

### Author: SHAIK NIHASUHANI
