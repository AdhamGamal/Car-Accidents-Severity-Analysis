# Leeds 2011: Car Accidents Severity Analysis and Classification
## By: Adham Gamal

## Project Overview:

The primary goal of this project is to analyze and classify car accidents' severity using the Leeds 2011 dataset, which consists of two main phases:

### Phase 1: Data Analysis

In this project, we conducted a comprehensive analysis of the Car Accidents Severity dataset, guiding our exploration through the following key steps:

#### 1.1 Data Exploration and Cleansing:

- Loaded the dataset, ensuring a solid foundation for subsequent analysis.
- Handled missing data, removed duplicates, and resolved misformed values where necessary.
- Conducted features engineering, generating new meaningful attributes and reducing high cardinality in specific features.

#### 1.2 Features Distribution Analysis:

- Investigated the distribution of each feature through bar plots, unveiling imbalanced distributions across multiple features.
- Explored the reasons behind each distribution and gained insights into the dataset's nature.

#### 1.3 Further Data Preprocessing:

- Grouped minority classes and dropped unnecessary features.
- Employed label encoding for numerical representation of categorical features.
- Explored feature correlations, identifying weak linear relationships.

### Phase 2: Classification and Modeling

In this crucial phase, we transition into the heart of the project, where we delve into the complexities of classification and modeling:

#### 2.1 Feature Selection:

- Utilized different approaches to feature selection, identifying the top 5 common features for modeling.
- Validated the assumption of dropping features with high bias.

#### 2.2 Model Selection:

- Considering the categorical nature of features and weak linear relationships, opted for tree-based, complex, or naive Bayes models.

#### 2.3 Sampling Techniques:

- Addressed class imbalance using oversampling techniques, enhancing model generalization and mitigating bias towards the majority class.

#### 2.4 Methodology:

- Developed a cohesive pipeline encompassing data analysis, feature engineering, selection, and model training.
- Employed grid search to optimize hyperparameters for each selected model.
- Evaluated model performance through cross-validation and assessed the best estimator's performance on the test data using various metrics.
- Plotted metrics across all models for effective comparison.
- Introduced two paths in the pipeline: one without oversampling and the other incorporating oversampling, facilitating a comprehensive performance evaluation.

#### 2.5 Model Comparison:

- Highlighted the impact of oversampling in addressing generalization issues caused by imbalanced data, effectively mitigating bias towards the majority class.


## Dataset

The data attached is about Car Accidents across Leeds City - UK for 2011. The outcome is to predict the variable which is highlighted in orange below:

- **Features Description**
    - **Reference Number:** A unique identifier for each accident.
    - **Easting and Northing:** Geographic coordinates of the accident location.
    - **Number of Vehicles:** The count of vehicles involved in the accident.
    - **Accident Date:** The date when the accident occurred.
    - **Time (24hr):** The time of day when the accident occurred in 24-hour format.
    - **1st Road Class:** Classification of the first road involved in the accident.
    - **Road Surface:** The condition of the road surface at the time of the accident (e.g., Wet/Damp, Dry).
    - **Lighting Conditions:** Illumination conditions at the accident site (e.g., Daylight, Darkness).
    - **Weather Conditions:** Weather conditions at the time of the accident.
    - **Casualty Class:** Classification of individuals involved in the accident (e.g., Driver, Pedestrian, Passenger).
    - **Sex of Casualty:** Gender of the individuals involved.
    - **Age of Casualty:** Age of the individuals involved.
    - **Type of Vehicle:** The type of vehicle involved in the accident.

<br>

- **Output Required**
    - **Casualty Severity:** The severity of the casualties (Slight, Serious, Fatal).

<br>

## Project File Structure:

The project file structure is organized as follows:

``` 
project-root/
│
├── data/
│ ├── Road Accidents with Address.xlsx
│ └── Road Accidents.xlsx
│
├── utils/
│ ├── data_utils.py
│ ├── pipeline_utils.py
│ └── plot_utils.py
|
├── Analysis.ipynb
└── Pipeline.ipynb
```


## Conclusion:

In conclusion, The analysis phase involved meticulous data exploration, cleansing, and preprocessing to prepare the data for modeling. Key features were identified and engineered to enhance the understanding of the dataset's characteristics.

Moving into the classification and modeling phase, we strategically selected models based on the categorical nature of features and weak linear relationships observed. To address class imbalance, oversampling techniques were employed, and a comprehensive methodology was established to guide the entire pipeline. Model performance was evaluated rigorously, comparing results with and without oversampling.

The project's file structure ensures organization and ease of navigation, facilitating collaboration and future enhancements. Overall, this project provides valuable insights into car accidents' severity and lays the foundation for further research and application in real-world scenarios.