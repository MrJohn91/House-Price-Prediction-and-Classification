# House-Price-Prediction-and-Classification

## Project Reflection

### Summary
In this project, I worked on both classification and regression tasks, with the goal of predicting house prices. For the regression task, the aim was to predict the actual sale prices of houses, while the classification task involved grouping houses into categories (Expensive or Unexpensive). This project allowed me to explore different machine learning models and evaluate their effectiveness in predicting both continuous values and categorizing data into meaningful groups.

### Languages and Libraries Used
- **Languages:** Python
- **Libraries:**
  - **Pandas:** For data handling and cleaning.
  - **NumPy:** For numerical operations.
  - **Scikit-learn:** For building and evaluating different machine learning models.
  - **XGBoost & Gradient Boosting Regressor:** For boosting models in regression.
  - **Matplotlib/Seaborn/Plotly:** For creating visualizations to better understand the data.
  - **GridSearchCV:** For hyperparameter tuning.

### Key Learnings
- **Evaluation Metrics:** 
  - For the regression task, I used metrics like R-squared, Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE) to measure how well the models predicted house prices.
  - In the classification task, I relied on metrics like accuracy, precision, and recall to assess how accurately the model categorized houses based on their features.
  - Each metric provided a different perspective on model performance. For example, R-squared showed how much variation in house prices the model could explain, while MAPE helped measure the prediction error as a percentage.
  
- **Model Exploration:** 
  - I experimented with different machine learning models to see which one performed best for both the classification and regression tasks. Trying various models like Random Forest, Gradient Boosting, and Decision Trees helped me understand the strengths of each. Some models worked better for classification, while others were more suited to regression.
  
- **Feature Selection:** 
  - I focused on feature selection to improve model performance by removing redundant features (like highly correlated ones). For example, in the regression task, I dropped features that provided overlapping information, which helped simplify the model without losing accuracy.

### Challenges Overcame
- **Data Handling:** 
  - Handling the missing data in the dataset was a key challenge. I used **Simple Imputer** to fill in the gaps with appropriate values, such as using the best strategy such as median or mean for numerical data and placeholder for categorical data. This ensured that the model could still make accurate predictions without bias.
  
- **One-Hot and Ordinal Encoding:**
  - Another challenge was dealing with categorical data. For unordered categories, I used **One-Hot Encoding** to convert them into separate columns so the machine can understand. For ordered categories, I applied **Ordinal Encoding** to retain the rankings. This required careful preprocessing to ensure the model interpreted the data correctly.

- **Pipeline Integration:**
  - Creating an efficient **Pipeline** for numerical and categorical features was crucial. The pipeline allowed the data preprocessing steps, such as scaling and encoding, to be seamlessly integrated with the model training. This saved time and ensured consistent processing during model evaluation.

- **Model Tuning:**
  - Using **GridSearchCV** to tune hyperparameters across different models was essential for getting the best performance. Finding the right balance between accuracy and model complexity was key in improving predictions for both tasks.

### Additional Reflections
The most satisfying part of the project was seeing how different models performed on the same data in both classification and regression tasks. By trying various approaches, I gained a deeper understanding of how machine learning models work, how they can be improved, and how important it is to evaluate them using the right metrics. It was a rewarding process that gave me practical experience with real-world data and predictive modeling.
