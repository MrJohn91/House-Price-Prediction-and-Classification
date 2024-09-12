# House-Price-Prediction-and-Classification

# Project Reflection

## Summary
In this project, I worked on both classification and regression tasks, with a shared goal of predicting house prices. For the regression task, the aim was to predict the actual sale prices of houses, while the classification task involved grouping houses based on their characteristics(Expensive or Unexpensive). The project allowed me to explore different machine learning models and evaluate their effectiveness in both predicting continuous values and classifying data into categories.

## Languages and Libraries Used
- **Languages:** Python
- **Libraries:**
  - Pandas: For data handling and cleaning.
  - NumPy: For numerical operations.
  - Scikit-learn: For building and evaluating different machine learning models.
  - XGBoost& Gradients Boost Regressor: For boosting models in regression.
  - Matplotlib/Seaborn/Plotly: For creating visualizations to better understand the data.
  - GridSearchCV : for hyperparameter tuning.

## Key Learnings
- **Evaluation Metrics:** 
  - For the regression task, I used metrics like R-squared, Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE) to measure how well the models predicted house prices.
  - In the classification task, I relied on metrics like accuracy, precision, and recall to assess how accurately the model categorized houses based on their features.
  - Each metric gave me a different perspective on model performance. For example, R-squared shows how much variation in house prices the model can explain, while MAPE helped measure the prediction error as a percentage.
  
- **Model Exploration:** 
  - I experimented with different machine learning models to see which one performed best for both the classification and regression tasks. Trying various models like Random Forest, Gradient Boosting, and Dtree helped me understand the strengths of each. Some models worked better for classification, while others were more suited to regression.
  
- **Feature Selection:** 
  - I focused on feature selection to improve model performance by removing redundant features (like highly correlated ones). For example, in the regression task, I dropped features that provided overlapping information, which helped simplify the model without losing accuracy.

## Challenges Overcame
- **Regression Task Challenges:**
  - Handling the missing data in the dataset was a key challenge. I used imputation techniques to fill in the gaps, ensuring that the model could still make accurate predictions without bias.
  - Another challenge was finding the right model and tuning its parameters to get the best results. By using cross-validation and RandomizedSearchCV, I was able to optimize models like Gradient Boosting and XGBoost.

- **Classification Task Challenges:**
  - In the classification task, balancing the categories and ensuring that the model could generalize well was a challenge. Evaluating the performance with metrics like precision and recall helped me identify which models were handling the task best.
  - I also had to make sure the classification model was not overfitting by selecting the right set of features and balancing the data.

## Additional Reflections
The most satisfying part of the project was seeing how different models performed on the same data in both classification and regression tasks. By trying different approaches, I gained a deeper understanding of how ML models work, how they can be improved, and how important it is to evaluate them using the right metrics. It was a rewarding process that gave me practical experience with real world data and predictive modeling.
