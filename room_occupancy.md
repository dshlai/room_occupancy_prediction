

- Data Source
    - UCI Machine Learning Repository: Occupancy Detection Data set
    - http://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+

- Data attribute:
    - date: time year-month-day hour:minute:second
    - Temperature: in Celsius
    - Relative Humidity: % (1~100%)
    - Light: in Lux
    - CO2: in ppm
    - Humidity Ratio: Derived quantity from temperature and relative humidity, in kgwatervapor/kg-air
    - Occupancy: 0 or 1, 0 for not occupied, 1 for occupied status
- EDA
    - Correlation Between Independent Variables
        - Humidity and Humidity Ratio are highly related to each other
    - Parallel Coordinates
        - **Light** has most visually separability
    - Feature Importance
        - **Light** has highest importance
    - Feature correlate to dependent variable (Y)
        - **Light** has most correlation
    - Class Imbalance
        - Classes are imbalanced because the monitored room is not occupied very often
        - 0 (Not Occupied) is four times more likely to occur than 1 (Occupied)
        - Classifier may have more accuracy predicting room has 0 (Not Occupied) than 1 (Occupied)
           
- Data Pre-processing
    - 'date' column was only used for timestamp only so we assume there is no dependency between timestamp and room occupancy. 
      So I drop the column prior to training the model. We might retain the 'date' column when I have the opportunity to test time series model.
    - 'Relative Humidity' and 'Humidity Ratio' are highly linear dependent so we drop the 'Humidity Ratio' column as well.

- Data Imbalance
    - From class balance EDA it is clear there are some data imbalance between the two classes.
    - I test several models with built-in sample balance and also re-sample the dataset with SMOTEENN algorithm.
    - Best performing model is Balanced Bagging Classifier, perform close to in normal accuracy score and slight better than Logistic Regression in balanced accuracy score.
 
- Preliminary Model Evaluations:
    - Logistic Regression:
        - Validation Accuracy: 0.9924
        - Test Accuracy: 0.9782
        - ROC-AUC Score: 0.9920
        - Mean CV Scores: 0.9904
    - Random Forest (No. of Tree = 1250):
        - Validation Accuracy: 0.9772
        - Test Accuracy: 0.9636
        - ROC-AUC Score: 0.9860
        - Mean CV Scores: 0.9641
    - Gradient Boost Tree:
        - Validation Accuracy: 0.9881
        - Test Accuracy: 0.9718
        - ROC-AUC Score: 0.9821
        - Mean CV Scores: 0.9845


        