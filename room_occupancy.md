

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
    
- Data Pre-processing
    - 'date' column was only used for timestamp only so we assume there is no dependency between timestamp and room occupancy. 
      So I drop the column prior to training the model. We might retain the 'date' column when I have the opportunity to test time series model.
    - 'Relative Humidity' and 'Humidity Ratio' are highly linear dependent so we drop the 'Humidity Ratio' column as well.
 
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


        