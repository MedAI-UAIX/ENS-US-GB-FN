1. The core codes of an ensemble machine learning algorithm (ENS) for predicting the false-negative risk of ultrasound-guided percutaneous biopsy (US-GB) of liver lesions.

2. In the training group, the best model was selected for subsequent model combination. Five-fold cross-validation and 1000 iterations of recursive feature elimination were used to select the optimal features and parameters for each model. 

3. The voting method was employed to determine the best model combination weights.

Final ENS: ENS=Score_RF×0.2+Score_GBDT×0.5+Score_BAG×0.3+Score_DT×0.1