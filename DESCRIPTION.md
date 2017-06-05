I have coded a Python project with 3 scripts to generate the final results. These scripts use 5 modules I built with the help of 3 libraries:

Scripts:
analyze.py - loads a dataset and creates analysis reports as .report files
experiment.py - loads, handles and encodes the training set. Also loads a ML model, runs the experiment and save the model to a pickle file
predict.py - loads, handles and encodes the test set. Then, loads a ML model from a previosly generated file, make the predictions and save them to the CSV

Modules:
IOModule.py - contains the IOProcessor class, which has functions to read and write to files
AnalyzerModule.py - contains the Analyzer class, which creates reports about some dataset
HandlerModule.py - contains the Handler class, which handles missing data and cleans unninformative values
EncoderModule.py - contains the Encoder class, which appropriately encodes values for a ML model
ExperimentationModule.py - contains the Experimentation class, which does cross-validation training on the dataset with some ML model

Libraries:
pandas - to help read, write to CSV and impute or clean values
numpy - to recognize the data types read by pandas and do statistical calculations
scikit-learn - to create the ML model and do cross-validation evaluation

First, I ran the analysis script on the training and testing sets to generate reports so I could better understand the data. These reports showed the distribution of values per column in the datasets, and the original data type of those values. This way, I could decide which columns were informative, which needed imputing of missing values, and how this imputation was going to be made. It also allowed me to check which columns needed encoding.

Then, using a 5-fold cross-validation, I ran the experimentation script with Logistic Regression and Stochasting Gradient Descent Regression (also changing their parameters) on the training set to check which model had the best results based on the F-Measure score per fold, the average of them and the standard deviation.

Finally, the probabilities were generated using the model with highest F-Measure [SEE INSTRUCTIONS].