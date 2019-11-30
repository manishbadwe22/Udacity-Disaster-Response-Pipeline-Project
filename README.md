# Welcome to the Disaster Response Pipeline Project :+1:

## Table of Contents
- Libraries
- Project Description
- File Description
- Analysis
- Results
- Future Improvements
- Licensing, Authors, and Acknowledgements
- Instructions

### Libraries
- json
- pandas
- numpy
- sqlalchemy
- matplotlib
- plotly
- NLTK
- sklearn
- joblib
- flask
- re
- pickle
- string

### Project Summary
Figure Eight organization has provided a Data Set: [Disaster Response Messages](https://www.figure-eight.com/dataset/combined-disaster-response-data/) which stores a large amount of messages received at the time of various disasters on various digital/ social media. There are mapped to 36 different categories indicative of relevant areas. The sorting of these messages to various categories such as Water, Fire, Earth Quake,etc. help the disaster response team to expedite their aid efforts.

The key objective of this project is to build a web-app which will analyze the incoming messages and classify them to specific categories which in turn would be mapped with particular disaster recovery area such as water, fire, medical departments helping their personnels to dedicate their focus and time on actual helping than analysing the messages! 

### File Explaination
1. data
- disaster_categories.csv: dataset of the categories
- disaster_messages.csv: dataset of all the messages
- process_data.py: Extract, Transform, and Load pipeline scripts to read, clean, and save data into a SQLite database
- DisasterResponse.db: Output of the ETL pipeline, a SQLite database containing cleaned messages and categories data
2. models
- train_classifier.py: machine learning pipeline scripts to train a multi-output classification model, test it for accuracy and export the model to a pickle file
- classifier.pkl: output of the machine learning pipeline-  a trained classifier
3. app
- run.py: Flask file to run the web application and to build data visualizations
- templates contains html file for the web application

### Licensing, Authors, and Acknowledgements
Thank you FigureEight for providing the data set for this project.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to https://view6914b2f4-3001.udacity-student-workspaces.com/


