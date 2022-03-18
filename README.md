# Disaster Response Pipeline Project

### Summary
This is a Disaster Response Pipeline Project created for the Udacity Data Scientist Nanodegree Program. The result is a web application that allows classifying disaster related messages into 36 categories, as well as provides a visual overview of the existing disaster related messages.

### Project Structure
The project is organized into three folders that contain the following scripts:
- process_data.py - loading, cleaning, and processing data containing disaster related messages and their categories;
- train_classifier.py - building, training, evaluating, and saving classification model;
- run.py - creating a web application for classifying input disaster related messages.

### Installation instructions:
1. Clone the repository:
    `git clone https://github.com/osydorchuk89/disaster_response_pipeline`

2. Run the following commands in the project's root directory to set up your database and model:

    - To run ETL pipeline that cleans data and stores in database:
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves:
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run the following command in the app's directory to run your web app:
    `python run.py`

4. Go to http://0.0.0.0:3001/ (alternatively, type http://localhost:3001/ in your browser).

### Packages
Python Version: 3.9.7

To install all necessary packages, run `pip install -r requirements.txt`

### Licensing
The code is released under the [MIT License](https://github.com/osydorchuk89/movie_box_office/blob/main/LICENSE).

## Acknowledgement
This project was created for the Udacity Data Scientist Nanodegree program.
