### Hotel Reviews Classification
This project focuses on classifying hotel reviews from TripAdvisor using machine learning techniques. The dataset used in this project is sourced from Kaggle, specifically the TripAdvisor Hotel Reviews dataset.

### Project Overview
The main goal of this project is to preprocess the text data, perform exploratory data analysis (EDA), and train a classification model using XGBoost. The project involves the following steps:

- Data preprocessing: The raw text data from the hotel reviews is cleaned and processed to remove noise, perform tokenization, lemmatization, and other necessary text preprocessing techniques.

- EDA: The dataset is analyzed to gain insights into the distribution of reviews, explore patterns, and extract relevant features for the classification task.

- Model Training: XGBoost, a popular gradient boosting algorithm, is used to train a classification model on the preprocessed data. The model is trained to predict the sentiment or rating of hotel reviews based on the text content.

- Model Improvement: To enhance the performance of the model, additional techniques such as TF-IDF vectorization and Doc2Vec (document embedding) are applied to the text data.

### Libraries Used
The project utilizes several Python libraries for different tasks, including:

- NLTK (Natural Language Toolkit): Used for text preprocessing, POS tagging, lemmatization, and accessing lexical resources.
- WordCloud: Used to visualize text data, particularly for generating word clouds.
- Scikit-learn: Used for TF-IDF vectorization, converting raw text data into numerical features.
- Keras

Make sure to have the required libraries installed before running the notebook.

To run the notebook and reproduce the project's results, follow these steps:

Download the TripAdvisor Hotel Reviews dataset from Kaggle and place the tripadvisor_reviews.csv file in the data/ directory.

Install the necessary libraries and dependencies by running pip install -r requirements.txt.

Open the Jupyter Notebook hotel_reviews.ipynb in the notebook/ directory and run each cell sequentially to execute the code.

Conclusion
- This project demonstrates the process of preprocessing text data, performing exploratory data analysis, and training a classification model using XGBoost. The trained model can be used to predict the sentiment or rating of hotel reviews based on their text content. 
