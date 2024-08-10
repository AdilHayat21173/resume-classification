# Resume Classification Model 

## Description

This project involves developing a machine learning model designed to classify resumes into specific career roles or fields based on their content. Utilizing Deep Learning techniques, the system employs an LSTM (Long Short-Term Memory) model to predict the most appropriate job role or category from a given resume. The categories include:

- **Data Science**
- **HR**
- **Advocate**
- **Arts**
- **Web Designing**
- **Mechanical Engineer**
- **Sales**
- **Health and Fitness**
- **Civil Engineer**
- **Java Developer**
- **Business Analyst**
- **SAP Developer**
- **Automation Testing**
- **Electrical Engineering**
- **Operations Manager**
- **Python Developer**
- **DevOps Engineer**
- **Network Security Engineer**
- **PMO**
- **Database**
- **Hadoop**
- **ETL Developer**
- **DotNet Developer**
- **Blockchain**
- **Testing**

By analyzing key elements in the resume, such as skills, experiences, and education, the LSTM model leverages its ability to capture sequential dependencies in text to provide accurate predictions.

The model is trained on a diverse dataset of resumes, ensuring it can handle a wide range of job roles and industries. This classification system can be used by job recruiters, career counselors, and job seekers to better match resumes with job requirements or to gain insights into job market trends.

## Key Features

- **Resume Parsing:** Extracts relevant information from resumes.
- **Career Role Classification:** Predicts the appropriate job role based on resume content using an LSTM model.
- **Customizable:** Can be adapted to classify resumes for various industries and job functions.
- **User-Friendly Interface:** Integrates with platforms where resumes are submitted or reviewed.

## Technologies Used

- **Deep Learning (LSTM):** For sequential text analysis and feature extraction.
- **Natural Language Processing (NLP):** For text preprocessing and feature engineering.
- **Machine Learning Algorithms:** For training and classification tasks.
- **Streamlit:** For creating an interactive web application interface.
## Google Colab Notebook

You can view and run the Google Colab notebook for this project [here](https://colab.research.google.com/drive/1UoCic56G46gm1J-Todkxmm_0K9oNVVph).

![image](https://github.com/user-attachments/assets/7d71a974-000a-478d-9223-4bf51cffa81e)

You can access the dataset [dataset](https://www.kaggle.com/datasets/jillanisofttech/updated-resume-dataset).



## Interfaces

### 1. Text Input Interface (`app.py`)

- **Description:** This interface allows users to paste resume text and get a prediction of the job role using the LSTM model.
**Screenshots:**
  - **Actual Data (Java Developer):**
    ![java_Developer](https://github.com/user-attachments/assets/fa2b36a8-a512-4c2f-935a-efa5f2031dfd)
  - **Prediction Result:**
    ![image](https://github.com/user-attachments/assets/f20526bc-c92f-4a47-af4b-f1a06e960155)
  - **Actual Data (python_Developer):**
    ![python_Developer](https://github.com/user-attachments/assets/1c6674af-34aa-4d8d-a01f-ad5863c9e7f1)
  - **Prediction Result:**
    ![image](https://github.com/user-attachments/assets/07bb1b95-f7d8-4217-98ed-ab0db11abe28)
  - **Data for Civil Engineer:**
    ![image](https://github.com/user-attachments/assets/351c0840-fc2c-44a0-ba00-5f8f0c9c1193)

### 2. PDF Upload Interface (`app1.py`)

- **Description:** This interface allows users to upload a PDF resume and get a prediction of the job role using the LSTM model.
- **Screenshot:**
  ![HR](https://github.com/user-attachments/assets/4d63b886-4caf-444b-b40a-87297abc2a2a)

## Deployment

The application for text input and classification can be accessed [here](https://adilhayat21173-resume-classification-app-ah3hnh.streamlit.app/).

The application for PDF upload and classification can be accessed [here](https://adilhayat21173-resume-classification-app1-adfqsd.streamlit.app/).


## Getting Started

1. **Clone the Repository**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Install Dependencies**
    - For `app.py` and `app1.py` :
      ```bash
      pip install -r requirements.txt
      ```
    

3. **Run the Applications**
    - For text input and classification (`app.py`):
      ```bash
      streamlit run app.py
      ```
    - For PDF upload and classification (`app1.py`):
      ```bash
      streamlit run app1.py
      ```

4. **Provide a Resume:** 
    - In `app.py`, paste the resume text to classify.
    - In `app1.py`, upload the PDF resume to classify.

## Acknowledgements

- [TensorFlow](https://www.tensorflow.org/) for deep learning libraries.
- [NLTK](https://www.nltk.org/) for natural language processing.
- [Streamlit](https://streamlit.io/) for creating the interactive web interface.

## Author
[@AdilHayat](https://github.com/AdilHayat21173)

## Feedback
If you have any feedback, please reach out to us at [hayatadil300@gmail.com](mailto:hayatadil300@gmail.com).

