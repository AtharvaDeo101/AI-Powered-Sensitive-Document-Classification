<h2>File Sensitivity Predictor:</h2>

![image](https://github.com/user-attachments/assets/e7ad15d5-936b-4f40-aceb-ae051ee73df2)
![image](https://github.com/user-attachments/assets/ee25e289-11e8-4703-a0aa-6c67af0d3e56)


The File Sensitivity Predictor is a Streamlit-based web application that predicts whether a file is sensitive or non-sensitive based on its path, name, and other features. The app uses a trained machine learning model to classify files and provides a confidence score for each prediction. It also integrates with an SQLite database to store predictions and allows users to view, delete, and manage records.
<hr>
<h2>Features:</h2>

1. File Sensitivity Prediction : Predicts whether a file is sensitive (e.g., contains sensitive keywords like "bank," "payroll," etc.).
2. Confidence Score : Displays the confidence level of the prediction as a percentage.
3. Database Integration : Stores file details, predictions, and confidence scores in an SQLite database.
4. View Database : Allows users to view all stored predictions in a table format.
5. Delete Records : Provides a "Delete" button to remove specific records from the database.
6. Dark Mode UI : A sleek dark theme for a modern and visually appealing interface.
7. Go Back Button : Enables users to return to the main interface after viewing the database.

<hr>

<h2>Installation</h2>
<h3>Prerequisites</h3>
1. Python 3.8 or higher<br>
2. Pip (Python package manager)
<h3>Steps to Set Up:</h3>
1. Clone the Repository:<br>
<i></i>git clone https://github.com/your-username/file-sensitivity-predictor.git</i><br>
cd file-sensitivity-predictor<br>
<br>
2. Install Dependencies :<br>
Install the required Python libraries using pip:
<br>
<br>
3. Prepare the Dataset :<br>
Place your dataset (new_metadata.csv) in the project directory.<br>
Ensure the dataset contains columns such as File Name, File Path, File Type, File Size (Bytes), and Predicted Sensitivity.<br>
<br>
4. Run the App :<br>
Start the Streamlit app by running:<br>
<ul>1. cd..</ul>
<ul>2. Path of pict Folder : (eg : C:\...\GitHub\Binary-Fetch-AISSMS-IOIT-\pict)<br></ul>
<ul>3. streamlit run app.py</ul><br>
5. Access the App :<br>
Open your browser and navigate to the provided local URL (e.g., http://localhost:8501).<br>
<br>
<hr>

<h2>Usage</h2>
1. Enter File Details :
<ul>Input the file path in the text box (e.g., C:/Users/ASUS/Desktop/file.docx).</ul>
<ul>Click the Predict button to analyze the file.</ul>
2. View Results :
<ul>The app will display whether the file is Sensitive or Non-Sensitive along with the confidence score.</ul>
3. View Database :
<ul>Click the View Database button to see all stored predictions.</ul>
<ul>Each record includes the file name, file path, prediction, and confidence score.</ul>
4. Delete Records :
<ul>Use the Delete button next to a record to remove it from the database.</ul>
5. Go Back :
<ul>Click the Go Back button to return to the main interface.</ul>
<hr>

<h2>Acknowledgments</h2>
1. Built using Streamlit , a powerful framework for creating data apps.<br>
2. Machine learning model trained using scikit-learn .<br>
3. Inspired by real-world use cases for file sensitivity classification.<br>
<hr>
