import pandas as pd
import os
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import sqlite3

# Set page configuration (dark theme)
st.set_page_config(
    page_title="File Sensitivity Predictor",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Styling
def apply_dark_theme():
    st.markdown("""
    <style>
        /* Dark Theme Styles */
        .stApp {
            background-color: #121212;
            color: #ffffff;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #BB86FC;
        }
        div.stTextInput > div > input {
            background-color: #1e1e1e;
            color: #ffffff;
            border: 2px solid #BB86FC;
        }
        div.stButton > button {
            background-color: #BB86FC;
            color: #121212;
            border-radius: 10px;
            padding: 10px 20px;
            font-size: 16px;
            width: 100%;
            transition: transform 0.3s ease;
        }
        div.stButton > button:hover {
            transform: scale(1.05);
        }
        .stMarkdown p {
            color: #ffffff;
        }
        .css-1d391kg {
            background-color: #1e1e1e;
            color: #ffffff;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stSidebar {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        .st-bw {
            background-color: #1e1e1e;
            color: #ffffff;
        }
    </style>
    """, unsafe_allow_html=True)

# Initialize SQLite Database
def initialize_database():
    conn = sqlite3.connect("file_predictions.db")
    cursor = conn.cursor()

    # Create a table to store file details, predictions, and confidence scores
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS file_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT,
            file_path TEXT,
            prediction TEXT,
            confidence_score REAL
        )
    """)
    conn.commit()
    conn.close()

# Function to save file details to the database
def save_to_database(file_name, file_path, prediction, confidence_score):
    conn = sqlite3.connect("file_predictions.db")
    cursor = conn.cursor()

    # Insert the file details into the database
    cursor.execute("""
        INSERT INTO file_records (file_name, file_path, prediction, confidence_score)
        VALUES (?, ?, ?, ?)
    """, (file_name, file_path, prediction, confidence_score))

    conn.commit()
    conn.close()

# Function to fetch file records from the database
def fetch_from_database():
    conn = sqlite3.connect("file_predictions.db")
    cursor = conn.cursor()

    # Fetch all records from the database
    cursor.execute("SELECT * FROM file_records")
    records = cursor.fetchall()

    conn.close()
    return records

# Function to delete a record from the database
def delete_from_database(record_id):
    conn = sqlite3.connect("file_predictions.db")
    cursor = conn.cursor()

    # Delete the record with the given ID
    cursor.execute("DELETE FROM file_records WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()

# Load the dataset
df = pd.read_csv("new_metadata.csv")

# Feature Engineering
def extract_features(df):
    sensitive_files = df[df['File Name'].str.lower().str.contains("bank", na=False)]
    print(sensitive_files[['File Name', 'Predicted Sensitivity']])
    
    # Extract file extension from File Type (already in uppercase)
    df['File Type'] = df['File Type'].apply(lambda x: x.lower())

    # Check for sensitive keywords in the file path
    sensitive_keywords = [
        "payroll", "tax", "insurance", "contract", "agreement", 
        "aadhar", "aadharcard", "aadhar_card", "aadhar-card", 
        "pan", "pancard", "pan_card", "pan-card", "bank"
    ]
    df['Path Contains Sensitive Keyword'] = df['File Path'].apply(
        lambda path: 1 if any(keyword in path.lower() for keyword in sensitive_keywords) else 0
    )
    df['Name Contains Sensitive Keyword'] = df['File Name'].apply(
        lambda name: 1 if any(keyword in name.lower() for keyword in sensitive_keywords) else 0
    )

    # Encode the target variable (Predicted Sensitivity)
    le = LabelEncoder()
    df['Predicted Sensitivity'] = le.fit_transform(df['Predicted Sensitivity'])

    return df

# Apply feature extraction to the dataset
df = extract_features(df)

# Select Features (X) and Target (y)
X = df[['File Type', 'File Size (Bytes)', 'Path Contains Sensitive Keyword', 'Name Contains Sensitive Keyword']]
y = df['Predicted Sensitivity']

# One-hot encode 'File Type'
X = pd.get_dummies(X, columns=['File Type'], drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Example function to predict sensitivity of a given file path
def predict_sensitivity(file_path):
    # Remove any leading/trailing whitespace or quotes from the file path
    file_path = file_path.strip().strip('"').strip("'")
    
    # Handle backslashes in Windows paths by converting them to forward slashes
    file_path = file_path.replace("\\", "/")
    
    # Extract file name and extension
    file_name = file_path.split('/')[-1]
    file_extension = file_path.split('.')[-1].lower()
    
    # Assuming that we have a mapping of file extensions to one-hot encoded values
    file_types = ['pdf', 'xlsx', 'docx', 'jpg', 'txt']  # List of file types from training data
    file_type_encoded = {f"File Type_{ft}": 0 for ft in file_types}  # Initialize all to 0
    
    if file_extension in file_types:
        file_type_encoded[f"File Type_{file_extension}"] = 1  # Set the corresponding file type to 1
    
    # Define sensitive keywords (including variations)
    sensitive_keywords = [
        "payroll", "tax", "insurance", "contract", "agreement", 
        "aadhar", "aadharcard", "aadhar_card", "aadhar-card", 
        "pan", "pancard", "pan_card", "pan-card", "bank"
    ]

    # Check if any sensitive keyword is present in the file path or name
    path_sensitive_keyword = 1 if any(keyword in file_path.lower() for keyword in sensitive_keywords) else 0
    name_sensitive_keyword = 1 if any(keyword in file_name.lower() for keyword in sensitive_keywords) else 0

    # Dynamically calculate the file size (in bytes)
    try:
        file_size = os.path.getsize(file_path)  # Get the actual file size
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return "Non-Sensitive", 0  # Default to non-sensitive if file is not found
    
    # Prepare the input data for prediction (file type, file size, and sensitive keyword check)
    input_data = pd.DataFrame({
        'File Size (Bytes)': [file_size],
        'Path Contains Sensitive Keyword': [path_sensitive_keyword],
        'Name Contains Sensitive Keyword': [name_sensitive_keyword],

        **file_type_encoded  # Add the one-hot encoded file type columns
    })
    
    # Ensure the input data has the same columns as the training data
    # Reorder and add missing columns with default value 0
    input_data = input_data.reindex(columns=X_train.columns, fill_value=0)
    
    # Make prediction
    sensitivity_prediction = model.predict(input_data)[0]
    confidence_score = model.predict_proba(input_data).max() * 100  # Confidence score as percentage
    result = "Sensitive" if sensitivity_prediction == 1 else "Non-Sensitive"
    return result, confidence_score

# Initialize the database
initialize_database()

# Apply dark theme
apply_dark_theme()

# Streamlit App
st.title("File Sensitivity Predictor")

# Sidebar for Additional Options
with st.sidebar:
    st.header("About")
    st.info("""
        This app predicts whether a file is sensitive based on its path, name, and other features.
        Enter the file path below and click 'Predict'.
    """)

# Main Content
if 'view_database' not in st.session_state:
    st.session_state.view_database = False

if not st.session_state.view_database:
    st.subheader("Enter File Details")
    file_path = st.text_input("File Path:", placeholder="e.g., C:/Users/ASUS/Desktop/file.docx")

    if st.button("Predict", key="predict_button"):
        if file_path.strip() == "":
            st.warning("Please enter a valid file path.")
        else:
            with st.spinner("Analyzing file..."):
                result, confidence_score = predict_sensitivity(file_path)
            
            # Extract file name from the path
            file_name = file_path.split('/')[-1]

            # Save the prediction to the database
            save_to_database(file_name, file_path, result, confidence_score)

            # Display the prediction and confidence score
            if result == "Sensitive":
                st.error(f"The file is predicted to be: **{result}**")
            else:
                st.success(f"The file is predicted to be: **{result}**")
            st.write(f"Confidence Score: **{confidence_score:.2f}%**")

    if st.button("View Database", key="view_database_button"):
        st.session_state.view_database = True

else:
    st.subheader("File Prediction History")
    records = fetch_from_database()

    if records:
        # Convert records to a DataFrame for better visualization
        records_df = pd.DataFrame(records, columns=["ID", "File Name", "File Path", "Prediction", "Confidence Score"])
        
        # Display records with a delete button for each row
        for index, row in records_df.iterrows():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"**ID:** {row['ID']} | **File Name:** {row['File Name']} | **Prediction:** {row['Prediction']} | **Confidence Score:** {row['Confidence Score']:.2f}%")
            with col2:
                if st.button("Delete", key=f"delete_{row['ID']}"):
                    delete_from_database(row['ID'])
                    st.rerun()  # Refresh the page to reflect changes
    else:
        st.info("No predictions have been made yet.")

    if st.button("Go Back", key="go_back_button"):
        st.session_state.view_database = False