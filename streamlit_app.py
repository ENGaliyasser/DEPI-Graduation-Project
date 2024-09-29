import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import datetime
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
# Load the CSV data
grouped = pd.read_csv("NewRFM.csv")
All_Customers_df = pd.read_csv("All_Customers.csv")
All_Customers_Address_df = pd.read_csv("All_Customers_Address.csv")
Transactions_df = pd.read_csv("Transactions.csv")

# Set the Streamlit page configuration
st.set_page_config(page_title="Customer Insights Dashboard", initial_sidebar_state='expanded')

# Add a sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Customer Overview", "Customer Addresses", "Transactions","RFM Analysis","Classification"])

# Home Page
# Load team images
ali_img = Image.open('ali.png')  # Replace with actual file path or variable
youssef_img = Image.open('youssef.jpg')  # Replace with actual file path or variable

# Welcome Page
if page == "Home":
    # Welcome message
    st.title("Welcome to the Customer Insights Dashboard")
    st.write('''This dashboard provides a comprehensive overview of customer demographics, addresses, 
                and transaction data. You can explore each section to gain detailed insights and make informed decisions.''')

    # Meet Our Team section
    st.subheader("Meet Our Team")
    col1, col2 = st.columns(2)
    with col1:
        st.image(ali_img, caption="Ali - Data Scientist")
        st.write("Ali is our lead data analyst who specializes in customer insights and RFM modeling.")
    with col2:
        st.image(youssef_img, caption="Youssef - Data Scientist")
        st.write("Youssef leads our data science team, focusing on machine learning and data-driven strategies.")

    # About the Project: RFM Analysis
    st.subheader("About the Project: RFM Analysis")
    st.write('''RFM (Recency, Frequency, Monetary) Analysis is a customer segmentation technique 
                that categorizes customers based on how recently they purchased (Recency), 
                how often they purchase (Frequency), and how much they spend (Monetary value). 
                This helps businesses understand customer behavior and target their most valuable segments.''')

    # Steps Taken for RFM Analysis
    st.subheader("Steps in Our RFM Analysis Process")
    st.markdown('''
    - **Exploring the Data**: Initially, we loaded and explored the dataset to understand its structure, identify any missing values, and examine data types.
    - **Data Cleaning**: We handled missing or inconsistent data entries, ensuring the dataset was clean and ready for analysis.
    - **Preprocessing**: The data was preprocessed by normalizing or scaling the Recency, Frequency, and Monetary values where needed. We also categorized customers based on predefined rules.
    - **RFM Scoring**: Customers were assigned scores based on Recency, Frequency, and Monetary values. Each customer received a score for each dimension, allowing us to classify them into segments like 'High Value', 'At Risk', etc.
    - **Classification Model**: We implemented a classification machine learning model to further refine customer segmentation. This model helped predict customer behavior and classify customers into predefined categories, enhancing our understanding of customer dynamics.
    - **Segmentation and Insights**: After scoring and classification, we segmented the customers into groups and derived insights from the analysis, identifying key customer segments for targeted marketing strategies.
    - **Visualization and Reporting**: Finally, we visualized the results with tables, charts, and metrics, which are available throughout this dashboard.
    ''')

    # Classification Strategy
    st.subheader("Classification Strategy")
    st.write('''We define the customer segments based on the RFM model to classify customers into different categories 
                for targeted actions. Here's a breakdown of the classification rules:''')
    st.markdown('''
    - **High Value**: R ≤ 4, F ≥ 7, M ≥ 7
    - **Medium Value**: R ≤ 6, F ≥ 4, M ≥ 4
    - **Low Value**: R > 6, F < 4, M < 4
    - **At Risk**: R ≤ 4, F < 4
    - **New Customers**: R ≤ 4, F = 1
    - **Loyal Customers**: F ≥ 7 with any R and M
    ''')



    # Button to the presentation
    st.subheader("Our Presentation")
    st.write('''Click the button below to view the full presentation of this project, where we discuss the RFM analysis and its business implications.''')
    # Create a clickable button that redirects to the presentation link
    # Hyperlink button
    st.markdown(f'''
        <a href="https://www.canva.com/design/DAGR9Jxx5QI/MNojsa70sr1VAqVbvYmowA/edit?utm_content=DAGR9Jxx5QI&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton" target="_blank">
            <button style="background-color:red;color:white;padding:10px 20px;border:none;border-radius:5px;cursor:pointer;">
                View Presentation
            </button>
        </a>
    ''', unsafe_allow_html=True)


# Customer Overview Page
elif page == "Customer Overview":
    st.header("Customer Demographics")
    st.write("### **Customer Metadata**")

    customer_meta = pd.DataFrame({
        'Column': ['customer_id', 'first_name', 'last_name', 'gender', 'DOB', 'past_3_years_bike_related_purchases',
                   'job_title', 'job_industry_category', 'wealth_segment', 'deceased_indicator', 'owns_car', 'tenure'],
        'Datatype': ['Number', 'Text', 'Text', 'Text', 'Date', 'Number', 'Text', 'Text', 'Text', 'Text', 'Text',
                     'Number'],
        'Description': [
            'Unique identifier for customers (Primary Key)',
            'Customer\'s first name',
            'Customer\'s last name',
            'Customer\'s gender',
            'Customer\'s date of birth',
            'Number of bike-related purchases in the last 3 years',
            'Customer\'s job title',
            'The industry category in which the customer works',
            'Classification based on customer\'s wealth (Mass, Affluent, High Net Worth)',
            'Indicates if the customer is deceased (Y/N)',
            'Indicates if the customer owns a car (Yes/No)',
            'The length of time (in years) the customer has been associated with the store.']
    })
    st.dataframe(customer_meta, use_container_width=True)

    # Display sample data
    st.write("### **Sample Customer Data**")
    st.dataframe(All_Customers_df.sample(10), use_container_width=True)
    st.write(f"###### Dataframe Shape: **{All_Customers_df.shape}**")

# Customer Addresses Page
elif page == "Customer Addresses":
    st.header("Customer Addresses")
    st.write("### **Customer Address Metadata**")

    address_meta = pd.DataFrame({
        'Column': ['customer_id', 'address', 'postcode', 'state', 'country', 'property_valuation'],
        'Datatype': ['Number', 'Text', 'Text', 'Text', 'Text', 'Number'],
        'Description': [
            'Unique identifier for customers (Foreign Key)',
            'The full address of the customer',
            'The postal code associated with the address',
            'State of residence',
            'Country of residence (Australia)',
            'Numeric property valuation rating (1-12)']
    })
    st.dataframe(address_meta, use_container_width=True)

    # Display sample data
    st.write("### **Sample Customer Address Data**")
    st.dataframe(All_Customers_Address_df.sample(10), use_container_width=True)
    st.write(f"###### Dataframe Shape: **{All_Customers_Address_df.shape}**")

# Transactions Page
elif page == "Transactions":
    st.header("Transactions Data")
    st.write("### **Transactions Metadata**")

    transactions_meta = pd.DataFrame({
        'Column': ['transaction_id', 'product_id', 'customer_id', 'transaction_date', 'online_order', 'order_status',
                   'brand', 'product_line', 'product_class', 'product_size', 'list_price', 'standard_cost',
                   'product_first_sold_date'],
        'Datatype': ['Number', 'Number', 'Number', 'Date', 'Boolean', 'Text', 'Text', 'Text', 'Text', 'Text',
                     'Currency', 'Currency', 'Date'],
        'Description': [
            'Unique identifier for each transaction (Primary Key)',
            'Product identifier (Foreign Key)',
            'Customer identifier (Foreign Key)',
            'Date of transaction',
            'True/False if the order was online',
            'Order status (Approved/Cancelled)',
            'Product brand',
            'Product line (e.g., Road, Touring)',
            'Product classification (e.g., high, medium, low)',
            'Product size (small, medium, large)',
            'Product list price',
            'Standard cost of the product',
            'Date when the product was first sold']
    })
    st.dataframe(transactions_meta, use_container_width=True)

    # Display sample data
    st.write("### **Sample Transaction Data**")
    st.dataframe(Transactions_df.sample(10), use_container_width=True)
    st.write(f"###### Dataframe Shape: **{Transactions_df.shape}**")
# Classification Page
elif page == "Classification":
  st.header("Classification")
  # Split data into features and target variable
  def classify_customer(row):
      if row['R'] <= 4 and row['F'] >= 7 and row['M'] >= 7:
          return 'High Value'
      elif row['R'] <= 6 and row['F'] >= 4 and row['M'] >= 4:
          return 'Medium Value'
      elif row['R'] > 6 and row['F'] < 4 and row['M'] < 4:
          return 'Low Value'
      elif row['R'] <= 4 and row['F'] < 4:
          return 'At Risk'
      elif row['R'] <= 4 and row['F'] == 1:
          return 'New Customer'
      elif row['F'] >= 7:
          return 'Loyal Customer'
      else:
          return 'Other'


  # Apply classification
  grouped['Customer_Class'] = grouped.apply(classify_customer, axis=1)

  print(grouped['Customer_Class'].value_counts())

  grouped.to_csv('NewRFM (1).csv', index=False)
  X = grouped[['R', 'F', 'M']]
  y = grouped['Customer_Class']

  # Split data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Train a Random Forest classifier
  model = RandomForestClassifier(random_state=42)
  model.fit(X_train, y_train)

  # Make predictions on the testing set
  y_pred = model.predict(X_test)


  # Evaluate the model  

  report = classification_report(y_test, y_pred)
  # Expected Outcomes
  st.write("### Expected Outcomes")
    
  st.markdown('''
    - **Customer Segments**: Based on the RFM scores, customers will be categorized into groups such as:
      - **Loyal Customers**: High frequency and monetary scores.
      - **At-Risk Customers**: Low recency, low frequency, but high monetary value in the past.
      - **Potential Loyalists**: Recent but infrequent buyers with high monetary value.
    
    - **Business Impact**: The segmentation allows for:
      - Targeted marketing campaigns for different customer groups.
      - Retention strategies for at-risk customers.
      - Reward programs for loyal customers.
    ''')

  # Visualize confusion matrix
  cm = confusion_matrix(y_test, y_pred)
  st.write("### Confusion Matrix")

  fig, ax = plt.subplots()
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
  st.pyplot(fig)


  # Visualize feature importances
  importances = model.feature_importances_
  feature_names = X.columns
  st.write("### Feature Importances")
  figa = plt.figure()  # Create a figure object
  plt.barh(feature_names, importances, color='skyblue')
  st.pyplot(figa)  # Display the figure using st.pyplot




# ... (rest of your code)

# RFM Analysis Page
elif page == "RFM Analysis":
    # Load data
    df = pd.read_excel('NewRFM (1).xlsx', sheet_name='Sheet1')

    # Streamlit Tab Setup
    st.title('Customer Analysis')  # Main title

    # RFM Analysis Page
    st.header("RFM Analysis")

    # 1. RFM Summary Statistics
    st.write("### RFM Summary Statistics")
    rfm_summary = df[['Recency', 'Frequency', 'Monetary']].describe()
    st.dataframe(rfm_summary, use_container_width=True)

    # 2. RFM Distribution Visuals
    st.write("### RFM Distributions")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Recency Distribution
    sns.histplot(df['Recency'], ax=axes[0], bins=30, color='skyblue')
    axes[0].set_title('Recency Distribution')

    # Frequency Distribution
    sns.histplot(df['Frequency'], ax=axes[1], bins=30, color='orange')
    axes[1].set_title('Frequency Distribution')

    # Monetary Distribution
    sns.histplot(df['Monetary'], ax=axes[2], bins=30, color='green')
    axes[2].set_title('Monetary Value Distribution')

    st.pyplot(fig)

    # 3. Customer Class Distribution
    st.write("### Customer Class Distribution")
    class_distribution = df['Customer_Class'].value_counts()
    fig, ax = plt.subplots()
    class_distribution.plot(kind='bar', ax=ax, color='coral')
    ax.set_title("Customer Class Distribution")
    st.pyplot(fig)

    # 4. RFM Heatmap
    st.write("### RFM Score Heatmap")
    rfm_scores = df[['R', 'F', 'M']]
    fig, ax = plt.subplots()
    sns.heatmap(rfm_scores.corr(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('RFM Score Correlation Heatmap')
    st.pyplot(fig)

    # 5. Top 10 Customers by Monetary Value
    st.write("### Top 10 Customers by Monetary Value")
    top_customers = df[['customer_id', 'Monetary']].sort_values(by='Monetary', ascending=False).head(10)
    st.dataframe(top_customers, use_container_width=True)

    # 6. RFM Segments
    st.write("### RFM Segmentation")
    df['RFM_Segment'] = df['R'].astype(str) + df['F'].astype(str) + df['M'].astype(str)
    segment_distribution = df['RFM_Segment'].value_counts().head(10)
    fig, ax = plt.subplots()
    segment_distribution.plot(kind='bar', ax=ax, color='purple')
    ax.set_title("Top 10 RFM Segments")
    st.pyplot(fig)
# Footer
st.sidebar.write("---")
st.sidebar.write("Developed with ❤️ by Ali Yasser & Yousef Saeed")
