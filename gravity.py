import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.linear_model import LogisticRegression

# Title
st.title("Customer Purchase Prediction and Analysis")

# Database Connection
st.sidebar.subheader("Database Connection")
host = st.sidebar.text_input("Host", "127.0.0.1")
user = st.sidebar.text_input("User", "root")
password = st.sidebar.text_input("Password", "Kiran3594", type="password")
database = st.sidebar.text_input("Database", "gravity_books")

if st.sidebar.button("Connect to Database"):
    try:
        # Create database connection
        connection_string = f"mysql+pymysql://{"root"}:{"Kiran3594"}@{"127.0.0.1"}/{"gravity_books"}"
        engine = create_engine(connection_string)

        # Query data
        query = "SELECT * FROM unique_books"
        df = pd.read_sql(query, engine)

        st.success("Connection successful!")
        st.subheader("Raw Data")
        st.dataframe(df.head())

        # Data Preprocessing
        st.subheader("Data Preprocessing")

        # Fill missing values
        df['book_price'] = df['book_price'].fillna(df['book_price'].mean())
        df['book_title'] = df['book_title'].fillna('Unknown Title')
        df['author_name'] = df['author_name'].fillna(method='ffill')
        df['publisher_name'] = df['publisher_name'].fillna(method='bfill')
        st.write("Missing values handled.")

        # Convert order_date to datetime and extract year and month
        df['order_date'] = pd.to_datetime(df['order_date'])
        df['year'] = df['order_date'].dt.year
        df['month'] = df['order_date'].dt.month

        # Demand data
        demand_data = df.groupby(['year', 'month', 'book_id']).agg({
            'book_price': 'sum',
            'order_id': 'count'
        }).reset_index()
        demand_data.rename(columns={'order_id': 'total_orders'}, inplace=True)

        st.subheader("Demand Data")
        st.dataframe(demand_data.head())

        # Visualize sales trends
        st.subheader("Sales Trend Over Time")
        df['year_month'] = df['order_date'].dt.to_period('M')
        sales_trend = df.groupby('year_month')['book_price'].sum().reset_index()
        sales_trend['year_month'] = sales_trend['year_month'].astype(str)

        plt.figure(figsize=(12, 6))
        sns.lineplot(data=sales_trend, x='year_month', y='book_price')
        plt.title("Sales Trend Over Time")
        plt.xticks(rotation=45)
        st.pyplot(plt)

        # Feature Engineering
        st.subheader("Feature Engineering")
        customer_features = df.groupby('customer_id').agg(
            total_orders=('order_id', 'count'),
            avg_order_value=('book_price', 'mean'),
            recency=('order_date', lambda x: (pd.Timestamp('today') - x.max()).days)
        ).reset_index()
        st.dataframe(customer_features.head())

        # Label Encoding
        encoder = LabelEncoder()
        df['book_id_encoded'] = encoder.fit_transform(df['book_id'])

        # ANN Model
        st.subheader("ANN Model Development")

        # Splitting data
        X = customer_features[['total_orders', 'avg_order_value', 'recency']]
        y = (customer_features['total_orders'] < 5).astype(int)  # Binary classification

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Build ANN
        model = Sequential([
            Dense(64, activation='relu', input_dim=X_train.shape[1]),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train model
        st.write("Training the model...")
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

        # Evaluate model
        st.subheader("Model Evaluation")
        y_pred = (model.predict(X_test) > 0.5).astype(int)
        report = classification_report(y_test, y_pred, output_dict=True)
        st.write(pd.DataFrame(report).transpose())

    except Exception as e:
        st.error(f"Error: {e}")



## Deployment with Streamlit
# Sample Data for Demonstration
data = {
    'total_orders': [10, 20, 15, 25, 5],
    'avg_order_value': [50.0, 60.0, 55.0, 70.0, 45.0],
    'recency': [30, 20, 40, 10, 50],
    'churn': [0, 0, 1, 0, 1]  # Target column
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Features and Target Variable
X = df[['total_orders', 'avg_order_value', 'recency']]  
y = df['churn']  

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and Fit the Scaler
scaler = StandardScaler()
scaler.fit(X_train)  # Fit the scaler on the training data

# Scale Training and Testing Data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Streamlit App for Prediction
st.title("Customer Churn Prediction")

# User Inputs
total_orders = st.number_input("Total Orders", min_value=0, max_value=100, value=10)
avg_order_value = st.number_input("Average Order Value", min_value=0.0, value=50.0)
recency = st.number_input("Recency (days since last order)", min_value=0, max_value=365, value=30)

if st.button("Predict Churn"):
    try:
        # Prepare input data
        input_data = np.array([[total_orders, avg_order_value, recency]])
        
        # Scale the input data
        input_data_scaled = scaler.transform(input_data)
        
        # Predict Churn
        prediction = model.predict(input_data_scaled)
        
        # Display Result
        st.write("Churn Probability:", prediction[0])
    except Exception as e:
        st.write("Error:", str(e))
