import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from io import StringIO # Import StringIO for df.info()

# Set up matplotlib for better plot aesthetics
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (10, 6) # Default figure size

# --- Data Loading and Preprocessing ---
@st.cache_data # Cache the data loading and processing for performance
def load_and_preprocess_data():
    df = pd.read_csv('Real estate.csv')
    df.drop(columns='No', inplace=True)

    # Function to convert a fractional year to a specific date
    def fractional_year_to_date(fractional_year):
        year = int(fractional_year)
        # Handle leap years correctly
        is_leap = (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))
        day_of_year = (fractional_year - year) * (366 if is_leap else 365)
        # Handle cases where day_of_year might be slightly negative due to float precision
        if day_of_year < 0:
            day_of_year = 0
        return datetime(year, 1, 1) + timedelta(days=day_of_year)

    df['X1 transaction date'] = df['X1 transaction date'].apply(fractional_year_to_date).dt.strftime('%Y-%m-%d')
    df['X1 transaction date'] = pd.to_datetime(df['X1 transaction date'])

    # Convert 'X1 transaction date' to datetime and extract year and month
    df['transaction_year'] = df['X1 transaction date'].dt.year
    df['transaction_month'] = df['X1 transaction date'].dt.month

    # Drop the original 'X1 transaction date' column
    df = df.drop('X1 transaction date', axis=1)
    return df

# --- Model Training with Hyperparameter Tuning ---
@st.cache_resource # Cache the model training for performance
def train_model_tuned(df_processed):
    # Define features (X) and target (y)
    X = df_processed.drop('Y house price of unit area', axis=1)
    y = df_processed['Y house price of unit area']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the parameter grid for GridSearchCV
    # These ranges are examples; you might need to adjust them based on your data.
    param_grid = {
        'n_estimators': [50, 100, 200],  # Number of trees in the forest
        'max_depth': [10, 20, 30, None], # Maximum depth of the tree (None means unlimited)
        'min_samples_split': [2, 5, 10], # Minimum number of samples required to split an internal node
        'min_samples_leaf': [1, 2, 4]    # Minimum number of samples required to be at a leaf node
    }

    # Initialize RandomForestRegressor
    rf = RandomForestRegressor(random_state=42)

    # Set up K-Fold Cross-Validation
    # n_splits: number of folds (e.g., 5-10 is common)
    # shuffle: ensures randomness in splits
    # random_state: for reproducibility
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize GridSearchCV
    # estimator: the model to tune
    # param_grid: the parameters to search
    # cv: cross-validation strategy
    # scoring: metric to optimize (e.g., 'neg_mean_squared_error' for MSE, 'r2' for R-squared)
    # n_jobs: number of CPU cores to use (-1 means use all available)
    # verbose: controls the verbosity of the output
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv,
                               scoring='neg_mean_squared_error', n_jobs=-1, verbose=0) # Set verbose to 0 for Streamlit

    # Fit GridSearchCV to the training data
    with st.spinner("Training model with GridSearchCV... This may take a moment."):
        grid_search.fit(X_train, y_train)

    # Get the best estimator (model with optimal hyperparameters)
    best_model = grid_search.best_estimator_

    # Make predictions on the training and test set using the best model
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    # Evaluate the best model
    mse_train = mean_squared_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)

    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)

    return best_model, X_train, X_test, y_train, y_test, y_pred_train, y_pred_test, grid_search.best_params_

# --- Streamlit App Layout ---
def main():
    st.set_page_config(layout="wide", page_title="Real Estate Price Prediction")

    st.title("🏡 Real Estate Price Prediction Dashboard")
    st.markdown("This application analyzes real estate data, visualizes key trends, and predicts house prices using a Random Forest Regressor model.")

    df = load_and_preprocess_data()
    # Call the new tuned model training function
    model, X_train, X_test, y_train, y_test, y_pred_train, y_pred_test, best_params = train_model_tuned(df)

    # --- Sidebar Navigation ---
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Data Overview", "Exploratory Data Analysis (EDA)", "Model Performance", "Prediction Visualizations", "Make a Prediction"])

    if page == "Data Overview":
        st.header("🔍 Data Overview")
        st.subheader("First 5 Rows of the Dataset")
        st.dataframe(df.head())

        st.subheader("Dataset Description")
        st.write(df.describe())

        st.subheader("Dataset Information")
        buffer = StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

    elif page == "Exploratory Data Analysis (EDA)":
        st.header("📊 Exploratory Data Analysis (EDA)")

        # 1. Histplot of House Price of Unit Area
        st.subheader("Distribution of House Price of Unit Area")
        fig1, ax1 = plt.subplots()
        sns.histplot(df['Y house price of unit area'], kde=True, ax=ax1)
        ax1.set_title('Distribution of House Price of Unit Area')
        ax1.set_xlabel('House Price of Unit Area')
        ax1.set_ylabel('Frequency')
        st.pyplot(fig1)

        # 2. Bar Plot: Average House Price by Number of Convenience Stores
        st.subheader("Average House Price by Number of Convenience Stores")
        fig2, ax2 = plt.subplots()
        sns.barplot(x='X4 number of convenience stores', y='Y house price of unit area', data=df, errorbar=None, ax=ax2)
        ax2.set_title('Average House Price by Number of Convenience Stores')
        ax2.set_xlabel('Number of Convenience Stores')
        ax2.set_ylabel('Average House Price of Unit Area')
        st.pyplot(fig2)

        # 3. Box Plot: House Price of Unit Area by Number of Convenience Stores
        st.subheader("House Price Distribution by Number of Convenience Stores")
        fig3, ax3 = plt.subplots()
        sns.boxplot(x='X4 number of convenience stores', y='Y house price of unit area', data=df, ax=ax3)
        ax3.set_title('House Price Distribution by Number of Convenience Stores')
        ax3.set_xlabel('Number of Convenience Stores')
        ax3.set_ylabel('House Price of Unit Area')
        st.pyplot(fig3)

        # 4. Scatter Plot: Distance to MRT vs. House Price of Unit Area
        st.subheader("Distance to Nearest MRT Station vs. House Price")
        fig4, ax4 = plt.subplots()
        sns.scatterplot(x='X3 distance to the nearest MRT station', y='Y house price of unit area', data=df, ax=ax4)
        ax4.set_title('Distance to Nearest MRT Station vs. House Price')
        ax4.set_xlabel('Distance to Nearest MRT Station')
        ax4.set_ylabel('House Price of Unit Area')
        st.pyplot(fig4)

        # 5. Scatter Plot: Latitude vs. Longitude (Geographical Price Distribution)
        fig5, ax5 = plt.subplots()
        sc = ax5.scatter(df['X6 longitude'], df['X5 latitude'],
                         c=df['Y house price of unit area'],
                         s=50, alpha=0.7)
        plt.colorbar(sc, ax=ax5, label='House Price')
        ax5.set_title('Geographical Distribution of House Prices')
        ax5.set_xlabel('Longitude')
        ax5.set_ylabel('Latitude')
        st.pyplot(fig5)


        # 6. Histplot of House Age
        st.subheader("Distribution of House Age")
        fig6, ax6 = plt.subplots()
        sns.histplot(df['X2 house age'], kde=True, bins=20, ax=ax6)
        ax6.set_title('Distribution of House Age')
        ax6.set_xlabel('House Age')
        ax6.set_ylabel('Frequency')
        st.pyplot(fig6)

    elif page == "Model Performance":
        st.header("🧠 Model Training & Evaluation")

        st.subheader("Random Forest Regressor Model")
        st.write("The model has been trained on the real estate dataset to predict house prices.")

        # Display best hyperparameters found
        st.subheader("Best Hyperparameters Found (from GridSearchCV):")
        for param, value in best_params.items():
            st.write(f"- `{param}`: `{value}`")

        # Evaluate the model on the training set
        mse_train = mean_squared_error(y_train, y_pred_train)
        r2_train = r2_score(y_train, y_pred_train)

        st.subheader("Training Set Performance (Tuned Model)")
        st.write(f"Mean Squared Error (MSE): `{mse_train:.2f}`")
        st.write(f"R-squared (R2): `{r2_train:.2f}`")

        # Evaluate the model on the test set
        mse_test = mean_squared_error(y_test, y_pred_test)
        r2_test = r2_score(y_test, y_pred_test)

        st.subheader("Test Set Performance (Tuned Model)")
        st.write(f"Mean Squared Error (MSE): `{mse_test:.2f}`")
        st.write(f"R-squared (R2): `{r2_test:.2f}`")

    elif page == "Prediction Visualizations":
        st.header("📈 Prediction Visualizations")

        # Actual vs. Predicted House Prices Plot
        st.subheader("Actual vs. Predicted House Prices (Test Set)")
        fig_ap, ax_ap = plt.subplots()
        ax_ap.scatter(y_test, y_pred_test, alpha=0.7)
        ax_ap.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) # Ideal line
        ax_ap.set_xlabel("Actual House Price of Unit Area")
        ax_ap.set_ylabel("Predicted House Price of Unit Area")
        ax_ap.set_title("Actual vs. Predicted House Prices")
        st.pyplot(fig_ap)

        # Residuals Plot
        st.subheader("Residuals Plot (Test Set)")
        residuals = y_test - y_pred_test
        fig_res, ax_res = plt.subplots()
        ax_res.scatter(y_pred_test, residuals, alpha=0.7)
        ax_res.axhline(y=0, color='r', linestyle='--') # Zero residual line
        ax_res.set_xlabel("Predicted House Price of Unit Area")
        ax_res.set_ylabel("Residuals (Actual - Predicted)")
        ax_res.set_title("Residuals Plot")
        st.pyplot(fig_res)

        st.subheader("Distribution of Residuals")
        fig_res_dist, ax_res_dist = plt.subplots()
        sns.histplot(residuals, kde=True, ax=ax_res_dist)
        ax_res_dist.set_title('Distribution of Residuals')
        ax_res_dist.set_xlabel('Residual Value')
        ax_res_dist.set_ylabel('Frequency')
        st.pyplot(fig_res_dist)

    elif page == "Make a Prediction":
        st.header("🔮 Make a House Price Prediction")
        st.write("Enter the parameters below to get a predicted house price.")

        # Get the feature names from the training data for consistent order
        feature_names = X_train.columns.tolist()

        input_data = {}
        col1, col2 = st.columns(2)

        # Create input widgets for each feature
        # Using iter_cols for better layout in two columns
        for i, feature in enumerate(feature_names):
            if i % 2 == 0:
                with col1:
                    if feature == 'X2 house age':
                        input_data[feature] = st.number_input(f"House Age (years):", min_value=0.0, max_value=100.0, value=20.0, step=0.1)
                    elif feature == 'X3 distance to the nearest MRT station':
                        input_data[feature] = st.number_input(f"Distance to nearest MRT (meters):", min_value=0.0, value=500.0, step=10.0)
                    elif feature == 'X4 number of convenience stores':
                        input_data[feature] = st.number_input(f"Number of Convenience Stores:", min_value=0, max_value=20, value=5, step=1)
                    elif feature == 'X5 latitude':
                        input_data[feature] = st.number_input(f"Latitude:", min_value=20.0, max_value=30.0, value=24.9, step=0.001, format="%.4f")
                    elif feature == 'X6 longitude':
                        input_data[feature] = st.number_input(f"Longitude:", min_value=120.0, max_value=122.0, value=121.5, step=0.001, format="%.4f")
                    elif feature == 'transaction_year':
                        input_data[feature] = st.number_input(f"Transaction Year:", min_value=2000, max_value=2025, value=2013, step=1)
                    elif feature == 'transaction_month':
                        input_data[feature] = st.number_input(f"Transaction Month:", min_value=1, max_value=12, value=7, step=1)
                    else:
                        input_data[feature] = st.number_input(f"{feature}:", value=0.0)
            else:
                with col2:
                    if feature == 'X2 house age':
                        input_data[feature] = st.number_input(f"House Age (years):", min_value=0.0, max_value=100.0, value=20.0, step=0.1, key=f"{feature}_input_col2")
                    elif feature == 'X3 distance to the nearest MRT station':
                        input_data[feature] = st.number_input(f"Distance to nearest MRT (meters):", min_value=0.0, value=500.0, step=10.0, key=f"{feature}_input_col2")
                    elif feature == 'X4 number of convenience stores':
                        input_data[feature] = st.number_input(f"Number of Convenience Stores:", min_value=0, max_value=20, value=5, step=1, key=f"{feature}_input_col2")
                    elif feature == 'X5 latitude':
                        input_data[feature] = st.number_input(f"Latitude:", min_value=20.0, max_value=30.0, value=24.9, step=0.001, format="%.4f", key=f"{feature}_input_col2")
                    elif feature == 'X6 longitude':
                        input_data[feature] = st.number_input(f"Longitude:", min_value=120.0, max_value=122.0, value=121.5, step=0.001, format="%.4f", key=f"{feature}_input_col2")
                    elif feature == 'transaction_year':
                        input_data[feature] = st.number_input(f"Transaction Year:", min_value=2000, max_value=2025, value=2013, step=1, key=f"{feature}_input_col2")
                    elif feature == 'transaction_month':
                        input_data[feature] = st.number_input(f"Transaction Month:", min_value=1, max_value=12, value=7, step=1, key=f"{feature}_input_col2")
                    else:
                        input_data[feature] = st.number_input(f"{feature}:", value=0.0, key=f"{feature}_input_col2")

        if st.button("Predict House Price"):
            # Create a DataFrame from the input data, ensuring column order matches X_train
            input_df = pd.DataFrame([input_data])
            input_df = input_df[feature_names] # Reorder columns to match training data

            try:
                prediction = model.predict(input_df)[0]
                st.success(f"**Predicted House Price of Unit Area: ${prediction:.2f}**")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.warning("Please ensure all input fields are filled correctly.")


if __name__ == '__main__':
    main()