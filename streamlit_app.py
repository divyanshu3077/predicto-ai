import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import streamlit as st

# Function to preprocess the data
def preprocess_data(data):
    # Convert date columns to datetime and extract features
    for col in data.columns:
        if data[col].dtype == 'object':
            try:
                data[col] = pd.to_datetime(data[col])
                data[col + '_year'] = data[col].dt.year
                data[col + '_month'] = data[col].dt.month
                data[col + '_day'] = data[col].dt.day
                data = data.drop(col, axis=1)
            except ValueError:
                pass

    # Custom encoding for categorical variables
    for col in data.columns:
        if data[col].dtype == 'object':
            unique_values = data[col].unique()
            encoding = {val: idx for idx, val in enumerate(unique_values)}
            data[col] = data[col].map(encoding)

    return data

# Custom data splitting function
def train_test_split_custom(X, y, test_size=0.2, random_state=None):
    np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split = int(len(indices) * (1 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    return X_train, X_test, y_train, y_test

# Linear Regression implementation
def linear_regression(X_train, y_train, X_test):
    X_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]  # Add intercept term
    theta = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)
    
    X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]  # Add intercept term
    predictions = X_test.dot(theta)
    
    return predictions, theta

# Ridge Regression implementation
class RidgeRegressor:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        n_samples, n_features = X.shape
        self.theta = np.linalg.inv(X.T.dot(X) + self.alpha * np.eye(n_features)).dot(X.T).dot(y)
    
    def predict(self, X):
        return X.dot(self.theta)

# Lasso Regression implementation
class LassoRegressor:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        n_samples, n_features = X.shape
        self.theta = np.linalg.inv(X.T.dot(X) + self.alpha * np.eye(n_features)).dot(X.T).dot(y)
    
    def predict(self, X):
        return X.dot(self.theta)

# Gradient Boosting Regression implementation
class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
    
    def fit(self, X, y):
        y_pred = np.zeros(len(y))
        for _ in range(self.n_estimators):
            residual = y - y_pred
            model = DecisionTreeRegressor(max_depth=3)  # Example with max_depth
            model.fit(X, residual)
            self.models.append(model)
            y_pred += self.learning_rate * model.predict(X)
    
    def predict(self, X):
        y_pred = np.zeros(len(X))
        for model in self.models:
            y_pred += self.learning_rate * model.predict(X)
        return y_pred

class DecisionTreeRegressor:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n_samples, self.n_features = X.shape
        self.tree = self._grow_tree(X, y, depth=0)
    
    def _grow_tree(self, X, y, depth):
        if self.max_depth is not None and depth >= self.max_depth or len(np.unique(y)) == 1:
            return np.mean(y)
        
        feat_idx = np.random.choice(self.n_features, int(np.sqrt(self.n_features)), replace=False)
        feat_threshold = np.random.rand(len(feat_idx))  # Adjusted to match feat_idx length

        splits = X[:, feat_idx] < feat_threshold.reshape(1, -1)

        if np.any(splits[:, 0]) and np.any(splits[:, 1]):
            return np.mean(y)
        
        return splits
    
    def predict(self, X):
        # Example prediction for a decision tree
        return np.mean(X, axis=1)  # Replace with actual prediction logic


# R-squared calculation
def r2_score(y_true, y_pred):
    mean_y_true = np.mean(y_true)
    ss_tot = np.sum((y_true - mean_y_true)**2)
    ss_res = np.sum((y_true - y_pred)**2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

# Mean Squared Error calculation
def mean_squared_error(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    return mse

# Main Streamlit app
def main():
    st.title('Custom Model Predictor')
    
    # Upload CSV data
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Data preview:")
        st.write(data.head())

        # Preprocess the data
        data = preprocess_data(data)
        st.write("Preprocessed Data:")
        st.write(data.head())

        # Select columns for prediction
        features = st.multiselect("Select feature columns", data.columns.tolist(), default=data.columns.tolist()[:-1])
        target = st.selectbox("Select target column", data.columns.tolist(), index=len(data.columns.tolist())-1)

        # Select prediction task
        task = st.selectbox("Select prediction task", ("Stock Sale", "Real Estate Prices", "Salaries", "Cryptocurrency"))

        if features and target:
            X = data[features].values
            y = data[target].values

            # Custom train-test split
            X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2, random_state=42)

            # Select prediction algorithm
            algorithm = st.selectbox(
                "Select prediction algorithm", 
                (
                    "Linear Regression", 
                    "Ridge Regression", 
                    "Lasso Regression", 
                    "Gradient Boosting Regression", 
                    "Decision Tree Regression"
                )
            )

            if st.button("Train and Evaluate"):
                st.subheader(f"Training and Evaluating using {algorithm}")
                
                # Train and evaluate the model
                if algorithm == "Linear Regression":
                    predictions, theta = linear_regression(X_train, y_train, X_test)
                    score = r2_score(y_test, predictions)
                    mse = mean_squared_error(y_test, predictions)
                    st.subheader("Evaluation Metrics")
                    st.write(f"R-squared: {score:.2f}")
                    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
                    fig, ax = plt.subplots()
                    ax.scatter(y_test, predictions, color='blue', label='Predicted')
                    ax.scatter(y_test, y_test, color='red', label='Actual')
                    ax.set_xlabel('Actual')
                    ax.set_ylabel('Predicted')
                    ax.set_title('Actual vs Predicted')
                    ax.legend()
                    st.pyplot(fig)

                elif algorithm == "Ridge Regression":
                    alpha = st.slider("Select alpha value", min_value=0.0, max_value=1.0, value=1.0, step=0.01)
                    model = RidgeRegressor(alpha=alpha)
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    score = r2_score(y_test, predictions)
                    mse = mean_squared_error(y_test, predictions)

                elif algorithm == "Lasso Regression":
                    alpha = st.slider("Select alpha value", min_value=0.0, max_value=1.0, value=1.0, step=0.01)
                    model = LassoRegressor(alpha=alpha)
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    score = r2_score(y_test, predictions)
                    mse = mean_squared_error(y_test, predictions)
                    
                elif algorithm == "Gradient Boosting Regression":
                    n_estimators = st.number_input("Number of estimators", min_value=1, max_value=1000, value=100, step=10)
                    learning_rate = st.slider("Learning rate", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
                    model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate)
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    score = r2_score(y_test, predictions)
                    mse = mean_squared_error(y_test, predictions)
                
                elif algorithm == "Decision Tree Regression":
                    model = DecisionTreeRegressor(max_depth=3)  
                    model.fit(X_train, y_train)  
                    predictions = model.predict(X_test)  
                    score = r2_score(y_test, predictions)  
                    mse = mean_squared_error(y_test, predictions)  

                if algorithm in ("Ridge Regression", "Lasso Regression", "Gradient Boosting Regression", "Decision Tree Regression"):
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    score = r2_score(y_test, predictions)
                    mse = mean_squared_error(y_test, predictions)
                    st.subheader("Evaluation Metrics")
                    st.write(f"R-squared: {score:.2f}")
                    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
                    fig, ax = plt.subplots()
                    ax.scatter(y_test, predictions, color='blue', label='Predicted')
                    ax.scatter(y_test, y_test, color='red', label='Actual')
                    ax.set_xlabel('Actual')
                    ax.set_ylabel('Predicted')
                    ax.set_title('Actual vs Predicted')
                    ax.legend()
                    st.pyplot(fig)

if __name__ == '__main__':
    main()
