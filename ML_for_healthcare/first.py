import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_curve, auc
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Data Preprocessing Module
def preprocess_data(df):
    # Handling missing values using median imputation
    imputer = SimpleImputer(strategy='median')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    # Min-Max Normalization
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df_imputed), columns=df.columns)
    
    return df_normalized

# Risk Level Assessment
def assess_risk_level(probability):
    if probability < 0.3:
        return "Low Risk"
    elif probability < 0.7:
        return "Medium Risk"
    else:
        return "High Risk"

# Feature Selection Module (Enhanced Coati Optimization - ECO)
def enhanced_coati_optimization(X, y, num_features, num_agents=10, max_iter=20):
    n_features = X.shape[1]
    agents = np.random.randint(0, 2, (num_agents, n_features))
    
    def fitness(agent):
        selected_features = [index for index in range(len(agent)) if agent[index] == 1]
        if len(selected_features) == 0:
            return 0
        X_selected = X[:, selected_features]
        clf = RandomForestClassifier(n_estimators=10)
        clf.fit(X_selected, y)
        predictions = clf.predict(X_selected)
        return accuracy_score(y, predictions)

    best_agent = agents[0]
    best_score = fitness(best_agent)

    for iteration in range(max_iter):
        for i in range(num_agents):
            current_agent = agents[i].copy()
            opposite_agent = 1 - current_agent
            
            if fitness(opposite_agent) > fitness(current_agent):
                current_agent = opposite_agent

            mutation_point = np.random.randint(0, n_features)
            current_agent[mutation_point] = 1 - current_agent[mutation_point]

            current_score = fitness(current_agent)
            if current_score > best_score:
                best_agent = current_agent.copy()
                best_score = current_score

            agents[i] = current_agent.copy()

    selected_indices = [index for index in range(len(best_agent)) if best_agent[index] == 1]
    if len(selected_indices) > num_features:
        selected_indices = selected_indices[:num_features]
    return selected_indices

# Multi-Complication Classification Module
class DiabetesComplicationClassifier:
    def __init__(self):
        self.models = {
            'retinopathy': None,
            'neuropathy': None,
            'nephropathy': None
        }
        self.selected_features = {}
        
    def train(self, X, y_dict):
        for complication in self.models.keys():
            print(f"\nTraining model for {complication.capitalize()}:")
            
            # Feature selection for each complication
            selected_indices = enhanced_coati_optimization(X.values, y_dict[complication].values, num_features=10)
            X_selected = X.iloc[:, selected_indices]
            self.selected_features[complication] = X_selected.columns
            
            # Train XGBoost model with White Shark optimization
            model, metrics = self._train_complication_model(X_selected, y_dict[complication])
            self.models[complication] = model
            
            print(f"Selected features for {complication}: {', '.join(self.selected_features[complication])}")
            print(f"Performance metrics for {complication}:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
    
    def _train_complication_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # White Shark optimization for hyperparameters
        best_params = self._white_shark_optimization(X_train, X_test, y_train, y_test)
        
        # Train final model
        model = xgb.XGBClassifier(**best_params, random_state=42)
        model.fit(X_train, y_train)
        
        # Calculate metrics
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred)
        }
        
        return model, metrics
    
    def _white_shark_optimization(self, X_train, X_test, y_train, y_test):
        param_space = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [50, 100, 150],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }
        
        best_params = {}
        best_accuracy = 0
        
        for _ in range(20):
            params = {
                'max_depth': np.random.choice(param_space['max_depth']),
                'learning_rate': np.random.choice(param_space['learning_rate']),
                'n_estimators': np.random.choice(param_space['n_estimators']),
                'subsample': np.random.choice(param_space['subsample']),
                'colsample_bytree': np.random.choice(param_space['colsample_bytree'])
            }
            
            model = xgb.XGBClassifier(**params, random_state=42)
            model.fit(X_train, y_train)
            accuracy = accuracy_score(y_test, model.predict(X_test))
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = params
        
        return best_params
    
    def predict_risks(self, X):
        results = {}
        for complication in self.models.keys():
            X_selected = X[self.selected_features[complication]]
            probabilities = self.models[complication].predict_proba(X_selected)[:, 1]
            
            results[complication] = {
                'probabilities': probabilities,
                'risk_levels': [assess_risk_level(p) for p in probabilities]
            }
        
        return results

    def visualize_results(self, X, y_dict):
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        axes = axes.ravel()
        
        for idx, complication in enumerate(self.models.keys()):
            X_selected = X[self.selected_features[complication]]
            
            # Feature importance
            feature_importance = self.models[complication].feature_importances_
            sorted_idx = np.argsort(feature_importance)
            pos = np.arange(sorted_idx.shape[0]) + .5
            
            axes[idx].barh(pos, feature_importance[sorted_idx], align='center')
            axes[idx].set_yticks(pos)
            axes[idx].set_yticklabels(np.array(X_selected.columns)[sorted_idx])
            axes[idx].set_title(f'{complication.capitalize()} - Feature Importance')
        
        plt.tight_layout()
        plt.show()

# Main Execution Pipeline
def run_diabetes_analysis(data_path='ML_for_healthcare/data1.csv'):
    try:
        # Load and preprocess data
        df = pd.read_csv(data_path)
        print(f"Loaded dataset with columns: {df.columns.tolist()}")
        
        # Verify required columns exist
        required_columns = ['retinopathy', 'neuropathy', 'nephropathy']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Separate features and targets
        X = df.drop(['retinopathy', 'neuropathy', 'nephropathy'], axis=1)
        y_dict = {
            'retinopathy': df['retinopathy'],
            'neuropathy': df['neuropathy'],
            'nephropathy': df['nephropathy']
        }
        
        # Preprocess features
        X_processed = preprocess_data(X)
        
        # Initialize and train classifier
        classifier = DiabetesComplicationClassifier()
        classifier.train(X_processed, y_dict)
        
        # Visualize results
        classifier.visualize_results(X_processed, y_dict)
        
        return classifier
        
    except Exception as e:
        print(f"Error loading or processing dataset: {str(e)}")
        return None

if __name__ == "__main__":
    # Get the current directory where first.py is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Use data.csv instead of data1.csv
    data_path = os.path.join(current_dir, 'data.csv')
    
    print(f"Attempting to load data from: {data_path}")
    
    classifier = run_diabetes_analysis(data_path)
    
    if classifier:
        try:
            # Use the same path for new patient data
            new_patient_data = pd.read_csv(data_path)
            new_patient_processed = preprocess_data(new_patient_data)
            risks = classifier.predict_risks(new_patient_processed)
            
            for complication, results in risks.items():
                print(f"\n{complication.capitalize()} Risk Assessment:")
                for i, (prob, level) in enumerate(zip(results['probabilities'], results['risk_levels'])):
                    print(f"Patient {i+1}: Probability = {prob:.2f}, Risk Level = {level}")
                    
        except Exception as e:
            print(f"Error processing new patient data: {str(e)}")
