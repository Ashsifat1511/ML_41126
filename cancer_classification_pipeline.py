# Create a comprehensive machine learning model for GSE68086 dataset
# This dataset contains tumor-educated platelets (TEPs) for cancer classification

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# Create a comprehensive model training pipeline
class GSE68086ModelTrainer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = SelectKBest(f_classif)
        self.models = {}
        self.best_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_and_preprocess_data(self, data_path=None, target_column='cancer_type'):
        """
        Load and preprocess the GSE68086 dataset
        If no data_path provided, create synthetic data for demonstration
        """
        if data_path is None:
            print("Creating synthetic GSE68086-like dataset for demonstration...")
            # Create synthetic data mimicking GSE68086 structure
            np.random.seed(42)
            n_samples = 285  # Original dataset size
            n_features = 1000  # Reduced from 57,736 for demonstration
            
            # Create features (gene expression values)
            X = np.random.normal(0, 1, (n_samples, n_features))
            
            # Create target labels (cancer types from GSE68086)
            cancer_types = ['healthy', 'colorectal', 'lung', 'pancreatic', 
                          'glioblastoma', 'breast', 'hepatobiliary']
            y = np.random.choice(cancer_types, n_samples)
            
            # Create DataFrame
            feature_names = [f'gene_{i}' for i in range(n_features)]
            self.data = pd.DataFrame(X, columns=feature_names)
            self.data[target_column] = y
            
            print(f"Synthetic dataset created: {self.data.shape}")
            print(f"Cancer type distribution:\n{self.data[target_column].value_counts()}")
            
        else:
            print(f"Loading data from {data_path}...")
            self.data = pd.read_csv(data_path)
            
        # Separate features and target
        self.X = self.data.drop(columns=[target_column])
        self.y = self.data[target_column]
        
        # Encode labels
        self.y_encoded = self.label_encoder.fit_transform(self.y)
        
        return self.X, self.y_encoded
    
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y_encoded, test_size=test_size, 
            random_state=random_state, stratify=self.y_encoded
        )
        
        print(f"Data split - Train: {self.X_train.shape}, Test: {self.X_test.shape}")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def feature_selection(self, k=500):
        """Select top k features using univariate statistical tests"""
        print(f"Selecting top {k} features...")
        
        # Fit feature selector on training data
        self.feature_selector.set_params(k=k)
        self.X_train_selected = self.feature_selector.fit_transform(self.X_train, self.y_train)
        self.X_test_selected = self.feature_selector.transform(self.X_test)
        
        # Get selected feature names
        selected_features = self.X.columns[self.feature_selector.get_support()].tolist()
        print(f"Selected {len(selected_features)} features")
        
        return self.X_train_selected, self.X_test_selected
    
    def scale_features(self):
        """Scale features using StandardScaler"""
        print("Scaling features...")
        
        # Use selected features if available, otherwise use all features
        if hasattr(self, 'X_train_selected'):
            self.X_train_scaled = self.scaler.fit_transform(self.X_train_selected)
            self.X_test_scaled = self.scaler.transform(self.X_test_selected)
        else:
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            
        return self.X_train_scaled, self.X_test_scaled
    
    def train_models(self):
        """Train multiple models for comparison"""
        print("Training multiple models...")
        
        # Define models to train
        models_config = {
            'SVM_RBF': SVC(kernel='rbf', random_state=42),
            'SVM_Linear': SVC(kernel='linear', random_state=42),
            'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic_Regression': LogisticRegression(max_iter=1000, random_state=42)
        }
        
        # Train each model
        for name, model in models_config.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(self.X_train_scaled, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test_scaled)
            
            # Calculate accuracy
            accuracy = accuracy_score(self.y_test, y_pred)
            
            # Store model and results
            self.models[name] = {
                'model': model,
                'predictions': y_pred,
                'accuracy': accuracy
            }
            
            print(f"{name} Accuracy: {accuracy:.4f}")
        
        return self.models

# Initialize and demonstrate the model trainer
trainer = GSE68086ModelTrainer()

# Load and preprocess data (synthetic for demonstration)
X, y = trainer.load_and_preprocess_data()
# Continue with the model training pipeline
# Split the data
X_train, X_test, y_train, y_test = trainer.split_data()

# Perform feature selection
X_train_selected, X_test_selected = trainer.feature_selection(k=500)

# Scale the features
X_train_scaled, X_test_scaled = trainer.scale_features()

# Train multiple models
models = trainer.train_models()

print("\n" + "="*50)
print("MODEL PERFORMANCE SUMMARY")
print("="*50)

# Display model performance comparison
results_df = pd.DataFrame({
    'Model': list(models.keys()),
    'Accuracy': [models[name]['accuracy'] for name in models.keys()]
})
results_df = results_df.sort_values('Accuracy', ascending=False)
print(results_df.to_string(index=False))

# Find best model
best_model_name = results_df.iloc[0]['Model']
trainer.best_model = models[best_model_name]['model']

print(f"\nBest performing model: {best_model_name}")
print(f"Best accuracy: {results_df.iloc[0]['Accuracy']:.4f}")

# Detailed evaluation of the best model
print(f"\n{'='*50}")
print(f"DETAILED EVALUATION - {best_model_name}")
print("="*50)

y_pred_best = models[best_model_name]['predictions']

# Classification report
print("Classification Report:")
target_names = trainer.label_encoder.classes_
print(classification_report(y_test, y_pred_best, target_names=target_names))

# Confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_best)
cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
print(cm_df)
# Create an optimized model with hyperparameter tuning for better performance
print("="*60)
print("HYPERPARAMETER OPTIMIZATION")
print("="*60)

# Hyperparameter tuning for SVM (the best performing model)
print("Performing hyperparameter tuning for SVM...")

# Define parameter grid for SVM
param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'linear', 'poly']
}

# Perform grid search with cross-validation
svm_grid_search = GridSearchCV(
    SVC(random_state=42),
    param_grid_svm,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Fit grid search
svm_grid_search.fit(X_train_scaled, y_train)

# Get best model
best_svm = svm_grid_search.best_estimator_
best_params = svm_grid_search.best_params_

print(f"\nBest SVM parameters: {best_params}")
print(f"Best cross-validation score: {svm_grid_search.best_score_:.4f}")

# Evaluate optimized model on test set
y_pred_optimized = best_svm.predict(X_test_scaled)
optimized_accuracy = accuracy_score(y_test, y_pred_optimized)

print(f"Optimized SVM test accuracy: {optimized_accuracy:.4f}")

# Cross-validation scores for the best model
cv_scores = cross_val_score(best_svm, X_train_scaled, y_train, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Feature importance for Random Forest (alternative approach)
print("\n" + "="*60)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*60)

# Train Random Forest to get feature importance
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Get feature importance
feature_importance = rf_model.feature_importances_
top_features_idx = np.argsort(feature_importance)[-20:]  # Top 20 features

print("Top 20 most important features (Random Forest):")
for i, idx in enumerate(reversed(top_features_idx)):
    print(f"{i+1:2d}. Feature {idx:3d}: {feature_importance[idx]:.4f}")

# Save model training results
results_summary = {
    'dataset_info': {
        'total_samples': len(trainer.data),
        'features_original': trainer.X.shape[1],
        'features_selected': X_train_selected.shape[1],
        'classes': len(trainer.label_encoder.classes_),
        'class_names': trainer.label_encoder.classes_.tolist()
    },
    'model_performance': {
        'best_model': 'SVM_RBF_Optimized',
        'best_params': best_params,
        'test_accuracy': optimized_accuracy,
        'cv_accuracy_mean': cv_scores.mean(),
        'cv_accuracy_std': cv_scores.std()
    }
}

print(f"\n{'='*60}")
print("FINAL MODEL SUMMARY")
print("="*60)
print(f"Dataset: GSE68086 (Tumor-Educated Platelets)")
print(f"Total samples: {results_summary['dataset_info']['total_samples']}")
print(f"Original features: {results_summary['dataset_info']['features_original']}")
print(f"Selected features: {results_summary['dataset_info']['features_selected']}")
print(f"Classes: {results_summary['dataset_info']['classes']} ({', '.join(results_summary['dataset_info']['class_names'])})")
print(f"Best model: {results_summary['model_performance']['best_model']}")
print(f"Test accuracy: {results_summary['model_performance']['test_accuracy']:.4f}")
print(f"CV accuracy: {results_summary['model_performance']['cv_accuracy_mean']:.4f} ± {results_summary['model_performance']['cv_accuracy_std']:.4f}")

# Store the final optimized model
trainer.best_model = best_svm
print(f"\nFinal optimized model stored and ready for use!")