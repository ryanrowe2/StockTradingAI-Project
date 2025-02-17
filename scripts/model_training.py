# scripts/model_training.py

import pandas as pd
import numpy as np
import os
import json
import logging
import argparse
from functools import reduce
from sklearn.model_selection import train_test_split
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt

# Import additional metrics from scikit-learn.
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# Configure logging for diagnostics
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


class BaseModelTrainer:
    """
    Base class for training and evaluating probabilistic models.
    This class encapsulates common functionality such as data loading,
    target creation, evaluation, and logging.
    """
    def __init__(self, config):
        """
        Initialize the trainer with configuration parameters.
        
        Parameters:
            config (dict): Contains:
                - processed_data_path: Path to processed CSV file.
                - cpts_path: Path to CPT JSON file.
                - test_size: Fraction of data used for testing.
                - random_state: Seed for train/test splitting.
                - inference_method: 'VE' or 'enumeration'
                - num_bins: Number of bins for discretization.
        """
        self.processed_data_path = config.get('processed_data_path')
        self.cpts_path = config.get('cpts_path')
        self.test_size = config.get('test_size', 0.2)
        self.random_state = config.get('random_state', 42)
        self.inference_method = config.get('inference_method', 'VE')
        self.config = config

    def load_processed_data(self):
        """
        Load the processed stock market data from a CSV file.
        
        Returns:
            pd.DataFrame: DataFrame containing the processed data.
        """
        logging.info(f"Loading processed data from: {self.processed_data_path}")
        try:
            df = pd.read_csv(self.processed_data_path)
            logging.info(f"Data loaded successfully with shape: {df.shape}")
            return df
        except Exception as e:
            logging.error(f"Failed to load processed data: {e}")
            raise

    def load_cpts(self):
        """
        Load the Conditional Probability Tables (CPTs) from a JSON file.
        
        Returns:
            dict: Dictionary of CPTs.
        """
        logging.info(f"Loading CPTs from: {self.cpts_path}")
        try:
            with open(self.cpts_path, 'r') as f:
                cpts = json.load(f)
            logging.info("CPTs loaded successfully.")
            return cpts
        except Exception as e:
            logging.error(f"Failed to load CPTs: {e}")
            raise

    @staticmethod
    def prepare_target(df):
        """
        Create a binary target column 'Trend' based on the 'Close' price.
        'Trend' is 1 if the next day's Close is higher than today's, otherwise 0.
        Rows with missing values (due to shifting) are dropped.
        
        Returns:
            pd.DataFrame: DataFrame with the new 'Trend' column and dropped NaNs.
        """
        logging.info("Preparing target variable 'Trend'.")
        df['Trend'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df.dropna(inplace=True)
        logging.info(f"Target prepared. Data shape is now: {df.shape}")
        return df

    def evaluate_model(self, model, df, features, target):
        """
        Evaluate the model using Variable Elimination (VE) inference.
        Computes accuracy, precision, recall, F1, confusion matrix, and ROC curve.
        
        Returns:
            tuple: (predictions, actuals, accuracy)
        """
        logging.info("Starting model evaluation using Variable Elimination.")
        train, test = train_test_split(df, test_size=self.test_size, random_state=self.random_state)
        logging.info(f"Training set shape: {train.shape}; Testing set shape: {test.shape}")
        
        # Re-fit model on training data.
        model.fit(train[features + [target]], estimator=MaximumLikelihoodEstimator)
        logging.info("Model re-trained on training set for evaluation.")
        
        inference = VariableElimination(model)
        predictions = []
        predicted_probs = []
        actuals = test[target].tolist()
        
        logging.info("Performing inference on test data (VE).")
        for index, row in test[features].iterrows():
            evidence = row.to_dict()
            query_result = inference.query(variables=[target], evidence=evidence, show_progress=False)
            prob_target_1 = query_result.values[1]
            predicted_probs.append(prob_target_1)
            predictions.append(1 if prob_target_1 >= 0.5 else 0)
        
        accuracy = np.mean(np.array(predictions) == np.array(actuals))
        precision = precision_score(actuals, predictions)
        recall = recall_score(actuals, predictions)
        f1 = f1_score(actuals, predictions)
        cm = confusion_matrix(actuals, predictions)
        fpr, tpr, thresholds = roc_curve(actuals, predicted_probs)
        roc_auc = auc(fpr, tpr)
        
        logging.info(f"VE: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}")
        logging.info(f"VE: Confusion Matrix:\n{cm}")
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix (VE)")
        plt.show()
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (VE)')
        plt.legend(loc="lower right")
        plt.show()
        
        return predictions, actuals, accuracy

    def evaluate_model_enumeration(self, model, df, features, target):
        """
        Evaluate the model using a brute-force enumeration inference approach.
        (Note: In our case, all features are observed so this yields similar results as VE.)
        
        Returns:
            tuple: (predictions, actuals, accuracy)
        """
        logging.info("Starting model evaluation using Enumeration inference.")
        train, test = train_test_split(df, test_size=self.test_size, random_state=self.random_state)
        logging.info(f"Training set shape: {train.shape}; Testing set shape: {test.shape}")
        
        model.fit(train[features + [target]], estimator=MaximumLikelihoodEstimator)
        logging.info("Model re-trained on training set for evaluation (Enumeration).")
        
        inference = VariableElimination(model)  # Using VE as a proxy for enumeration.
        predictions = []
        predicted_probs = []
        actuals = test[target].tolist()
        
        logging.info("Performing inference on test data (Enumeration).")
        for index, row in test[features].iterrows():
            evidence = row.to_dict()
            query_result = inference.query(variables=[target], evidence=evidence, show_progress=False)
            prob_target_1 = query_result.values[1]
            predicted_probs.append(prob_target_1)
            predictions.append(1 if prob_target_1 >= 0.5 else 0)
        
        accuracy = np.mean(np.array(predictions) == np.array(actuals))
        precision = precision_score(actuals, predictions)
        recall = recall_score(actuals, predictions)
        f1 = f1_score(actuals, predictions)
        cm = confusion_matrix(actuals, predictions)
        fpr, tpr, thresholds = roc_curve(actuals, predicted_probs)
        roc_auc = auc(fpr, tpr)
        
        logging.info(f"Enumeration: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}")
        logging.info(f"Enumeration: Confusion Matrix:\n{cm}")
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix (Enumeration)")
        plt.show()
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (Enumeration)')
        plt.legend(loc="lower right")
        plt.show()
        
        return predictions, actuals, accuracy


class BayesianNetworkTrainer(BaseModelTrainer):
    """
    Trainer class for building and evaluating a Bayesian Network.
    Inherits common functionality from BaseModelTrainer.
    """
    def __init__(self, config):
        super().__init__(config)

    def build_model(self, df, features, target):
        """
        Build the Bayesian network model.
        
        Parameters:
            df (pd.DataFrame): Processed DataFrame.
            features (list): List of feature names (ideally binned).
            target (str): Target variable name.
            
        Returns:
            BayesianNetwork: Trained Bayesian network model.
        """
        logging.info("Building Bayesian network model in BayesianNetworkTrainer.")
        # Define network structure: each feature influences the target.
        edges = [(feature, target) for feature in features]
        model = BayesianNetwork(edges)
        logging.info(f"Network structure (edges): {edges}")
        model.fit(df[features + [target]], estimator=MaximumLikelihoodEstimator)
        logging.info("Model training complete in BayesianNetworkTrainer.")
        for node in model.nodes():
            cpd = model.get_cpds(node)
            logging.info(f"Learned CPD for {node}: {cpd}")
        return model


def main():
    # Parse command-line arguments for configuration.
    import argparse
    parser = argparse.ArgumentParser(description="Train and evaluate a Bayesian Network for Stock Trading AI.")
    parser.add_argument('--processed_data_path', type=str, default=os.path.join('..', 'data', 'processed', 'all_stocks_2006-01-01_to_2018-01-01.csv'),
                        help='Path to the processed CSV file.')
    parser.add_argument('--cpts_path', type=str, default=os.path.join('..', 'data', 'processed', 'all_stocks_2006-01-01_to_2018-01-01_CPTs.json'),
                        help='Path to the CPT JSON file.')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Fraction of data to use for testing.')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for train/test splitting.')
    parser.add_argument('--num_bins', type=int, default=5,
                        help='Number of bins for discretization (if applicable).')
    parser.add_argument('--inference_method', type=str, choices=['VE', 'enumeration'], default='VE',
                        help='Inference method to use: VE for Variable Elimination, or enumeration.')
    args = parser.parse_args()
    
    # Create configuration dictionary from command-line arguments.
    config = vars(args)
    
    # Instantiate the trainer using the BayesianNetworkTrainer.
    trainer = BayesianNetworkTrainer(config)
    
    # Load processed data and CPTs.
    df = trainer.load_processed_data()
    cpts = trainer.load_cpts()
    logging.info(f"Processed data shape: {df.shape}")
    
    # Prepare the binary target 'Trend'.
    df = trainer.prepare_target(df)
    
    # Identify numerical columns and select a subset of features.
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Trend' in numeric_cols:
        numeric_cols.remove('Trend')
    # For simplicity, choose the first three numerical columns.
    features = numeric_cols[:3]
    # Prefer binned versions if they exist.
    features_binned = [f"{col}_binned" if f"{col}_binned" in df.columns else col for col in features]
    target = 'Trend'
    
    logging.info(f"Features used for modeling: {features_binned}")
    
    # Build the Bayesian network model.
    model = trainer.build_model(df, features_binned, target)
    
    # Choose inference method based on configuration.
    if config.get('inference_method', 'VE') == 'enumeration':
        predictions, actuals, accuracy = trainer.evaluate_model_enumeration(model, df, features_binned, target)
    else:
        predictions, actuals, accuracy = trainer.evaluate_model(model, df, features_binned, target)
    
    logging.info(f"Baseline Bayesian Network Model Accuracy: {accuracy:.4f}")
    
if __name__ == '__main__':
    main()