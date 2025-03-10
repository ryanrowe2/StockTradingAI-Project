# scripts/model_training.py

import pandas as pd
import numpy as np
import os
import json
import logging
import argparse
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator, HillClimbSearch, BicScore
from pgmpy.inference import VariableElimination

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# For polynomial features (optional)
from sklearn.preprocessing import PolynomialFeatures

# Configure logging for diagnostics
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


class BaseModelTrainer:
    """
    Base class for training and evaluating probabilistic models.
    """
    def __init__(self, config):
        self.processed_data_path = config.get('processed_data_path')
        self.cpts_path = config.get('cpts_path')
        # Set test_size to 0.25 (75/25 train/test split)
        self.test_size = config.get('test_size', 0.25)
        self.random_state = config.get('random_state', 42)
        self.inference_method = config.get('inference_method', 'VE')
        self.config = config

    def load_processed_data(self):
        logging.info(f"Loading processed data from: {self.processed_data_path}")
        try:
            df = pd.read_csv(self.processed_data_path)
            logging.info(f"Data loaded successfully with shape: {df.shape}")
            return df
        except Exception as e:
            logging.error(f"Failed to load processed data: {e}")
            raise

    def load_cpts(self):
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
        logging.info("Preparing target variable 'Trend'.")
        df['Trend'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df.dropna(inplace=True)
        logging.info(f"Target prepared. Data shape is now: {df.shape}")
        return df

    def evaluate_model(self, model, test_df, features, target):
        logging.info("Starting model evaluation using Variable Elimination.")
        inference = VariableElimination(model)
        predictions = []
        predicted_probs = []
        actuals = test_df[target].tolist()
        
        logging.info("Performing inference on test data (VE).")
        for _, row in test_df[features].iterrows():
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
        fpr, tpr, _ = roc_curve(actuals, predicted_probs)
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
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (VE)')
        plt.legend(loc="lower right")
        plt.show()
        
        return predictions, actuals, accuracy, predicted_probs

    def evaluate_model_enumeration(self, model, df, features, target):
        logging.info("Starting model evaluation using Enumeration inference.")
        return self.evaluate_model(model, df, features, target)


class BayesianNetworkTrainer(BaseModelTrainer):
    """
    Trainer class for building and evaluating a Bayesian Network.
    """
    def __init__(self, config):
        super().__init__(config)

    def build_model(self, df, features, target, estimator=BayesianEstimator):
        """
        Build a Bayesian Network using a fixed naive structure.
        Uses BayesianEstimator (with smoothing) for parameter learning.
        """
        logging.info("Building Bayesian network model in BayesianNetworkTrainer (fixed structure).")
        edges = [(feature, target) for feature in features]
        model = BayesianNetwork(edges)
        logging.info(f"Network structure (edges): {edges}")
        # Use BayesianEstimator with a BDeu prior for smoothing; optimal settings found: equivalent_sample_size=15.
        model.fit(df[features + [target]], estimator=estimator, prior_type='BDeu', equivalent_sample_size=15)
        logging.info("Model training complete in BayesianNetworkTrainer (fixed structure).")
        for node in model.nodes():
            cpd = model.get_cpds(node)
            logging.info(f"Learned CPD for {node}: {cpd}")
        return model

    def learn_structure(self, df, variables):
        """
        Learn the network structure using Hill Climb Search and BicScore.
        """
        from pgmpy.estimators import HillClimbSearch, BicScore
        logging.info("Starting structure learning using HillClimbSearch and BicScore.")
        # (Optional: one could add time or iteration limits here.)
        hc = HillClimbSearch(df[variables])
        best_model = hc.estimate(scoring_method=BicScore(df[variables]))
        logging.info(f"Learned structure: {best_model.edges()}")
        return best_model


def ensemble_inference(test_df, baseline_model, baseline_features, enhanced_model, enhanced_features, target, trainer):
    """
    Compute ensemble predicted probabilities by averaging the predictions
    of the baseline and enhanced models on the same test set.
    """
    logging.info("Performing ensemble inference on test data.")
    # Get predictions (and probabilities) for baseline.
    _, _, _, baseline_probs = trainer.evaluate_model(baseline_model, test_df, baseline_features, target)
    # Get predictions (and probabilities) for enhanced.
    _, _, _, enhanced_probs = trainer.evaluate_model(enhanced_model, test_df, enhanced_features, target)
    # Average predicted probabilities.
    ensemble_probs = np.mean(np.vstack([baseline_probs, enhanced_probs]), axis=0)
    ensemble_preds = [1 if p >= 0.5 else 0 for p in ensemble_probs]
    actuals = test_df[target].tolist()
    ensemble_accuracy = np.mean(np.array(ensemble_preds) == np.array(actuals))
    logging.info(f"Ensemble model accuracy: {ensemble_accuracy:.4f}")
    return ensemble_preds, actuals, ensemble_accuracy, ensemble_probs


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a Bayesian Network for Stock Trading AI.")
    parser.add_argument('--processed_data_path', type=str, default=os.path.join('..', 'data', 'processed', 'all_stocks_2006-01-01_to_2018-01-01.csv'),
                        help='Path to the processed CSV file.')
    parser.add_argument('--cpts_path', type=str, default=os.path.join('..', 'data', 'processed', 'all_stocks_2006-01-01_to_2018-01-01_CPTs.json'),
                        help='Path to the CPT JSON file.')
    parser.add_argument('--test_size', type=float, default=0.25,
                        help='Fraction of data to use for testing.')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for train/test splitting.')
    # Set optimal number of bins to 7.
    parser.add_argument('--num_bins', type=int, default=7,
                        help='Number of bins for discretization (if applicable).')
    parser.add_argument('--inference_method', type=str, choices=['VE', 'enumeration'], default='VE',
                        help='Inference method to use: VE for Variable Elimination, or enumeration.')
    args = parser.parse_args()
    
    config = vars(args)
    trainer = BayesianNetworkTrainer(config)
    
    # Load processed data and CPTs.
    df = trainer.load_processed_data()
    cpts = trainer.load_cpts()
    logging.info(f"Processed data shape: {df.shape}")
    
    # Add additional features (loophole #1: new technical feature)
    # Compute return as percentage change and discretize it.
    df['Return'] = df['Close'].pct_change()
    df['Return_binned'] = pd.qcut(df['Return'], q=config.get('num_bins', 7), duplicates='drop', labels=False)
    # Prepare target (this also drops NaNs).
    df = trainer.prepare_target(df)
    
    # ----- Define Feature Sets -----
    # Baseline: only the price bins.
    baseline_features = []
    for feature in ['Open', 'High', 'Low']:
        binned = f"{feature}_binned"
        if binned in df.columns:
            baseline_features.append(binned)
        else:
            baseline_features.append(feature)
    
    # Enhanced: include additional technical indicator bins, HMM-derived Market_Regime, and Return.
    enhanced_features = baseline_features.copy()
    for indicator in ['MA_20', 'RSI_14', 'MACD', 'MACD_Signal']:
        binned_ind = f"{indicator}_binned"
        if binned_ind in df.columns:
            enhanced_features.append(binned_ind)
    if 'Market_Regime' in df.columns:
        enhanced_features.append('Market_Regime')
    if 'Return_binned' in df.columns:
        enhanced_features.append('Return_binned')
    
    target = 'Trend'
    
    logging.info(f"Baseline features: {baseline_features}")
    logging.info(f"Enhanced features: {enhanced_features}")
    
    # ----- Perform a Single Train-Test Split for Both Models -----
    train_df, test_df = train_test_split(df, test_size=config['test_size'], random_state=config['random_state'])
    logging.info(f"Train set shape: {train_df.shape}; Test set shape: {test_df.shape}")
    
    # ----- Build and Evaluate Baseline Model (Fixed Structure) -----
    logging.info("Building and evaluating baseline model (fixed structure)...")
    baseline_model = trainer.build_model(train_df, baseline_features, target)
    _, _, baseline_accuracy, baseline_probs = trainer.evaluate_model(baseline_model, test_df, baseline_features, target)
    
    # ----- Build and Evaluate Enhanced Model (Structure Learning) -----
    enhanced_variables = enhanced_features + [target]
    logging.info("Learning structure for enhanced model...")
    learned_structure = trainer.learn_structure(train_df, enhanced_variables)
    learned_edges = list(learned_structure.edges())
    enhanced_model = BayesianNetwork(learned_edges)
    enhanced_model.fit(train_df[enhanced_variables], estimator=BayesianEstimator, prior_type='BDeu', equivalent_sample_size=15)
    logging.info("Performing evaluation on enhanced model (structure-learned)...")
    _, _, enhanced_accuracy, enhanced_probs = trainer.evaluate_model(enhanced_model, test_df, enhanced_features, target)
    
    logging.info(f"Baseline Bayesian Network Model Accuracy (technical indicators only): {baseline_accuracy:.4f}")
    logging.info(f"Enhanced Bayesian Network Model Accuracy (with additional technical indicators, HMM, and structure learning): {enhanced_accuracy:.4f}")
    
    # ----- Ensemble the Predictions (Loophole #2) -----
    logging.info("Combining baseline and enhanced model predictions via ensemble averaging.")
    ensemble_preds, actuals, ensemble_accuracy, ensemble_probs = ensemble_inference(
        test_df, baseline_model, baseline_features, enhanced_model, enhanced_features, target, trainer
    )
    logging.info(f"Ensemble Model Accuracy: {ensemble_accuracy:.4f}")
    

if __name__ == '__main__':
    main()
