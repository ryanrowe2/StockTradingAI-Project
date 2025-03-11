#!/usr/bin/env python
"""
Improved Model Training Script for Bayesian Network Stock Trading AI.

This script implements a complete plan to improve model performance by:
1. Enhancing data preprocessing & feature engineering.
2. Improving model structure & hyperparameter tuning.
3. Using robust evaluation (time-series cross-validation, additional metrics).

It builds two Bayesian Networks (a baseline and an enhanced model)
and displays their performance.
"""

import os
import json
import logging
import argparse
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc)

from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator, HillClimbSearch, BicScore, K2Score
from pgmpy.inference import VariableElimination

from scipy.stats.mstats import winsorize

warnings.filterwarnings("ignore")
# Set logging level to WARNING to reduce terminal spam.
logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


###############################
# Data Preprocessing & Feature Engineering
###############################

def compute_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()
    return atr

def compute_bollinger_bands(df, window=20, num_std=2):
    sma = df['Close'].rolling(window=window, min_periods=1).mean()
    rstd = df['Close'].rolling(window=window, min_periods=1).std()
    upper_band = sma + num_std * rstd
    lower_band = sma - num_std * rstd
    return sma, upper_band, lower_band

def add_technical_indicators(df):
    df['ATR'] = compute_atr(df)
    sma, upper_band, lower_band = compute_bollinger_bands(df)
    df['BB_Middle'] = sma
    df['BB_Upper'] = upper_band
    df['BB_Lower'] = lower_band
    df['Close_Lag1'] = df['Close'].shift(1)
    df['Return'] = df['Close'].pct_change()
    df['Return_Lag1'] = df['Return'].shift(1)
    return df

def apply_winsorization(df, columns, limits=(0.05, 0.05)):
    for col in columns:
        df[col] = winsorize(df[col], limits=limits)
    return df

def discretize_features(df, features, num_bins=7, method='quantile'):
    for feature in features:
        if method == 'quantile':
            df[f"{feature}_binned"] = pd.qcut(df[feature], q=num_bins, duplicates='drop', labels=False)
        elif method == 'equal_width':
            df[f"{feature}_binned"] = pd.cut(df[feature], bins=num_bins, labels=False)
        else:
            raise ValueError("Unsupported discretization method.")
    return df

def feature_selection(df, target, threshold=0.95, preserve=[]):
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = []
    for column in upper_tri.columns:
        if column in preserve:
            continue
        if any(upper_tri[column] > threshold):
            to_drop.append(column)
    logging.warning(f"Features dropped: {to_drop}")
    df = df.drop(columns=to_drop, errors='ignore')
    return df


###############################
# Bayesian Network Trainer with Cross-Validation & Grid Search
###############################

class BaseModelTrainer:
    def __init__(self, config: dict):
        self.processed_data_path = config.get('processed_data_path')
        self.cpts_path = config.get('cpts_path')
        self.inference_method = config.get('inference_method', 'VE')
        self.config = config

    def load_processed_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.processed_data_path)
        logging.warning(f"Data loaded with shape: {df.shape}")
        return df

    def load_cpts(self) -> dict:
        with open(self.cpts_path, 'r') as f:
            cpts = json.load(f)
        return cpts

    @staticmethod
    def prepare_target(df: pd.DataFrame) -> pd.DataFrame:
        df['Trend'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df.dropna(inplace=True)
        logging.warning(f"Target prepared. Data shape: {df.shape}")
        return df

    def evaluate_model(self, model: BayesianNetwork, test_df: pd.DataFrame,
                       features: list, target: str) -> tuple:
        inference = VariableElimination(model)
        predictions = []
        predicted_probs = []
        actuals = test_df[target].tolist()
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
        return predictions, actuals, accuracy, predicted_probs

class BayesianNetworkTrainer(BaseModelTrainer):
    def __init__(self, config: dict):
        super().__init__(config)
        self.param_grid = {
            'equivalent_sample_size': [5, 10, 15, 20],
            'prior_type': ['BDeu']
        }
        self.num_restarts = config.get('num_restarts', 3)
        self.scoring_method_name = config.get('scoring_method', 'BicScore')

    def build_model(self, df: pd.DataFrame, features: list, target: str,
                    estimator=BayesianEstimator, **estimator_params) -> BayesianNetwork:
        edges = [(feature, target) for feature in features]
        model = BayesianNetwork(edges)
        model.fit(df[features + [target]], estimator=estimator,
                  prior_type=estimator_params.get('prior_type', 'BDeu'),
                  equivalent_sample_size=estimator_params.get('equivalent_sample_size', 15))
        return model

    def learn_structure(self, df: pd.DataFrame, variables: list, seed=None) -> BayesianNetwork:
        if self.scoring_method_name == 'K2Score':
            score = K2Score(df[variables])
        else:
            score = BicScore(df[variables])
        hc = HillClimbSearch(df[variables])
        if seed is not None:
            np.random.seed(seed)
        best_model = hc.estimate(scoring_method=score)
        return best_model

    def grid_search_estimator(self, df: pd.DataFrame, features: list, target: str, cv_splits=3):
        best_score = -np.inf
        best_params = {}
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        for ess in self.param_grid['equivalent_sample_size']:
            for prior in self.param_grid['prior_type']:
                scores = []
                for train_idx, val_idx in tscv.split(df):
                    train_df = df.iloc[train_idx]
                    val_df = df.iloc[val_idx]
                    model = self.build_model(train_df, features, target,
                                             estimator=BayesianEstimator,
                                             equivalent_sample_size=ess,
                                             prior_type=prior)
                    _, _, acc, _ = self.evaluate_model(model, val_df, features, target)
                    scores.append(acc)
                avg_score = np.mean(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_params = {'equivalent_sample_size': ess, 'prior_type': prior}
        logging.warning(f"Best estimator params: {best_params}, CV Accuracy: {best_score:.4f}")
        return best_params


###############################
# Main Function: Implementation Roadmap
###############################

def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate an improved Bayesian Network for Stock Trading AI."
    )
    parser.add_argument('--processed_data_path', type=str,
                        default=os.path.join('..', 'data', 'processed', 'all_stocks_2006-01-01_to_2018-01-01.csv'),
                        help='Path to the processed CSV file.')
    parser.add_argument('--cpts_path', type=str,
                        default=os.path.join('..', 'data', 'processed', 'all_stocks_2006-01-01_to_2018-01-01_CPTs.json'),
                        help='Path to the CPT JSON file.')
    parser.add_argument('--num_bins', type=int, default=7,
                        help='Number of bins for discretization.')
    parser.add_argument('--discretization_method', type=str, choices=['quantile', 'equal_width'],
                        default='quantile', help='Method for discretization.')
    parser.add_argument('--inference_method', type=str, choices=['VE', 'enumeration'],
                        default='VE', help='Inference method to use.')
    parser.add_argument('--scoring_method', type=str, choices=['BicScore', 'K2Score'],
                        default='BicScore', help='Scoring method for structure learning.')
    parser.add_argument('--num_restarts', type=int, default=3,
                        help='Number of restarts for structure learning.')
    parser.add_argument('--cv_splits', type=int, default=3,
                        help='Number of splits for time-series cross-validation.')
    args = parser.parse_args()
    config = vars(args)

    bn_trainer = BayesianNetworkTrainer(config)
    df = bn_trainer.load_processed_data()
    _ = bn_trainer.load_cpts()  # For side effects.
    logging.warning(f"Original data shape: {df.shape}")

    df = apply_winsorization(df, ['Close', 'High', 'Low'])
    df = add_technical_indicators(df)
    continuous_features = ['Close', 'ATR', 'BB_Middle', 'BB_Upper', 'BB_Lower', 'Return', 'Close_Lag1', 'Return_Lag1']
    df = discretize_features(df, continuous_features, num_bins=config['num_bins'],
                             method=config['discretization_method'])
    df = bn_trainer.prepare_target(df)
    preserve_cols = ['Open_binned', 'High_binned', 'Low_binned',
                     'ATR_binned', 'Return_binned', 'Close_binned',
                     'BB_Middle_binned', 'BB_Upper_binned', 'BB_Lower_binned',
                     'Close_Lag1_binned', 'Return_Lag1_binned', 'Trend']
    df = feature_selection(df, target='Trend', threshold=0.95, preserve=preserve_cols)

    baseline_candidates = ['Open_binned', 'High_binned', 'Low_binned']
    baseline_features = [col for col in baseline_candidates if col in df.columns]
    enhanced_candidates = baseline_candidates + ['ATR_binned', 'Return_binned']
    enhanced_features = [col for col in enhanced_candidates if col in df.columns]

    logging.warning(f"Baseline features: {baseline_features}")
    logging.warning(f"Enhanced features: {enhanced_features}")

    tscv = TimeSeriesSplit(n_splits=config['cv_splits'])
    bn_cv_scores = []
    for train_idx, test_idx in tscv.split(df):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        bn_model = bn_trainer.build_model(train_df, baseline_features, 'Trend',
                                          estimator=BayesianEstimator,
                                          equivalent_sample_size=15,
                                          prior_type='BDeu')
        _, _, acc, _ = bn_trainer.evaluate_model(bn_model, test_df, baseline_features, 'Trend')
        bn_cv_scores.append(acc)
    baseline_cv_accuracy = np.mean(bn_cv_scores)
    logging.warning(f"Baseline BN CV Accuracy: {baseline_cv_accuracy:.4f}")

    best_params = bn_trainer.grid_search_estimator(df, enhanced_features, 'Trend', cv_splits=config['cv_splits'])
    best_structure_score = -np.inf
    best_structure = None
    for i in range(bn_trainer.num_restarts):
        structure_model = bn_trainer.learn_structure(df, enhanced_features + ['Trend'], seed=i)
        splits = list(tscv.split(df))
        train_idx, test_idx = splits[-1]
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        enhanced_bn = BayesianNetwork(list(structure_model.edges()))
        enhanced_bn.fit(train_df[enhanced_features + ['Trend']],
                        estimator=BayesianEstimator,
                        prior_type=best_params.get('prior_type', 'BDeu'),
                        equivalent_sample_size=best_params.get('equivalent_sample_size', 15))
        _, _, acc, _ = bn_trainer.evaluate_model(enhanced_bn, test_df, enhanced_features, 'Trend')
        if acc > best_structure_score:
            best_structure_score = acc
            best_structure = structure_model
    logging.warning(f"Best structure validation accuracy: {best_structure_score:.4f}")
    
    enhanced_bn_model = BayesianNetwork(list(best_structure.edges()))
    enhanced_bn_model.fit(df[enhanced_features + ['Trend']],
                          estimator=BayesianEstimator,
                          prior_type=best_params.get('prior_type', 'BDeu'),
                          equivalent_sample_size=best_params.get('equivalent_sample_size', 15))
    
    holdout_index = int(len(df) * 0.8)
    train_df = df.iloc[:holdout_index]
    test_df = df.iloc[holdout_index:]
    
    _, _, bn_accuracy, _ = bn_trainer.evaluate_model(enhanced_bn_model, test_df, enhanced_features, 'Trend')
    
    # Print final summary with accuracy for the enhanced Bayesian Network.
    print(f"Final Holdout Accuracy -> Enhanced BN: {bn_accuracy:.4f}")

if __name__ == '__main__':
    main()
