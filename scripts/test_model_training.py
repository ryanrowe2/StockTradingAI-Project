import os
import json
import pandas as pd
import numpy as np
import pytest
from model_training import BaseModelTrainer, BayesianNetworkTrainer
from pgmpy.models import BayesianNetwork

# A sample configuration for testing.
TEST_CONFIG = {
    'processed_data_path': 'dummy_processed.csv',
    'cpts_path': 'dummy_cpts.json',
    'test_size': 0.5,  # use 50% for testing to make small datasets work
    'random_state': 0,
    'num_bins': 5,
    'inference_method': 'VE'
}

# Create a dummy processed dataframe for testing.
@pytest.fixture
def dummy_df(tmp_path):
    # Create a small dummy DataFrame with a "Close" column.
    data = {
        'Close': [100, 105, 102, 108],
        'Open': [99, 106, 101, 107],
        'High': [102, 107, 103, 110],
        'Low': [98, 104, 100, 106]
    }
    df = pd.DataFrame(data)
    # Save to a temporary CSV file.
    file_path = tmp_path / "dummy_processed.csv"
    df.to_csv(file_path, index=False)
    return file_path, df

# Create a dummy CPT file.
@pytest.fixture
def dummy_cpts(tmp_path):
    # For simplicity, create a dummy CPT dict.
    cpts = {
        "Close": {"0": 0.5, "1": 0.5},
        "Open": {"0": 0.6, "1": 0.4}
    }
    file_path = tmp_path / "dummy_cpts.json"
    with open(file_path, "w") as f:
        json.dump(cpts, f)
    return file_path, cpts

# Fixture to create a dummy configuration that points to our temporary files.
@pytest.fixture
def trainer_config(dummy_df, dummy_cpts):
    file_path, _ = dummy_df
    cpts_path, _ = dummy_cpts
    config = TEST_CONFIG.copy()
    config['processed_data_path'] = str(file_path)
    config['cpts_path'] = str(cpts_path)
    return config

def test_prepare_target():
    # Create a small dummy DataFrame.
    df = pd.DataFrame({'Close': [100, 105, 102, 108]})
    # Call prepare_target. Expect last row dropped, so shape becomes (3,2) with a new column Trend.
    df_prepared = BaseModelTrainer.prepare_target(df)
    # Check that Trend column exists.
    assert 'Trend' in df_prepared.columns
    # Since we had 4 rows and the last row is dropped, expect 3 rows.
    assert df_prepared.shape[0] == 3
    # Check computed Trend values:
    # For row 0: 105 > 100 => Trend = 1
    # For row 1: 102 < 105 => Trend = 0
    # For row 2: 108 > 102 => Trend = 1
    expected_trend = [1, 0, 1]
    np.testing.assert_array_equal(df_prepared['Trend'].values, expected_trend)

def test_load_processed_data(tmp_path):
    # Create a dummy CSV file.
    dummy_data = pd.DataFrame({
        'Close': [100, 105, 102],
        'Open': [99, 106, 101],
        'High': [102, 107, 103],
        'Low': [98, 104, 100]
    })
    file_path = tmp_path / "test_data.csv"
    dummy_data.to_csv(file_path, index=False)
    # Create a minimal config dictionary.
    config = {'processed_data_path': str(file_path)}
    trainer = BaseModelTrainer(config)
    df = trainer.load_processed_data()
    pd.testing.assert_frame_equal(df, dummy_data)

def test_load_cpts(tmp_path):
    # Create a dummy CPT JSON file.
    cpts = {"Close": {"0": 0.5, "1": 0.5}}
    file_path = tmp_path / "test_cpts.json"
    with open(file_path, "w") as f:
        json.dump(cpts, f)
    config = {'cpts_path': str(file_path)}
    trainer = BaseModelTrainer(config)
    loaded_cpts = trainer.load_cpts()
    assert loaded_cpts == cpts

def test_build_model(trainer_config, dummy_df):
    # Use the dummy DataFrame from fixture.
    _, df = dummy_df
    df = BaseModelTrainer.prepare_target(df)
    # For simplicity, we use the 'Open' column as feature.
    features = ['Open']
    # Create a dummy column for binned feature (simulate that discretization was done)
    df['Open_binned'] = (df['Open'] > df['Open'].median()).astype(int)
    target = 'Trend'
    trainer = BayesianNetworkTrainer(trainer_config)
    model = trainer.build_model(df, ['Open_binned'], target)
    # Check that model is an instance of BayesianNetwork.
    assert isinstance(model, BayesianNetwork)
    # Check that expected nodes are in the network.
    nodes = model.nodes()
    assert 'Open_binned' in nodes
    assert target in nodes
    # Also check that CPDs have been learned.
    for node in nodes:
        cpd = model.get_cpds(node)
        assert cpd is not None

def test_evaluate_model(trainer_config, dummy_df):
    # Use dummy dataframe, add necessary columns.
    _, df = dummy_df
    df = BaseModelTrainer.prepare_target(df)
    # Use 'Open' as the feature and create a binned version.
    features = ['Open']
    df['Open_binned'] = (df['Open'] > df['Open'].median()).astype(int)
    target = 'Trend'
    
    trainer = BayesianNetworkTrainer(trainer_config)
    model = trainer.build_model(df, ['Open_binned'], target)
    
    # Evaluate model using VE inference.
    predictions, actuals, accuracy = trainer.evaluate_model(model, df, ['Open_binned'], target)
    # Check that predictions and actuals are lists of equal length.
    assert isinstance(predictions, list)
    assert isinstance(actuals, list)
    assert len(predictions) == len(actuals)
    # Check that accuracy is between 0 and 1.
    assert 0.0 <= accuracy <= 1.0

def test_evaluate_model_enumeration(trainer_config, dummy_df):
    # Similar to above, use the enumeration evaluation.
    _, df = dummy_df
    df = BaseModelTrainer.prepare_target(df)
    features = ['Open']
    df['Open_binned'] = (df['Open'] > df['Open'].median()).astype(int)
    target = 'Trend'
    
    trainer = BayesianNetworkTrainer(trainer_config)
    model = trainer.build_model(df, ['Open_binned'], target)
    
    predictions, actuals, accuracy = trainer.evaluate_model_enumeration(model, df, ['Open_binned'], target)
    assert isinstance(predictions, list)
    assert isinstance(actuals, list)
    assert len(predictions) == len(actuals)
    assert 0.0 <= accuracy <= 1.0

if __name__ == '__main__':
    pytest.main()
