#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import unittest
from unittest.mock import patch, MagicMock
import tkinter as tk
import pandas as pd
import numpy as np
from app import (
    read_metric_data, update_statistics, update_column_names, 
    validate_input, predict_yield, reset_fields, update_graph
)

class MockLabel:
    def __init__(self):
        self.text = ""
    
    def config(self, text=None, **kwargs):
        if text is not None:
            self.text = text

    def cget(self, param):
        if param == "text":
            return self.text

class TestMetricFunctions(unittest.TestCase):

    def setUp(self):
        self.root = MagicMock()
        self.mean_value_label = MockLabel()
        self.median_value_label = MockLabel()
        self.mode_value_label = MockLabel()
        self.min_value_label = MockLabel()
        self.max_value_label = MockLabel()
        self.std_value_label = MockLabel()
        self.var_value_label = MockLabel()
        self.q1_value_label = MockLabel()
        self.q2_value_label = MockLabel()
        self.q3_value_label = MockLabel()
        self.min_column_value_label = MockLabel()
        self.max_column_value_label = MockLabel()
        self.result_label = MockLabel()

    @patch('app.metric_to_file', {'Test Metric': 'test_data.csv'})
    @patch('app.resource_path')
    def test_read_metric_data(self, mock_resource_path):
        mock_resource_path.return_value = 'test_data.csv'
        pd.DataFrame({'Country Name': ['USA'], '2000': [100], '2001': [200]}).to_csv('test_data.csv', index=False)
        result = read_metric_data('Test Metric')
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (1, 3))

    @patch('app.mean_value_label', new_callable=MockLabel)
    @patch('app.median_value_label', new_callable=MockLabel)
    @patch('app.mode_value_label', new_callable=MockLabel)
    @patch('app.min_value_label', new_callable=MockLabel)
    @patch('app.max_value_label', new_callable=MockLabel)
    @patch('app.std_value_label', new_callable=MockLabel)
    @patch('app.var_value_label', new_callable=MockLabel)
    @patch('app.q1_value_label', new_callable=MockLabel)
    @patch('app.q2_value_label', new_callable=MockLabel)
    @patch('app.q3_value_label', new_callable=MockLabel)
    @patch('app.recent_value_label', new_callable=MockLabel)
    def test_update_statistics(self, recent_mock, q3_mock, q2_mock, q1_mock, var_mock, std_mock, max_mock, min_mock, mode_mock, median_mock, mean_mock):
        data = pd.DataFrame({'Value': [1, 2, 3, 4, 5]}, index=pd.date_range(start='2000-01-01', periods=5, freq='Y'))
        update_statistics(data)

        # Check mean
        self.assertEqual(mean_mock.text, " 3.000")
        # Check median
        self.assertEqual(median_mock.text, " 3.000")
        # Check mode
        self.assertEqual(mode_mock.text, " 1.000")
        # Check min
        self.assertEqual(min_mock.text, " 1.000")
        # Check max
        self.assertEqual(max_mock.text, " 5.000")
        # Check standard deviation
        self.assertEqual(std_mock.text, " 1.581")
        # Check variance
        self.assertEqual(var_mock.text, " 2.500")
        # Check quartiles
        self.assertEqual(q1_mock.text, " 2.00")
        self.assertEqual(q2_mock.text, " 3.00")
        self.assertEqual(q3_mock.text, " 4.00")
        # Check most recent year and value
        self.assertEqual(recent_mock.text, "Most Recent Year: 2004, Value: 5.000")

    

    @patch('app.min_column_value_label', new_callable=MockLabel)
    @patch('app.max_column_value_label', new_callable=MockLabel)
    def test_update_column_names(self, mock_max, mock_min):
        data = pd.DataFrame({'Value': [1, 2, 3]}, index=pd.date_range('2000-01-01', periods=3, freq='Y'))
        update_column_names(data)
        self.assertEqual(mock_min.text, "2000")
        self.assertEqual(mock_max.text, "2002")

    def test_validate_input(self):
        entries = {
            'Test1': MagicMock(get=lambda: "10"),
            'Test2': MagicMock(get=lambda: ""),
            'Test3': MagicMock(get=lambda: "abc")
        }
        non_numeric, empty_fields = validate_input(entries)
        self.assertEqual(non_numeric, ['Test3'])
        self.assertEqual(empty_fields, ['Test2'])

    @patch('app.model.predict')
    @patch('app.result_label', new_callable=MockLabel)
    def test_predict_yield(self, mock_result_label, mock_predict):
        mock_predict.return_value = np.array([1000])
        entries = {key: MagicMock(get=lambda: "100") for key in [
            'Agriculture, forestry, and fishing, value added',
            'Agricultural land.',
            'Forest area',
            'Rural population',
            'Agricultural land',
            'Arable land',
            'Agricultural methane emissions',
            'Land under cereal production',
            'Average precipitation in depth',
            'Land area',
            'Arable land.'
        ]}
        predict_yield(entries)
        self.assertTrue(mock_predict.called)
        self.assertIn("Prediction:", mock_result_label.text)

    @patch('app.result_label', new_callable=MockLabel)
    def test_reset_fields(self, mock_result_label):
        entries = {
            'Test1': MagicMock(delete=MagicMock()),
            'Test2': MagicMock(delete=MagicMock())
        }
        reset_fields(entries)
        for entry in entries.values():
            entry.delete.assert_called_with(0, tk.END)
        self.assertEqual(mock_result_label.text, "")



class TestIntegration(unittest.TestCase):
    @patch('app.model.predict')
    @patch('app.result_label', new_callable=MockLabel)
    def test_prediction_integration(self, mock_result_label, mock_predict):
        mock_predict.return_value = np.array([1000])
        
        entries = {key: MagicMock(get=lambda: "100") for key in [
            'Agriculture, forestry, and fishing, value added',
            'Agricultural land.',
            'Forest area',
            'Rural population',
            'Agricultural land',
            'Arable land',
            'Agricultural methane emissions',
            'Land under cereal production',
            'Average precipitation in depth',
            'Land area',
            'Arable land.'
        ]}
        
        predict_yield(entries)
        
        self.assertTrue(mock_predict.called)
        self.assertIn("Prediction:", mock_result_label.text)

if __name__ == '__main__':
    unittest.main()

