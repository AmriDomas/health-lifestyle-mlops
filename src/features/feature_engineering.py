# src/features/feature_engineering.py
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import joblib

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.numeric_features = ['age', 'bmi', 'daily_steps', 'sleep_hours', 'water_intake_l',
                               'calories_consumed', 'smoker', 'alcohol', 'resting_hr', 'systolic_bp',
                               'diastolic_bp', 'family_history']
        self.categorical_features = ['gender']
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Feature engineering
        X['bmi_category'] = pd.cut(X['bmi'], 
                                  bins=[0, 18.5, 25, 30, np.inf],
                                  labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        
        X['age_group'] = pd.cut(X['age'],
                               bins=[0, 30, 45, 60, np.inf],
                               labels=['Young', 'Adult', 'Middle-aged', 'Senior'])
        
        X['activity_level'] = pd.cut(X['daily_steps'], [0, 5000, 7500, 10000, float('inf')],
                                   labels=['sedentary', 'Low', 'moderate', 'active'])

        X['bp_risk'] = ((X['systolic_bp'] > 140) | (X['diastolic_bp'] > 00)).astype(int)

        X['lifestyle_risk'] = (X['smoker'] + X['alcohol'] + 
                              (X['sleep_hours'] < 6).astype(int) +
                              (X['water_intake_l'] < 2).astype(int))
        
        X['fitness_score'] = (X['daily_steps']/10000 + 
                             (X['resting_hr'] < 70).astype(int) + 
                             (X['bmi'] < 25).astype(int))
        
        X['age_bmi'] = X['age'] * X['bmi']

        X['stress_indicator'] = X['resting_hr'] * X['systolic_bp'] / 100

        X['hr_category'] = pd.cut(X['resting_hr'], 
                           bins=[0, 60, 100, float('inf')], 
                           labels=['low', 'normal', 'high'])
        
        X['health_score'] = ((X['daily_steps'] / 10000) +
                            (X['sleep_hours'] / 8) +
                            (X['water_intake_l'] / 2) -
                            (X['bmi'] / 30) -
                            (X['resting_hr'] / 100))

        return X

def create_feature_pipeline():
    """Membuat pipeline feature engineering lengkap"""
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, ['age', 'bmi', 'daily_steps', 'sleep_hours', 'water_intake_l',
                               'calories_consumed', 'smoker', 'alcohol', 'resting_hr', 'systolic_bp',
                               'diastolic_bp', 'family_history', 'bp_risk', 'lifestyle_risk',
                               'fitness_score', 'age_bmi', 'stress_indicator', 'health_score']),
            ('cat', categorical_transformer, ['gender', 'bmi_category', 'age_group', 'activity_level', 
                                              'hr_category'])
        ])
    
    return Pipeline(steps=[
        ('feature_engineer', FeatureEngineer()),
        ('preprocessor', preprocessor)
    ])

