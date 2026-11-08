"""Group 14 Random Forest: three engineered ratios, specialization encoding,
mutual-information top-25 selection, log1p target and final 200-tree configuration.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from feature_engineering import TARGET, META, LEAKAGE, engineer_all, matrix

class RandomForestModel:
    def _features(self,df):
        d = engineer_all(df)
        if 'SPECGRPDESC' in d:
            d['SPECGRPDESC_encoded'] = d['SPECGRPDESC'].astype(str).map(self.group_mapping).fillna(-1)
        return d
    def fit(self,train):
        self.group_mapping = ({g:i for i,g in enumerate(sorted(train['SPECGRPDESC'].dropna().astype(str).unique()))}
                              if 'SPECGRPDESC' in train else {})
        d = self._features(train)
        excluded = META | LEAKAGE | {'P3LTOT','P9LTOT','NALTOT','noncurrent_loan_ratio',
                                      'securities_to_assets_pct','premises_to_assets_pct',
                                      'intangibles_to_assets_pct','nonint_deposit_share_pct',
                                      'uninsured_deposit_share_pct','current_eeffr'}
        candidates = [c for c in d if c not in excluded and
                      pd.to_numeric(d[c],errors='coerce').notna().any() and
                      pd.to_numeric(d[c],errors='coerce').nunique(dropna=True)>1]
        if not candidates: raise ValueError('No usable Random Forest predictors.')
        x = matrix(d,candidates)
        self.rank_imputer = SimpleImputer(strategy='median',keep_empty_features=True)
        xi = self.rank_imputer.fit_transform(x)
        scores = mutual_info_regression(xi,np.log1p(train[TARGET]),random_state=42,n_neighbors=3)
        self.columns = pd.Series(scores,index=candidates).sort_values(ascending=False).head(25).index.tolist()
        base = make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),
                             RandomForestRegressor(n_estimators=200,min_samples_leaf=5,
                                                   min_samples_split=2,max_features=.5,
                                                   bootstrap=True,random_state=42,n_jobs=-1))
        self.model = TransformedTargetRegressor(regressor=base,func=np.log1p,inverse_func=np.expm1)
        self.model.fit(matrix(d,self.columns),train[TARGET].to_numpy(float))
        return self
    def predict(self,df): return np.maximum(0,self.model.predict(matrix(self._features(df),self.columns)))
    @property
    def feature_names(self): return self.columns

def train_model(train): return RandomForestModel().fit(train)
