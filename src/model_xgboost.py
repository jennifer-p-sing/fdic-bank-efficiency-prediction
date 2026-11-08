"""XGBoost refactor of original Group 14 notebook, with current-year EEFFR excluded.
Uses original explicit drops, numeric-column selection, native missing values,
and gain-based feature pruning; no test-set selection.
"""
import numpy as np
import pandas as pd
from feature_engineering import TARGET, engineer_all, matrix

RAW_DROP = {'ASSET','NUMEMP','DEP','LNLSNET','SC','CHBAL','BKPREM','INTAN','TRADE',
            'ORE','AOA','EQ','EQTOT','COREDEP','DEPUNINS','DEPNIDOM','FREPO',
            'TFRA','LIAB','NALTOT'}
CORR_DROP = {'IDT1CER','RBC1AAJ','DEPCSBQ','DEPIDOM','DEPBEFEX','DEPINS'}
UNSTABLE_DROP = {'P3LTOT','P9LTOT','noncurrent_loan_ratio'}
ID_COLS = {'CERT','NAME','NAMEFULL','period','year','quarter','REPDTE','SPECGRPDESC'}
# Shared engineer_all adds these MLR-only engineered fields, which did not
# exist in the original XGBoost feature-engineering function.
MLR_ONLY = {'log_assets_per_employee','securities_to_assets_pct',
            'premises_to_assets_pct','intangibles_to_assets_pct',
            'nonint_deposit_share_pct','uninsured_deposit_share_pct','current_eeffr'}

class XGBoostModel:
    def _model(self):
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=500,max_depth=6,learning_rate=.03,
                            subsample=.8,colsample_bytree=1.0,reg_alpha=.1,
                            objective='reg:pseudohubererror',huber_slope=31.89,
                            random_state=321,n_jobs=-1,importance_type='gain')

    def fit(self,train):
        d = engineer_all(train)
        # Match original select_dtypes(include=['number']) rather than coercing
        # object columns into additional predictors. Explicitly reject current ER.
        excluded = RAW_DROP | CORR_DROP | UNSTABLE_DROP | ID_COLS | MLR_ONLY | {TARGET,'EEFFR'}
        cols = [c for c in d.select_dtypes(include=['number']).columns
                if c not in excluded and d[c].notna().any() and d[c].nunique(dropna=True)>1]
        # Original notebook converts this boolean specialization indicator to int.
        if 'FLAG_NONTRADITIONAL_MODEL' in d and 'FLAG_NONTRADITIONAL_MODEL' not in cols:
            flag = pd.to_numeric(d['FLAG_NONTRADITIONAL_MODEL'],errors='coerce')
            if flag.notna().any() and flag.nunique(dropna=True)>1:
                d['FLAG_NONTRADITIONAL_MODEL'] = flag
                cols.append('FLAG_NONTRADITIONAL_MODEL')
        if not cols: raise ValueError('No eligible XGBoost features.')
        self.initial_features = cols
        x = matrix(d,cols)
        y = pd.to_numeric(train[TARGET],errors='raise').to_numpy(float)
        first = self._model().fit(x,y)
        importance = first.feature_importances_
        self.columns = [c for c,gain in zip(cols,importance) if gain > .001]
        if not self.columns: raise ValueError('No features passed gain > 0.001.')
        self.model = self._model().fit(matrix(d,self.columns),y)
        return self

    def predict(self,df):
        return self.model.predict(matrix(engineer_all(df),self.columns))

    @property
    def feature_names(self): return self.columns


def train_model(train): return XGBoostModel().fit(train)
