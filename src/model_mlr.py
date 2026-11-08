"""MLR without current-year EEFFR: reproduces original notebook's M3 (driver analysis),
NOT its M7 forecast, which explicitly used current-year EEFFR.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from feature_engineering import TARGET, numeric

FEATURE_MAP = {
    'log_assets': None, 'log_assets_per_employee': None,
    'net_interest_margin':'NIMY','cost_of_funds':'INTEXPY',
    'earning_assets_ratio':'ERNASTR','loans_to_assets':'LNLSNTV',
    'securities_to_assets':('SC','ASSET'), 'premises_to_assets':('BKPREM','ASSET'),
    'intangibles_to_assets':('INTAN','ASSET'), 'deposits_to_assets':'DEPDASTR',
    'nonint_deposit_share':('DEPNIDOM','DEP'),
    'uninsured_deposit_share':('DEPUNINS','DEP'),
    'equity_to_assets':'EQV','noncurrent_loans_ratio':'NCLNLSR',
    'chargeoff_ratio':'NTLNLSR','provision_to_assets':'ELNATRY',
    'nontraditional_flag':'FLAG_NONTRADITIONAL_MODEL'
}


def build_mlr_features(df):
    x = pd.DataFrame(index=df.index)
    for name, src in FEATURE_MAP.items():
        if name == 'log_assets':
            a = numeric(df,'ASSET'); x[name] = np.log(a.where(a > 0))
        elif name == 'log_assets_per_employee':
            a = numeric(df,'ASTEMPM'); x[name] = np.log(a.where(a > 0))
        elif isinstance(src,tuple):
            a,b = (numeric(df,s) for s in src)
            x[name] = a.div(b.where(b != 0)).mul(100)
        else:
            x[name] = numeric(df,src)
    return x.replace([np.inf,-np.inf],np.nan)


class MLRModel:
    def fit(self,train):
        if 'SPECGRPDESC' not in train or 'CERT' not in train:
            raise ValueError('Original MLR requires SPECGRPDESC and CERT.')
        x = build_mlr_features(train)
        self.bounds = {c:(x[c].quantile(.01),x[c].quantile(.99))
                       for c in x if c != 'nontraditional_flag'}
        counts = train.groupby('SPECGRPDESC')['CERT'].nunique().sort_values(ascending=False)
        self.groups = counts[counts >= 30].index.tolist()
        if not self.groups:
            raise ValueError('No specialization group has >=30 unique banks.')
        self.reference = self.groups[0]
        x = self._transform(train)
        self.columns = x.columns.tolist()
        self.medians = x.median().fillna(0)
        self.model = LinearRegression().fit(x.fillna(self.medians),numeric(train,TARGET))
        return self

    def _transform(self,df):
        x = build_mlr_features(df)
        for c,(lo,hi) in self.bounds.items():
            x[c] = x[c].clip(lo,hi)
        groups = df['SPECGRPDESC'].where(df['SPECGRPDESC'].isin(self.groups),'OTHER (small groups)')
        dummies = pd.get_dummies(groups,prefix='spec',dtype=float)
        all_groups = [*self.groups,'OTHER (small groups)']
        for g in all_groups:
            col = 'spec_'+str(g)
            if col not in dummies: dummies[col] = 0.0
        dummies = dummies.drop(columns=['spec_'+str(self.reference)])
        return pd.concat([x,dummies],axis=1)

    def predict(self,df):
        x = self._transform(df).reindex(columns=self.columns,fill_value=0)
        return self.model.predict(x.fillna(self.medians))

    @property
    def feature_names(self): return self.columns


def train_model(train): return MLRModel().fit(train)
