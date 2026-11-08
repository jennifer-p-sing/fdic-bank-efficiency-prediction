"""Group 14 Decision Tree: domain core/secondary features, train-only quality filter,
near-duplicate removal, inner bank-group permutation ranking, final top 10 tree.
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupShuffleSplit
from feature_engineering import TARGET, TREE_CORE, TREE_SECONDARY, matrix

class DecisionTreeModel:
    def fit(self,train):
        candidates = [c for c in dict.fromkeys(TREE_CORE+TREE_SECONDARY) if c in train and
                      pd.to_numeric(train[c],errors='coerce').notna().mean() >= .70 and
                      pd.to_numeric(train[c],errors='coerce').nunique(dropna=True)>1]
        if not candidates: raise ValueError('No usable Decision Tree predictors in CSV.')
        x = matrix(train,candidates)
        corr = x.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape),1).astype(bool))
        allowed = [c for c in candidates if c not in upper.columns[(upper>.9999).any(axis=0)]]
        if len(train) >= 12 and train.CERT.nunique() >= 4:
            a,b = next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=42).split(train,groups=train.CERT))
            ranker = make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),
                                   DecisionTreeRegressor(max_depth=6,min_samples_leaf=40,random_state=42))
            ranker.fit(x.iloc[a][allowed],np.log1p(train[TARGET].iloc[a]))
            imp = permutation_importance(ranker,x.iloc[b][allowed],np.log1p(train[TARGET].iloc[b]),
                                         scoring='neg_mean_absolute_error',n_repeats=2,random_state=42,n_jobs=1)
            ranking = pd.Series(imp.importances_mean,index=allowed).sort_values(ascending=False)
            self.columns = ranking.head(10).index.tolist()
        else: self.columns = allowed[:10]
        self.model = make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),
                                   DecisionTreeRegressor(max_depth=10,min_samples_leaf=100,
                                                         min_samples_split=2,random_state=42))
        self.model.fit(matrix(train,self.columns),np.log1p(train[TARGET]))
        return self
    def predict(self,df): return np.maximum(0,np.expm1(self.model.predict(matrix(df,self.columns))))
    @property
    def feature_names(self): return self.columns

def train_model(train): return DecisionTreeModel().fit(train)
