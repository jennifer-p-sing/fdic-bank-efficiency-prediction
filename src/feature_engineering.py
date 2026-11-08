"""FDIC Group 14: CSV ingestion, shared feature definitions and leakage-safe splits.
Source: four supplied modeling notebooks and Group 14 presentation (October 2026).
"""
import numpy as np
import pandas as pd

TARGET = 'TARGET_EEFFR_NEXT_Q1'
EPSILON = 1e-8
META = {'CERT','NAME','NAMEFULL','year','quarter','REPDTE','period','SPECGRPDESC',TARGET,
        'EEFFR','current_eeffr','FLAG_NONTRADITIONAL_MODEL'}
LEAKAGE = {'P3LTOT','P9LTOT','NALTOT','NETINC','NONIXAY','ROA','ROE',
           'ASTEMPM_CHG','NIMY_CHG','NUMEMP_GROWTH'}
TREE_CORE = ['INTINCY','NIMY','INTEXPY','TFRA','BKPREM','INTAN','DEPDOM','DEPUNINS','DEPNIDOM','DEPCSBQ','COREDEP','SC','ASSET','DEPALLEX','LIAB','DEPINS','LNLSNET','DEPFOR','EQ','FREPO','CHBAL','DEPIDOM','AOA','DEP','EQV','LNLSDEPR','IDLNCORR','ERNASTR','NUMEMP','ASTEMPM','DEPDASTR','LNLSNTV','AOAR','EINTEXP','INTINC']
TREE_SECONDARY = ['NTLNLSR','NTRER','ELNATRY','OBSDIR','NAASSET','P9ASSET','P3ASSET','TRADE','UC','LNNDEPC','LNNDEPD','ORE','DEPBEFEX','NFAA','EQTOT','IDT1RWAJR','IDNCCONR','IDT1CER','LNATRESR','NPERFV','LNRESNCR','IDNCCIR','RBC1AAJ','NCLNLSR','NCRER','IDNCOTHR','RBCRWAJ','EXTRA','IGLSEC','ITAX','ELNATR','NTLNLS','EQCDIV','EQCDIVNTINC','ELNANTR','IDERNCVR']


def read_csv(path):
    df = pd.read_csv(path, low_memory=False)
    if 'year' not in df:
        if 'REPDTE' in df:
            df['year'] = pd.to_datetime(df['REPDTE'], errors='coerce').dt.year
        elif 'period' in df:
            df['year'] = pd.to_numeric(df['period'].astype(str).str.extract(r'(\d{4})')[0], errors='coerce')
    if 'year' not in df or 'CERT' not in df:
        raise ValueError('CSV must contain CERT and year (or REPDTE/period).')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    if df['year'].isna().any(): raise ValueError('Invalid year values.')
    df['year'] = df['year'].astype(int)
    if 'quarter' in df:
        q = pd.to_numeric(df['quarter'], errors='coerce')
        if q.notna().any() and (q != 1).any():
            raise ValueError('Expected Q1-only ML snapshots; non-Q1 records found.')
    if df.duplicated(['CERT','year']).any():
        raise ValueError('Duplicate CERT/year rows: input must be one bank per Q1.')
    if TARGET not in df: df[TARGET] = np.nan
    df[TARGET] = pd.to_numeric(df[TARGET], errors='coerce')
    if 'EEFFR' not in df:
        print('WARNING: EEFFR (current-year ratio) missing; MLR forecasting and naive benchmark cannot use it.')
    else:
        df['EEFFR'] = pd.to_numeric(df['EEFFR'], errors='coerce')
    return df.reset_index(drop=True)


def split_by_year(df, train_years=(2022,2023,2024), test_year=2025, oot_year=2026):
    train = df.loc[df.year.isin(train_years) & df[TARGET].notna()].copy()
    test = df.loc[(df.year == test_year) & df[TARGET].notna()].copy()
    oot = df.loc[df.year == oot_year].copy()
    if train.empty or test.empty:
        raise ValueError('Missing labeled train/test years; edit train_years/test_year to match your CSV.')
    for name, subset in [('train',train),('test',test)]:
        if not np.isfinite(subset[TARGET]).all() or (subset[TARGET] <= 0).any():
            raise ValueError(f'{name}: targets must be finite and strictly positive.')
    if not oot.empty and oot[TARGET].notna().any():
        print('WARNING: OOT contains known labels; predictions will still be generated.')
    return train.reset_index(drop=True), test.reset_index(drop=True), oot.reset_index(drop=True)


def numeric(df, name):
    return pd.to_numeric(df[name], errors='coerce') if name in df else pd.Series(np.nan, index=df.index, dtype=float)


def ratio(df, numerator, denominator, multiplier=1., epsilon=EPSILON):
    return (numeric(df,numerator)/(numeric(df,denominator)+epsilon)*multiplier).replace([np.inf,-np.inf],np.nan)


def engineer_all(df):
    """Union of engineered ratios; model functions select their own subsets.
    XGBoost-style ratios are decimal fractions; MLR-style percentage ratios
    are separately suffixed _pct to prevent unit ambiguity.
    """
    out = df.copy()
    formulas = {
        'securities_to_assets':('SC','ASSET'), 'cash_to_assets':('CHBAL','ASSET'),
        'premises_to_assets':('BKPREM','ASSET'), 'intangibles_to_assets':('INTAN','ASSET'),
        'trading_assets_to_assets':('TRADE','ASSET'), 'other_real_estate_to_assets':('ORE','ASSET'),
        'equity_to_assets':('EQTOT','ASSET'), 'core_deposit_share':('COREDEP','DEP'),
        'uninsured_deposit_share':('DEPUNINS','DEP'), 'demand_deposit_share':('DEPNIDOM','DEP'),
        'repo_funding_to_assets':('FREPO','ASSET'), 'resale_agreement_to_assets':('TFRA','ASSET'),
        'liability_to_assets':('LIAB','ASSET'), 'noncurrent_loan_ratio':('NALTOT','LNLSNET'),
    }
    for name,(a,b) in formulas.items():
        if a in df and b in df: out[name] = ratio(df,a,b)
    if 'ASSET' in df: out['log_assets'] = np.log(numeric(df,'ASSET').clip(lower=1))
    if 'ASTEMPM' in df: out['log_assets_per_employee'] = np.log(numeric(df,'ASTEMPM').where(numeric(df,'ASTEMPM')>0))
    for name,a,b in [('securities_to_assets_pct','SC','ASSET'),('premises_to_assets_pct','BKPREM','ASSET'),
                     ('intangibles_to_assets_pct','INTAN','ASSET'),('nonint_deposit_share_pct','DEPNIDOM','DEP'),
                     ('uninsured_deposit_share_pct','DEPUNINS','DEP')]:
        if a in df and b in df: out[name] = ratio(df,a,b,100)
    if 'EEFFR' in df: out['current_eeffr'] = numeric(df,'EEFFR')
    return out.replace([np.inf,-np.inf],np.nan)


def matrix(df, columns):
    """Keep training column order; absent test columns become NaN."""
    return pd.DataFrame({c:numeric(df,c) for c in columns}, index=df.index)


def eligible_numeric(df, exclude=(), min_observed=0.0):
    blocked = META | LEAKAGE | set(exclude)
    return [c for c in df.columns if c not in blocked and
            pd.to_numeric(df[c],errors='coerce').notna().mean() >= min_observed and
            pd.to_numeric(df[c],errors='coerce').nunique(dropna=True)>1]


def expanding_folds(df):
    """Chronological validation folds: 2022->2023, 2022-23->2024."""
    for valid_year in (2023,2024):
        tr = df.loc[df.year < valid_year]
        va = df.loc[df.year == valid_year]
        if len(tr) and len(va): yield tr,va,valid_year


def evaluate(actual, predicted):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    y = np.asarray(actual,float); p = np.asarray(predicted,float)
    return {'MAE':float(mean_absolute_error(y,p)),
            'RMSE':float(np.sqrt(mean_squared_error(y,p))), 'R2':float(r2_score(y,p)), 'n':len(y)}
