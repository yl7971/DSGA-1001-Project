def surgery_preprocess():
    import pandas as pd
    import seaborn as sns
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    spirometry = pd.read_sas('SPX_G.XPT')
    demographics = pd.read_sas('DEMO_G.XPT')
    body = pd.read_sas('BMX_G.XPT')

    # 1 is male, 2 is female for sex
    subset = spirometry[['SPXNFVC', 'SPXNFEV1', 'SEQN']]
    demo_sub = demographics[['RIDAGEYR', 'RIAGENDR', 'SEQN']]
    body_sub = body[['BMXHT', 'BMXWT', 'SEQN']]
    subset = pd.merge(subset, demo_sub, 'left', 'SEQN')
    subset = pd.merge(subset, body_sub, 'left', 'SEQN')
    subset.rename(columns = {'SPXNFVC': 'FVC', 'SPXNFEV1': 'FEV1',
                            'RIDAGEYR': 'Age', 'RIAGENDR': 'Sex', 
                            'BMXHT': 'Height', 'BMXWT': 'Weight'}, inplace=True)
    subset = subset.dropna()
    subset
    
    
    estimates = subset.groupby('Age')[['FVC', 'FEV1']].mean()
    estimates.rename(columns={'FVC':'mean_FVC', 'FEV1':'mean_FEV1'}, inplace=True)
    
    
    
    surgery = pd.read_csv('ThoracicSurgery.csv', index_col = 0)
    surgery.rename(columns = {'DNG': 'Diagnosis', 'PRE4': 'FVC',
                            'PRE5': 'FEV1', 'PRE6': 'Performance',
                            'PRE7': 'Pain', 'PRE8': 'Haemoptysis',
                            'PRE9': 'Dyspnoea', 'PRE10': 'Cough',
                            'PRE11': 'Weakness', 'PRE14': 'Tumor_size',
                            'PRE17': 'Type2_diabetes', 'PRE19': 'MI_6months',
                            'PRE25': 'PAD', 'PRE30': 'Smoking',
                            'PRE32': 'Asthma', 'AGE': 'Age'}, inplace=True)
    
    
    
    for i in range(80,int(surgery.Age.max())+1):
        estimates = estimates.append(pd.Series(data=dict({'mean_FVC':estimates.loc[79.0,'mean_FVC'],
                                                      'mean_FEV1':estimates.loc[79.0,'mean_FEV1']}), 
                                           name=float(i)))
    
    
    
    surgery['FEV1/FVC'] = surgery['FEV1'] / surgery['FVC']
    surgery['expected_FVC'] = estimates.loc[surgery.Age, 'mean_FVC'].values / 1000
    surgery['expected_FEV1'] = estimates.loc[surgery.Age, 'mean_FEV1'].values / 1000
    surgery['expected_FEV1/FVC'] = surgery['expected_FEV1'] / surgery['expected_FVC']
    surgery['FVC_deficit'] = surgery['expected_FVC']-surgery['FVC']
    surgery['FEV1_deficit'] = surgery['expected_FEV1']-surgery['FEV1']
    surgery['FEV1/FVC_deficit'] = surgery['expected_FEV1/FVC']-surgery['FEV1/FVC']
    surgery['FEV1^2'] = surgery['FEV1']**2
    surgery['FVC^2'] = surgery['FVC']**2


    surgery['Age*FVC'] = surgery['Age']*surgery['FVC']
    surgery['Age*FEV1'] = surgery['Age']*surgery['FEV1']
    surgery['FVC*FEV1'] = surgery['FVC']*surgery['FEV1']
    surgery['FVC^2*FEV1'] = surgery['FEV1']*(surgery['FVC']**2)
    surgery['FVC*FEV1^2'] = surgery['FVC']*(surgery['FEV1']**2)
    
    surgery.drop(columns=['expected_FVC', 'expected_FEV1', 'expected_FEV1/FVC'], inplace=True)


    numeric_cols = surgery.select_dtypes('number').columns
    
    surgery.loc[:,surgery.dtypes=='object']= surgery.loc[:,surgery.dtypes=='object'].apply(lambda s: (s.str.replace("\'", "").str.replace('b', '')))
    surgery.replace(to_replace = ['F', 'T'], value=[0,1], inplace=True)

    ord_enc = OrdinalEncoder()
    bin_enc = LabelBinarizer()
    ord_cols = ['Performance', 'Tumor_size']

    for col in ord_cols:
        surgery[col] = surgery[col].str.strip().str[-1].astype(int)
    #surgery[col] = surgery[col]
    surgery = surgery[~(surgery.DGN.isin(['DGN1', 'DGN6', 'DGN8']))]
    surgery = pd.get_dummies(surgery, prefix=[''], columns=['DGN'])
    #surgery['Tumor_size'] = surgery['Tumor_size']+1 # Since a tumor size of 0 is unintuitive
    
    X_train, X_test, y_train, y_test = train_test_split(surgery.drop('Risk1Yr', axis=1), 
                                                    surgery['Risk1Yr'], test_size=0.2)

    scaler = StandardScaler()

    # Fit the scalar on train only to prevent data leakage from test set
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    return X_train, X_test, y_train, y_test