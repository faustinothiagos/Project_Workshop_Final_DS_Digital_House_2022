import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing   import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import numpy as np

df_pipeline = pd.read_csv('../Pipeline/df_nonull.csv', low_memory=False)

feats_pipeline = ['Outstanding_Debt','Monthly_Inhand_Salary','Credit_History_age','Amount_invested_monthly', 'Payment_of_Min_Amount']

X_train, X_test, y_train, y_test = train_test_split(df_pipeline[feats_pipeline], df_pipeline['Credit_Score'], test_size=0.2, stratify = df_pipeline['Credit_Score'], random_state=42)

def factorize_pipeline(df=df_pipeline):
    for col in df.select_dtypes('object'):
        df[col], _ = df[col].factorize()
        
def stdscaler_pipeline(df=df_pipeline):
    for col in df.select_dtypes(['int64', 'float64']):
        df[col] = StandardScaler().fit_transform(df[col].values.reshape(-1,1))
        
def ordenc_pipeline(df=df_pipeline):
    df['Credit_Score'] = OrdinalEncoder().fit_transform(df['Credit_Score'].values.reshape(-1,1)).astype(int)


passos = [('Factorize', factorize_pipeline()),
          ('StandardScaler', stdscaler_pipeline()),
          ('OrdinalEncoder', ordenc_pipeline()),
          ('XGBClassifier', XGBClassifier(eval_metric = 'logloss',
                                          use_label_encoder = False,
                                          random_state = 42,
                                          colsample_bytree = 0.6,
                                          gamma = 1.0,
                                          learning_rate = 0.1,
                                          max_depth = None,
                                          min_child_weight = 0.7,
                                          n_estimators = 1500,
                                          subsample = 1.0)) 
          ]

pipeline = Pipeline(passos)

pickle.dump(pipeline.fit(X_train, y_train), open('modelostreamlit.pkl', 'wb'))

#

''' 
datasets    = generate_datasets()
transform   = Transform()
models      = Models()
grid_search = GridSeach()


for X, y in datasets:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    X_train, X_test = transform.normalize(X_train,X_test)

    for model_name,parameter in grid_search.generate_parameter():
        
        if model_name in models.emsemble_names:
            for decision_tree_parameter in grid_search.iterate_by_name("DecisionTreeClassifier"):
                model = models.instantiate(model_name,(parameter,decision_tree_parameter))
        else:
            model = models.instantiate(model_name,parameter)

        cv     = StratifiedKFold(n_splits=10, random_state=41, shuffle=True)
        result = cross_val_score(model, X, y, cv=cv, n_jobs=-1) #evaluate_performance(model,model_name,X_train,y_train,cv)
        
        model.fit(X_train,y_train)
        y_hat_test   = model.predict(X_test)
        y_hat_train  = model.predict(X_train)

        parameter["model_name"] = model_name

        metrics = {
            "10-fold accuracy":  result.mean().round(3),
            "10-fold deviation": result.std().round(3),
            "train accuracy_score":  accuracy_score(  y_train, y_hat_train).round(3),
            "train f1_score":        f1_score(        y_train, y_hat_train).round(3),
            "train precision_score": precision_score( y_train, y_hat_train).round(3),
            "train recall_score":    recall_score(    y_train, y_hat_train).round(3),
            "test accuracy_score":  accuracy_score(  y_test, y_hat_test).round(3),
            "test f1_score":        f1_score(        y_test, y_hat_test).round(3),
            "test precision_score": precision_score( y_test, y_hat_test).round(3),
            "test recall_score":    recall_score(    y_test, y_hat_test).round(3)
        }

        print(parameter)
        print(metrics)

'''