from modules.validation      import Validate
from modules.data            import Data
from modules.log_experiments import MlFlow
from modules.gridsearch      import GridSeach
from modules.models          import Models
       
class Pipeline:
    
    def __init__(self):

        self.data        = Data()
        self.models      = Models()
        self.grid_search = GridSeach()
        self.mlFlow      = MlFlow()
        self.validate    = Validate()

    def train(self,model,X_train,y_train):

        model.fit(X_train,y_train)

        return model


    def initialize_model(self,model_name,parameter):

        if model_name in self.models.emsemble_names:
            for decision_tree_parameter in self.grid_search.iterate_by_name("DecisionTreeClassifier"):
                model = self.models.instantiate(model_name,(parameter,decision_tree_parameter)) 
                 
        else:
            model = self.models.instantiate(model_name,parameter)
        
        parameter["model_name"] = model_name

        return model

    
    def fit(self):
        
        self.datasets = self.data.generate()
        for X, y in self.datasets:
            X_train, X_test, y_train, y_test = self.data.transform.split(X, y)
            X_train, X_test                  = self.data.transform.normalize(X_train,X_test)

            for model_name,parameter in self.grid_search.generate_parameter():
                model   = self.initialize_model(model_name,parameter)
                model   = self.train(model,X_train,y_train)
                metrics = self.validate.eval(model,X_test,X_train,y_train,y_test)
                self.mlFlow.log_result(parameter,metrics)


pipeline = Pipeline()
pipeline.fit()


























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





