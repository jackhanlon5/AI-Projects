import pandas as pd
from xgboost import XGBRegressor as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import joblib


def load_data():
    df = pd.read_csv("HousingPredictor/app/Housing.csv", delimiter=',')

    need_encode = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus']
    
    le = LabelEncoder()

    for col in need_encode:
        le.fit(df[col])
        df[col] = le.transform(df[col])

    X=df.loc[:,df.columns!='price']
    y=df.loc[:,'price']

    return X, y


def param_tuning(attributes, labels):
    best_overall_acc = float('inf')
    best_overall_params = None
    best_overall_preds = None

    for split in range(10, 18):
        size = split * 0.05
        X_train, X_test, y_train, y_test = train_test_split(attributes, labels, train_size=size, random_state=42)
        model = xgb(objective='reg:squarederror', nthread=16)
        parameters = {
            'max_depth': range(2, 10, 1),
            'n_estimators': range(10, 200, 10),
            'learning_rate': [0.1, 0.01, 0.05, 0.25]
        }
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=parameters,
            scoring='neg_mean_squared_error',
            n_jobs=3,
            cv=2,
            verbose=True
        )
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        trained_model = grid_search.best_estimator_
        xgb_preds = trained_model.predict(X_test)
        
        accuracy = mean_squared_error(y_test, xgb_preds, squared=False)
        
        if accuracy < best_overall_acc:
            best_overall_acc = accuracy
            best_overall_params = best_params
            best_overall_preds = xgb_preds
            prices = y_test.array
            best_size = size
            best_model = xgb(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'], learning_rate=best_params['learning_rate'], objective='reg:squarederror')

    return best_overall_acc, best_overall_preds, prices, best_overall_params, best_model, size


# def run_model(attributes, labels):
#     # Useless function now - Only purpose is holding the best parameters
#     X_train, X_test, y_train, y_test = train_test_split(attributes, labels, train_size=.85, random_state=42)
#     model = xgb(n_estimators=20, max_depth=2, learning_rate=0.05, objective='reg:squarederror')


def training(attributes, labels):
    best_acc, predictions, prices, params, model, size = param_tuning(attributes, labels)
    joblib.dump(model, 'xgb_model.joblib')
    print("Lowest RMSE: ", best_acc)
    print("Best parameters: ", params)
    print(size)
 

if __name__ == "__main__":
    attributes, labels = load_data()
    training(attributes, labels)