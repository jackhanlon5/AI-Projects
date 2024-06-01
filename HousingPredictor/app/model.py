import pandas as pd
from xgboost import XGBRegressor as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder


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

    return best_overall_acc, best_overall_preds, prices, best_overall_params


def main():
    attributes, labels = load_data()
    best_acc, predictions, prices, params = param_tuning(attributes, labels)
    print("Lowest RMSE: ", best_acc)
    print("Best parameters: ", params)
    # print(predictions)
    # print(prices)
    for p in range(len(predictions)):
        print(f"Prediction: {predictions[p]}\tActual: {prices[p]}\tDifference: {predictions[p] - prices[p]}")
 

if __name__ == "__main__":
    main()