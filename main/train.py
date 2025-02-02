# train.py

import pandas as pd
from sklearn.model_selection import train_test_split, ShuffleSplit, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV
data = pd.read_csv('data/Forcastingdataupdate.csv')

# Loại bỏ cột không cần thiết
if 'DATE' in data.columns:
    data = data.drop(columns=['DATE'])
data = pd.get_dummies(data, columns=['Location'], drop_first=False)
# Chuyển đổi các cột liên quan thành kiểu số
data['Seasonality_in_North'] = data['Seasonality_in_North'].astype(int) 
data['Seasonality_in_Central'] = data['Seasonality_in_Central'].astype(int)
data['Seasonality_in_South'] = data['Seasonality_in_South'].astype(int)

print(data)
# Kiểm tra và xử lý dữ liệu thiếu
if data.isnull().sum().any():
    print("Dữ liệu có giá trị thiếu, cần xử lý!") 
    data = data.dropna() 

# Chọn các tính năng (features) và nhãn (target)
features = ['Export_volume', 'Import_Volume', 'Seasonality_in_North', 'Seasonality_in_Central', 'Seasonality_in_South'] + [col for col in data.columns if col.startswith('Location_')]

target = 'Price'

X = data[features]
y = data[target]

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hàm tìm kiếm mô hình tốt nhất

algos = {
        'Linear Regression': {
            'model': LinearRegression(),
            'params': {}
        },
        'Lasso Regression': {
            'model': Lasso(),
            'params': {
                'alpha': [0.1, 0.5, 1, 2],
                'selection': ['random', 'cyclic']
            }
        },
        'Decision Tree Regressor': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['squared_error', 'friedman_mse'],
                'splitter': ['best', 'random'],
                'max_depth': [None, 5, 10, 20]
            }
        },
        'Random Forest Regressor': {
            'model': RandomForestRegressor(),
            'params': {
                'n_estimators': [50, 100, 200],  
                'max_depth': [None, 5, 10, 20],   
                'min_samples_split': [2, 5, 10],  
                'min_samples_leaf': [1, 2, 4],    
                'bootstrap': [True, False]        
            }
        }
    }
def find_best_model_using_gridsearchcv(X, y):
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X, y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })
    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])

# Tìm mô hình tốt nhất
best_models = find_best_model_using_gridsearchcv(X, y)
print("Các mô hình tốt nhất:\n", best_models)

# Chọn mô hình tốt nhất
best_model_name = best_models.iloc[best_models['best_score'].idxmax()]['model']
best_params = best_models.iloc[best_models['best_score'].idxmax()]['best_params']
print(f"Chọn mô hình: {best_model_name} với tham số {best_params}")

# Huấn luyện mô hình tốt nhất
best_model = algos[best_model_name]['model'].set_params(**best_params)
best_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = best_model.predict(X_test)

# Đánh giá mô hình
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'\nĐánh giá mô hình:')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'R² Score: {r2}')
# Vẽ đồ thị dự đoán so với giá trị thực tế
plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='b')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Cost')
plt.ylabel('Predicted Cost')
plt.title('Actual vs Predicted Cost')
plt.show()
# Lưu mô hình đã huấn luyện
joblib.dump(best_model, 'model/best_model.pkl')
print("Mô hình đã được lưu vào file 'best_model.pkl'")
