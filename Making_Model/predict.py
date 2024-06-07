import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor


# 이상치를 제거하기 위한 함수 정의
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


data = pd.read_csv("C:\\Users\\tlgus\Desktop\ML_alternative\pythonProject1\\after_data.csv")

data.drop(columns=['Unnamed: 0', 'MIN_L', 'MIN_H', 'MAX_L', 'MAX_H'], inplace=True)
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# 이상치 제거
clean_data = remove_outliers(data, 'dscamt')
#clean_data = remove_outliers(clean_data, 'RN_ST')

# # 상관계수 계산
# correlation_matrix = data[['SKY', 'PRE', 'WF']].corr()
# print("Correlation matrix:\n", correlation_matrix)

# # VIF 계산
# X_data = clean_data[['SKY', 'PRE', 'WF']]
#
#
# vif_data = pd.DataFrame()
# vif_data["feature"] = X_data.columns
# vif_data["VIF"] = [variance_inflation_factor(X_data.values, i) for i in range(len(X_data.columns))]
# print(vif_data)



# # 선형 회귀 모델
# linear_model = LinearRegression()
# linear_model.fit(X_train_clean, y_train_clean)
# y_pred_linear = linear_model.predict(X_test_clean)
# print("Linear Regression R^2:", r2_score(y_test_clean, y_pred_linear))
#
# # 의사결정 트리 모델
# tree_model = DecisionTreeRegressor(max_depth=5)
# tree_model.fit(X_train_clean, y_train_clean)
# y_pred_tree = tree_model.predict(X_test_clean)
# print("Decision Tree R^2:", r2_score(y_test_clean, y_pred_tree))
# print("Feature Importances:", tree_model.feature_importances_)

# training error, test error를 저장할 리스트
train_accuracies = []
test_accuracies = []
category_counts = range(1,20)

# # 릿지 회귀 모델 생성 및 훈련
# ridge_model = Ridge(alpha=1.0)
# ridge_model.fit(X_train_clean, y_train_clean)

# 의사결정트리 모델 생성 및 훈련 (가지치기 적용)
for categories in category_counts:
    # 데이터 범주화
    clean_data['dscamt_normalized'] = pd.qcut(clean_data['dscamt'], q=categories, labels=False)

    # 특성과 타겟 변수 분리
    X_clean = clean_data[['PRE', 'WF', 'RN_ST', 'MIN', 'MAX']]
    Y_clean = clean_data['dscamt_normalized']

    # 데이터를 훈련 세트와 테스트 세트로 분리
    X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(X_clean, Y_clean, test_size=0.2,
                                                                                random_state=42)

    pruned_tree_model = DecisionTreeClassifier(max_leaf_nodes=20, random_state=42)
    pruned_tree_model.fit(X_train_clean, y_train_clean)

    # 훈련 데이터에 대한 예측 및 정확도 계산
    y_train_pred = pruned_tree_model.predict(X_train_clean)
    train_accuracy = accuracy_score(y_train_clean, y_train_pred)
    train_accuracies.append(train_accuracy)

    # 테스트 데이터에 대한 예측 및 정확도 계산
    y_test_pred = pruned_tree_model.predict(X_test_clean)
    test_accuracy = accuracy_score(y_test_clean, y_test_pred)
    test_accuracies.append(test_accuracy)

# 모델 예측 및 평가
# y_pred = ridge_model.predict(X_test_clean)
# mse = mean_squared_error(y_test_clean, y_pred)
# r2 = r2_score(y_test_clean, y_pred)
# print("Test_data_Mean Squared Error:", mse)
# print("Test_data_R_2 : ", r2)

# # 모델 성능 평가
# mse_pruned_tree = mean_squared_error(y_test_clean, y_pred_pruned_tree)
# r2_pruned_tree = r2_score(y_test_clean, y_pred_pruned_tree)
#
# print(f'MSE: {mse_pruned_tree}')
# print(f'R²: {r2_pruned_tree}')

# 의사결정 트리 시각화
plt.figure(figsize=(10,6))
plt.plot(category_counts, train_accuracies, label='On training data', linestyle='-', marker='o')
plt.plot(category_counts, test_accuracies, label='On test data', linestyle='--', marker='x')
plt.xlabel('Number of Categories')
plt.ylabel('Accuracy')
plt.title('Impact of Number of Categories on Model Performance')
plt.legend()
plt.grid(True)
plt.show()

# # 결과 시각화
# plt.figure(figsize=(10, 6))
# plt.scatter(y_test_clean, y_pred, color='blue')
# plt.plot([y_test_clean.min(), y_test_clean.max()], [y_test_clean.min(), y_test_clean.max()], 'k--', lw=4)  # 대각선 추가
# plt.xlabel('Actual')
# plt.ylabel('Predicted')
# plt.title('Actual vs. Predicted')
# plt.savefig('linear_regression.png', dpi=300)
# plt.close()