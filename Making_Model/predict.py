import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# 이상치를 제거하기 위한 함수 정의
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


data = pd.read_csv("/after_data.csv")

data.drop(columns=['Unnamed: 0', 'MIN_L', 'MIN_H', 'MAX_L', 'MAX_H'], inplace=True)
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# 이상치 제거
clean_data = remove_outliers(data, 'dscamt')
clean_data = remove_outliers(clean_data, 'RN_ST')

# 특성과 타겟 변수 분리
X_clean = clean_data.drop('dscamt', axis=1)
y_clean = clean_data['dscamt']

# 데이터를 훈련 세트와 테스트 세트로 분리
X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

# 의사결정트리 모델 생성 및 훈련 (가지치기 적용)
pruned_tree_model = DecisionTreeRegressor(max_depth=5, random_state=42)
pruned_tree_model.fit(X_train_clean, y_train_clean)

# 모델 예측
y_pred_pruned_tree = pruned_tree_model.predict(X_test_clean)

# 모델 성능 평가
mse_pruned_tree = mean_squared_error(y_test_clean, y_pred_pruned_tree)
r2_pruned_tree = r2_score(y_test_clean, y_pred_pruned_tree)

print(f'MSE: {mse_pruned_tree}')
print(f'R²: {r2_pruned_tree}')

# 의사결정 트리 시각화
fig, ax = plt.subplots(figsize=(100, 20))
plot_tree(pruned_tree_model, feature_names=X_clean.columns, filled=True, ax=ax, fontsize=10)
fig.savefig('decision_tree.png', dpi=300)
plt.close(fig)

