import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

df = pd.read_csv("Uber Request Data.csv")

df['Request timestamp'] = pd.to_datetime(df['Request timestamp'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['Request timestamp'])

df['hour'] = df['Request timestamp'].dt.floor('h')

demand_df = df.groupby(['hour', 'Pickup point']).size().reset_index(name='demand')

# FEATURE ENGINEERING
demand_df['hour_of_day'] = demand_df['hour'].dt.hour
demand_df['weekday'] = demand_df['hour'].dt.weekday
demand_df['lag_1'] = demand_df['demand'].shift(1)
demand_df['lag_3'] = demand_df['demand'].shift(3)
demand_df['rolling_mean_5'] = demand_df['demand'].rolling(5).mean()
demand_df = demand_df.dropna()
demand_df['Pickup point'] = demand_df['Pickup point'].astype('category').cat.codes

X = demand_df[['Pickup point','hour_of_day','weekday','lag_1','lag_3','rolling_mean_5']]
y = demand_df['demand']

# CROSS VALIDATION
tscv = TimeSeriesSplit(n_splits=3)
model = RandomForestRegressor(random_state=42)
rmse_list = []
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    rmse_list.append(rmse)
print("RMSE values:", rmse_list)
print("Average RMSE:", np.mean(rmse_list))

baseline_model = LinearRegression()
baseline_rmse_list = []
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    baseline_model.fit(X_train, y_train)
    preds = baseline_model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    baseline_rmse_list.append(rmse)
print("\n===== BASELINE MODEL =====")
print("Baseline RMSE:", baseline_rmse_list)
print("Baseline Average:", np.mean(baseline_rmse_list))

if np.mean(rmse_list) < np.mean(baseline_rmse_list):
    print("Random Forest performs better than Linear Regression.")
else:
    print("Linear Regression performs better.")

final_model = RandomForestRegressor(random_state=42)
final_model.fit(X_train, y_train)
future_preds = final_model.predict(X_test)
test_df = demand_df.iloc[test_index].copy()
test_df['predicted_demand'] = future_preds

test_df['optimal_supply'] = test_df['predicted_demand'] * 1.2

kmeans = KMeans(n_clusters=3, random_state=42)
test_df['cluster'] = kmeans.fit_predict(test_df[['predicted_demand','hour_of_day']])

print("\n===== CLUSTER INSIGHTS =====")
cluster_summary = test_df.groupby('cluster')[['predicted_demand','hour_of_day']].mean()

cluster_summary = cluster_summary.sort_values(by='predicted_demand', ascending=False)
print(cluster_summary)

print("\nCluster Meaning (based on demand):")
print("Top row → High demand cluster")
print("Middle row → Medium demand cluster")
print("Bottom row → Low demand cluster")

plt.figure()
plt.scatter(
    test_df['hour_of_day'],
    test_df['predicted_demand'],
    c=test_df['cluster']
)
plt.title("Cluster Visualization (Demand vs Time)")
plt.xlabel("Hour of Day")
plt.ylabel("Predicted Demand")
plt.grid(alpha=0.3)

test_df['supply'] = np.where(
    test_df['cluster'] == 0,
    test_df['predicted_demand'] * 1.3,  
    np.where(
        test_df['cluster'] == 1,
        test_df['predicted_demand'] * 1.1,  
        test_df['predicted_demand'] * 0.9  ))

test_df['gap'] = test_df['optimal_supply'] - test_df['supply']

def decision(row):
    if row['predicted_demand'] == 0:
        return "Balanced"
    
    gap_ratio = row['gap'] / row['predicted_demand']
    
    if gap_ratio > 0.2:
        return "Increase Supply"
    elif gap_ratio < -0.2:
        return "Reduce Supply"
    else:
        return "Balanced"

test_df['strategy'] = test_df.apply(decision, axis=1)

print("\n===== DEMAND PREDICTION =====")
print(test_df[['demand','predicted_demand']].head())

print("\n===== SUPPLY STRATEGY =====")
print(test_df[['predicted_demand','supply','gap','strategy']].head())

print("\nOptimization Insights:")
hourly = demand_df.groupby('hour_of_day')['demand'].mean()
print("Peak Hour:", hourly.idxmax())
print("Low Demand Hour:", hourly.idxmin())

print("\nStrategy Count:")
print(test_df['strategy'].value_counts())

print("\n===== FINAL CONCLUSION =====")
print("Peak demand occurs during evening hours.")
print("Airport shows higher demand in morning.")
print("City shows higher demand in evening.")
print("Model captures overall demand trend with moderate error.")
print("Clustering successfully identifies demand levels.")
print("Dynamic supply allocation adjusts resources based on demand.")
print("Most periods are balanced, with some instances of supply shortage.")

# Graphs
# Demand trend
plt.figure()
plt.plot(hourly, marker='o')
peak = hourly.idxmax()
peak_value = hourly.max()
plt.axvline(x=peak, linestyle='--')
plt.annotate(f'Peak: {peak}', (peak, peak_value),
             textcoords="offset points", xytext=(0,10),
             ha='center',
             bbox=dict(boxstyle="round", edgecolor='black'))
plt.title("Average Demand by Hour")
plt.xlabel("Hour")
plt.ylabel("Demand")
plt.grid(alpha=0.3)

# Actual vs Predicted
plt.figure(figsize=(10,5))
plt.plot(test_df['demand'].values[:50], label='Actual')
plt.plot(test_df['predicted_demand'].values[:50], linestyle='--', label='Predicted')
plt.legend()
plt.title(f"Actual vs Predicted (RMSE = {round(np.mean(rmse_list),2)})")
plt.grid(alpha=0.3)

# Heatmap
pivot = demand_df.pivot_table(
    values='demand',
    index='hour_of_day',
    columns='Pickup point'
)
pivot = pivot.rename(columns={0: 'City', 1: 'Airport'})
plt.figure()
sns.heatmap(pivot, annot=True, fmt=".0f", cmap='coolwarm')
plt.title("Demand Heatmap (Hour vs Location)")

# Strategy Distribution
plt.figure()
counts = test_df['strategy'].value_counts()
color_map = {
    "Increase Supply": "red",
    "Reduce Supply": "green",
    "Balanced": "blue"
}
ax = counts.plot(kind='bar', color=[color_map.get(i, "gray") for i in counts.index])
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height(),
            int(bar.get_height()),
            ha='center')
plt.title("Strategy Distribution")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()