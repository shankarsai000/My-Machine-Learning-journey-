# ==========================================
# DAY 11 & 12: Pandas Part 2 (Advanced + ML)
# ==========================================

import pandas as pd
import numpy as np

# ----------- 1. SAMPLE DATASET -----------
data = {
    "name": ["A", "B", "C", "D", "E"],
    "city": ["X", "Y", "X", "Y", "X"],
    "salary": [50000, 60000, 70000, 80000, 90000],
    "age": [25, 30, 35, 40, 28]
}
df = pd.DataFrame(data)
print("Dataset:\n", df)


# ----------- 2. GROUPBY (CORE 🔥) -----------
print("\nMean Salary by City:\n", df.groupby("city")["salary"].mean())


# ----------- 3. MULTIPLE AGGREGATIONS -----------
print("\nAggregations:\n", df.groupby("city")["salary"].agg(["mean", "max", "min"]))


# ----------- 4. GROUPBY MULTIPLE COLUMNS -----------
print("\nCity + Age Mean:\n", df.groupby(["city", "age"])["salary"].mean())


# ----------- 5. RESET INDEX -----------
grouped = df.groupby("city")["salary"].mean().reset_index()
print("\nReset Index:\n", grouped)


# ----------- 6. SORTING -----------
print("\nSorted by Salary:\n", df.sort_values(by="salary", ascending=False))


# ----------- 7. RANKING -----------
df["rank"] = df["salary"].rank(ascending=False)
print("\nRanking:\n", df)


# ----------- 8. PIVOT TABLE -----------
pivot = df.pivot_table(values="salary", index="city", aggfunc="mean")
print("\nPivot Table:\n", pivot)


# ----------- 9. ADD SECOND DATAFRAME -----------
data2 = {
    "name": ["A", "B", "C", "F"],
    "department": ["HR", "IT", "Finance", "HR"]
}
df2 = pd.DataFrame(data2)


# ----------- 10. MERGE (INNER JOIN) -----------
merged = pd.merge(df, df2, on="name", how="inner")
print("\nMerged (Inner):\n", merged)


# ----------- 11. LEFT JOIN -----------
left_join = pd.merge(df, df2, on="name", how="left")
print("\nLeft Join:\n", left_join)


# ----------- 12. RIGHT JOIN -----------
right_join = pd.merge(df, df2, on="name", how="right")
print("\nRight Join:\n", right_join)


# ----------- 13. CONCATENATION -----------
df_concat = pd.concat([df, df])
print("\nConcatenated:\n", df_concat)


# ----------- 14. APPLY FUNCTION -----------
df["salary_k"] = df["salary"].apply(lambda x: x / 1000)
print("\nSalary in K:\n", df)


# ----------- 15. PIPELINE STYLE WORKFLOW 🔥 -----------

# Step 1: Filter
df_clean = df[df["salary"] < 85000]

# Step 2: Feature engineering
df_clean["high_salary"] = df_clean["salary"] > 70000

# Step 3: Group analysis
summary = df_clean.groupby("city")["salary"].mean()

print("\nFinal Pipeline Output:\n", summary)


# ----------- 16. FINAL ML READY DATA -----------
X = df_clean[["age", "salary"]].values
print("\nML Ready Data:\n", X)