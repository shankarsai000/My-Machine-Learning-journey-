# ==========================================
# DAY 10: Pandas Part 1 (Foundations)
# ==========================================

import pandas as pd
import numpy as np

# ----------- 1. SERIES -----------
s = pd.Series([10, 20, 30])
print("Series:\n", s)


# ----------- 2. DATAFRAME CREATION -----------
data = {
    "age": [25, 30, None, 35],
    "salary": [50000, 60000, 70000, None]
}
df = pd.DataFrame(data)
print("\nDataFrame:\n", df)


# ----------- 3. BASIC INFO -----------
print("\nInfo:")
df.info()

print("\nDescribe:\n", df.describe())


# ----------- 4. COLUMN SELECTION -----------
print("\nAge Column:\n", df["age"])


# ----------- 5. MULTIPLE COLUMN SELECTION -----------
print("\nAge & Salary:\n", df[["age", "salary"]])


# ----------- 6. ROW SELECTION (iloc) -----------
print("\nFirst Row:\n", df.iloc[0])


# ----------- 7. ROW SELECTION (loc) -----------
print("\nRow 1 Salary:\n", df.loc[1, "salary"])


# ----------- 8. FILTERING -----------
print("\nSalary < 65000:\n", df[df["salary"] < 65000])


# ----------- 9. MISSING VALUES CHECK -----------
print("\nMissing Values:\n", df.isnull().sum())


# ----------- 10. FILL MISSING VALUES -----------
df["age"] = df["age"].fillna(df["age"].mean())
df["salary"] = df["salary"].fillna(df["salary"].median())
print("\nAfter Filling:\n", df)


# ----------- 11. FEATURE ENGINEERING -----------
df["bonus"] = df["salary"] * 0.1
print("\nBonus Feature:\n", df)


# ----------- 12. BINARY FEATURE -----------
df["high_salary"] = df["salary"] > 65000
print("\nHigh Salary Flag:\n", df)


# ----------- 13. SCALING (NORMALIZATION) -----------
df["salary_scaled"] = (df["salary"] - df["salary"].mean()) / df["salary"].std()
print("\nScaled Salary:\n", df)


# ----------- 14. DROP COLUMN -----------
df = df.drop("bonus", axis=1)
print("\nAfter Dropping Bonus:\n", df)


# ----------- 15. ML READY DATA -----------
X = df.select_dtypes(include=[np.number])
print("\nML Ready Data:\n", X)