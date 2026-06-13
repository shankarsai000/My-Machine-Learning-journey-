# ==========================================
# DAY 3: Data Structures in ML
# ==========================================

# ----------- 1. LIST (Dataset Storage) -----------
# Think: features of data
features = [10, 20, 30, 40]

print("First feature:", features[0])

# Update value (like preprocessing)
features[1] = 25
print("Updated features:", features)


# ----------- 2. LIST COMPREHENSION (FAST PROCESSING) -----------
# Normalize data (very common in ML)
normalized = [x / 10 for x in features]
print("Normalized:", normalized)


# ----------- 3. TUPLE (IMMUTABLE DATA) -----------
# Fixed coordinates / configs
point = (3, 5)
print("Point:", point)


# ----------- 4. DICTIONARY (VERY IMPORTANT 🔥) -----------
# Represent a data sample (like JSON)
data_point = {
    "age": 21,
    "marks": 85,
    "passed": True
}

print("Age:", data_point["age"])

# Update (feature engineering)
data_point["marks"] = 90
print("Updated data:", data_point)


# ----------- 5. SET (UNIQUE VALUES) -----------
# Remove duplicate labels
labels = [1, 1, 0, 1, 0, 0]
unique_labels = set(labels)

print("Unique labels:", unique_labels)


# ----------- 6. COMBINING DATA (REAL ML STYLE) -----------
# Features + labels
X = [1, 2, 3, 4]
y = [2, 4, 6, 8]

dataset = list(zip(X, y))
print("Dataset:", dataset)


# ----------- 7. MINI ML-LIKE PROCESS -----------
# Simple transformation
processed_X = [x * 2 for x in X]

# Prediction using simple rule
predictions = [x * 2 for x in processed_X]

print("Predictions:", predictions)

# ----------- 8. MULTIPLE DATA POINTS (REAL DATASET STYLE) -----------
# List of dictionaries (very common in ML before Pandas)

dataset = [
    {"age": 20, "marks": 80},
    {"age": 22, "marks": 90},
    {"age": 19, "marks": 70}
]

print("Dataset:", dataset)

# Extract only marks (feature extraction)
marks = [data["marks"] for data in dataset]
print("Marks:", marks)


# ----------- 9. SIMPLE FEATURE ENGINEERING -----------
# Add new feature (pass/fail)

for data in dataset:
    data["passed"] = data["marks"] > 75

print("Updated dataset:", dataset)


# ----------- 10. FILTERING DATA (LIKE PREPROCESSING) -----------
# Keep only passed students

passed_students = [data for data in dataset if data["passed"]]
print("Passed students:", passed_students)


# ----------- 11. UNIQUE LABEL EXTRACTION (SET USE) -----------
labels = ["cat", "dog", "cat", "bird"]
unique = list(set(labels))

print("Unique labels:", unique)


# ----------- 12. MINI PIPELINE (END-TO-END FLOW) -----------
# Raw data
X = [1, 2, 3, 4]

# Step 1: preprocessing
X_processed = [x + 1 for x in X]

# Step 2: transformation
X_scaled = [x * 2 for x in X_processed]

# Step 3: prediction
predictions = [x * 3 for x in X_scaled]

print("Pipeline Output:", predictions)