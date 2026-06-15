# ==========================================
# DAY 18: STATISTICS & BAYES THEOREM FOR ML
# ==========================================

import numpy as np
from statistics import mode

# ------------------------------------------
# 1. SAMPLE DATASET
# ------------------------------------------

marks = np.array([45, 50, 55, 60, 65, 70, 75, 80, 85, 90])

print("Dataset:")
print(marks)


# ------------------------------------------
# 2. MEAN (Average)
# ------------------------------------------

mean_value = np.mean(marks)

print("\nMean:")
print(mean_value)

# ML Use:
# Average value of data


# ------------------------------------------
# 3. MEDIAN
# ------------------------------------------

median_value = np.median(marks)

print("\nMedian:")
print(median_value)

# ML Use:
# Robust against outliers


# ------------------------------------------
# 4. MODE
# ------------------------------------------

data = [10, 20, 20, 30, 40, 20]

print("\nMode:")
print(mode(data))

# Most frequent value


# ------------------------------------------
# 5. RANGE
# ------------------------------------------

range_value = np.max(marks) - np.min(marks)

print("\nRange:")
print(range_value)

# Spread of data


# ------------------------------------------
# 6. VARIANCE
# ------------------------------------------

variance = np.var(marks)

print("\nVariance:")
print(round(variance, 2))

# How much data varies


# ------------------------------------------
# 7. STANDARD DEVIATION
# ------------------------------------------

std = np.std(marks)

print("\nStandard Deviation:")
print(round(std, 2))

# Distance from mean


# ------------------------------------------
# 8. Z-SCORE
# ------------------------------------------

student_mark = 80

z_score = (student_mark - mean_value) / std

print("\nZ Score:")
print(round(z_score, 2))

# How far from average?


# ------------------------------------------
# 9. OUTLIER DETECTION
# ------------------------------------------

data = np.array([50,55,60,65,70,75,80,500])

mean_data = np.mean(data)
std_data = np.std(data)

outliers = []

for x in data:
    z = (x - mean_data) / std_data

    if abs(z) > 2:
        outliers.append(x)

print("\nOutliers:")
print(outliers)

# Used in fraud detection


# ------------------------------------------
# 10. NORMAL DISTRIBUTION
# ------------------------------------------

normal_data = np.random.normal(
    loc=70,
    scale=10,
    size=1000
)

print("\nNormal Distribution Sample:")
print(normal_data[:10])

# Many ML algorithms assume normality


# ------------------------------------------
# 11. BAYES THEOREM
# ------------------------------------------

# Disease Example

P_Disease = 0.01

P_Positive_given_Disease = 0.95

P_Positive_given_NoDisease = 0.05

P_NoDisease = 1 - P_Disease

P_Positive = (
    P_Positive_given_Disease * P_Disease +
    P_Positive_given_NoDisease * P_NoDisease
)

P_Disease_given_Positive = (
    P_Positive_given_Disease * P_Disease
) / P_Positive

print("\nBayes Theorem Result:")
print(round(P_Disease_given_Positive, 4))


# ------------------------------------------
# 12. SPAM DETECTION LOGIC
# ------------------------------------------

spam_prior = 0.20

free_given_spam = 0.80

free_given_not_spam = 0.10

not_spam = 1 - spam_prior

free_probability = (
    free_given_spam * spam_prior +
    free_given_not_spam * not_spam
)

spam_given_free = (
    free_given_spam * spam_prior
) / free_probability

print("\nSpam Probability:")
print(round(spam_given_free, 4))


# ------------------------------------------
# 13. ML INTERPRETATION
# ------------------------------------------

print("\nML Insight:")
print("Mean = Central tendency")
print("Std Dev = Variation")
print("Z Score = Relative position")
print("Bayes = Updating beliefs")
print("Statistics powers ML decisions")


# ------------------------------------------
# 14. WHY STATISTICS MATTERS
# ------------------------------------------

print("\nReal World Uses:")
print("Recommendation Systems")
print("Fraud Detection")
print("Spam Detection")
print("Medical Diagnosis")
print("Predictive Analytics")


# ------------------------------------------
# 15. FINAL TAKEAWAY
# ------------------------------------------

print("\nMathematics Behind ML:")
print("Vectors -> Data")
print("Matrices -> Dataset")
print("Dot Product -> Prediction")
print("Probability -> Confidence")
print("Statistics -> Understanding Data")
print("Bayes -> Intelligent Updating")