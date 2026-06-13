
# ==========================================
# DAY X: Matplotlib Complete (ML Perspective)
# ==========================================

import matplotlib.pyplot as plt
import numpy as np

# Sample data
x = np.array([1, 2, 3, 4, 5])
y = np.array([10, 20, 25, 30, 40])

# ----------- 1. BASIC LINE PLOT -----------
plt.figure(figsize=(6,4))
plt.plot(x, y)
plt.title("Line Plot")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.grid(True)
plt.show()



# ----------- 2. CUSTOM LINE STYLE -----------
plt.figure(figsize=(6,4))
plt.plot(x, y, linestyle="--", marker="o")
plt.title("Custom Line")
plt.show()


# ----------- 3. MULTIPLE LINES -----------
y2 = np.array([5, 15, 20, 25, 35])

plt.figure(figsize=(6,4))
plt.plot(x, y, label="Sales")
plt.plot(x, y2, label="Profit")
plt.legend()
plt.title("Multiple Lines")
plt.show()


# ----------- 4. BAR CHART -----------
categories = ["A", "B", "C", "D"]

values = [50, 70, 30, 90]

plt.figure(figsize=(6,4))
plt.bar(categories, values)
plt.title("Bar Chart")
plt.show()


# ----------- 5. HORIZONTAL BAR CHART -----------
plt.figure(figsize=(6,4))
plt.barh(categories, values)
plt.title("Horizontal Bar")
plt.show()


# ----------- 6. HISTOGRAM -----------
data = np.random.randn(1000)

plt.figure(figsize=(6,4))
plt.hist(data, bins=20)
plt.title("Histogram")
plt.show()


# ----------- 7. SCATTER PLOT (VERY IMPORTANT 🔥) -----------
x = np.random.rand(50)
y = np.random.rand(50)

plt.figure(figsize=(6,4))
plt.scatter(x, y)
plt.title("Scatter Plot")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


# ----------- 8. PIE CHART -----------
sizes = [40, 30, 20, 10]
labels = ["Python", "ML", "AI", "Data"]

plt.figure(figsize=(6,4))
plt.pie(sizes, labels=labels, autopct="%1.1f%%")
plt.title("Pie Chart")
plt.show()


# ----------- 9. SUBPLOTS -----------
fig, ax = plt.subplots(1, 2, figsize=(10,4))

ax[0].plot(x, y)
ax[0].set_title("Line")

ax[1].bar(categories, values)
ax[1].set_title("Bar")

plt.show()


# ----------- 10. CUSTOMIZATION -----------
plt.figure(figsize=(6,4))
plt.plot([1,2,3], [2,4,6], linewidth=3)
plt.title("Customized Plot")
plt.grid(True)
plt.show()


# ----------- 11. SAVE FIGURE -----------
plt.figure(figsize=(6,4))
plt.plot(x, y)
plt.title("Saved Plot")
plt.savefig("plot.png")


# ----------- 12. REAL ML LOSS CURVE 🔥 -----------
epochs = np.arange(1, 11)
loss = [10, 8, 6.5, 5, 4, 3.5, 3, 2.8, 2.5, 2.2]

plt.figure(figsize=(6,4))
plt.plot(epochs, loss, marker="o")
plt.title("Training Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.show()


# ----------- 13. ACCURACY CURVE -----------
accuracy = [50, 60, 68, 72, 78, 82, 85, 88, 90, 92]

plt.figure(figsize=(6,4))
plt.plot(epochs, accuracy, marker="o")
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()


# ----------- 14. DECISION BOUNDARY STYLE VISUALIZATION -----------
x = np.random.randn(100)
y = np.random.randn(100)

labels = np.random.randint(0, 2, 100)

plt.figure(figsize=(6,4))
plt.scatter(x, y, c=labels)
plt.title("Classification Visualization")
plt.show()


# ----------- 15. MINI ML VISUALIZATION PIPELINE 🔥 -----------

# Simulated predictions
actual = np.array([2, 4, 6, 8, 10])
predicted = np.array([2.1, 3.9, 6.2, 7.8, 9.7])

plt.figure(figsize=(6,4))
plt.plot(actual, label="Actual")
plt.plot(predicted, label="Predicted")

plt.title("Actual vs Predicted")
plt.xlabel("Samples")
plt.ylabel("Values")
plt.legend()
plt.grid(True)

plt.show()
