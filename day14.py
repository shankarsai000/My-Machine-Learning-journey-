# ==========================================
# SEABORN COMPLETE (0 → ADVANCED + ML)
# ==========================================

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ----------- 1. LOAD DATASET -----------
tips = sns.load_dataset("tips")

print(tips.head(50))


# ----------- 2. SCATTER PLOT -----------
# Relationship between bill & tip
sns.scatterplot(x="total_bill", y="tip", data=tips)

plt.title("Scatter Plot")
plt.show()


# ----------- 3. SCATTER WITH HUE -----------
# Category comparison
sns.scatterplot(
    x="total_bill",
    y="tip",
    hue="sex",
    data=tips
)

plt.title("Scatter with Hue")
plt.show()


# ----------- 4. LINE PLOT -----------
# Trend analysis
sns.lineplot(x="size", y="total_bill", data=tips)

plt.title("Line Plot")
plt.show()


# ----------- 5. HISTOGRAM -----------
# Distribution analysis
sns.histplot(tips["total_bill"], bins=20, kde=True)

plt.title("Histogram")
plt.show()


# ----------- 6. KDE PLOT -----------
# Smooth probability distribution
sns.kdeplot(tips["tip"], fill=True)

plt.title("KDE Plot")
plt.show()


# ----------- 7. BOX PLOT -----------
# Outlier detection
sns.boxplot(x="day", y="total_bill", data=tips)

plt.title("Box Plot")
plt.show()


# ----------- 8. VIOLIN PLOT -----------
# Distribution + density
sns.violinplot(x="day", y="total_bill", data=tips)

plt.title("Violin Plot")
plt.show()


# ----------- 9. BAR PLOT -----------
# Aggregation visualization
sns.barplot(x="day", y="total_bill", data=tips)

plt.title("Bar Plot")
plt.show()


# ----------- 10. COUNT PLOT -----------
# Frequency analysis
sns.countplot(x="day", data=tips)

plt.title("Count Plot")
plt.show()


# ----------- 11. HEATMAP (VERY IMPORTANT 🔥) -----------
corr = tips.corr(numeric_only=True)

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()


# ----------- 12. PAIRPLOT (EDA GOLD 🔥) -----------
sns.pairplot(tips, hue="sex")

plt.show()


# ----------- 13. JOINTPLOT -----------
# Combined scatter + distribution
sns.jointplot(
    x="total_bill",
    y="tip",
    data=tips,
    kind="scatter"
)

plt.show()


# ----------- 14. REGPLOT -----------
# Regression trend line
sns.regplot(
    x="total_bill",
    y="tip",
    data=tips
)

plt.title("Regression Plot")
plt.show()


# ----------- 15. FACETGRID (ADVANCED 🔥) -----------
g = sns.FacetGrid(tips, col="sex")
g.map(sns.scatterplot, "total_bill", "tip")

plt.show()


# ----------- 16. CATPLOT -----------
# Advanced categorical visualization
sns.catplot(
    x="day",
    y="total_bill",
    kind="box",
    data=tips
)

plt.show()


# ----------- 17. RELPLOT -----------
# Relationship plotting at high level
sns.relplot(
    x="total_bill",
    y="tip",
    hue="time",
    data=tips
)

plt.show()


# ----------- 18. SWARMPLOT -----------
# Better categorical scatter
sns.swarmplot(
    x="day",
    y="tip",
    data=tips
)

plt.title("Swarm Plot")
plt.show()


# ----------- 19. STRIP PLOT -----------
sns.stripplot(
    x="day",
    y="tip",
    data=tips
)

plt.title("Strip Plot")
plt.show()


# ----------- 20. STYLE CUSTOMIZATION -----------
sns.set_style("darkgrid")

sns.histplot(tips["tip"], kde=True)

plt.title("Styled Plot")
plt.show()


# ----------- 21. MINI ML VISUALIZATION PIPELINE 🔥 -----------

# Simulated predictions
actual = np.array([2, 4, 6, 8, 10])
predicted = np.array([2.1, 3.8, 6.2, 7.9, 9.7])

results = pd.DataFrame({
    "Actual": actual,
    "Predicted": predicted
})

sns.lineplot(data=results)

plt.title("Actual vs Predicted")
plt.grid(True)

plt.show()