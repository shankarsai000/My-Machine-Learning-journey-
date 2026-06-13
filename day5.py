# DAY 5: OOP in Python (ML Perspective)(Part 1)
# ==========================================
# ----------- 1. BASIC CLASS -----------
# Class = Blueprint of a model
class Model:
    pass

m1 = Model()
m2 = Model()
print("Objects:", m1, m2)


# ----------- 2. __init__ (INITIALIZATION) -----------
# Used to assign parameters to the model
class LinearModel:
    def __init__(self, weight):
        self.weight = weight   # model parameter

model = LinearModel(5)
print("Weight:", model.weight)


# ----------- 3. MULTIPLE PARAMETERS -----------
# Real ML models have multiple parameters
class LinearModel:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

model = LinearModel(2, 1)
print("Weight & Bias:", model.weight, model.bias)


# ----------- 4. PREDICT METHOD -----------
# Method = behavior of model (like prediction)
class LinearModel:
    def __init__(self, weight):
        self.weight = weight

    def predict(self, x):
        return self.weight * x

model = LinearModel(3)
print("Prediction:", model.predict(5))


# ----------- 5. LINEAR EQUATION MODEL -----------
class LinearModel:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def predict(self, x):
        return self.weight * x + self.bias

model = LinearModel(2, 1)
print("Prediction:", model.predict(5))


# ----------- 6. TRAIN METHOD (LEARNING SIMULATION) -----------
# Model updates its parameter based on error
class LinearModel:
    def __init__(self, weight):
        self.weight = weight

    def train(self, x, y):
        prediction = self.weight * x
        error = y - prediction
        self.weight += error * 0.1   # update rule

model = LinearModel(1)
model.train(2, 4)
print("Updated Weight:", model.weight)


# ----------- 7. MULTIPLE TRAIN STEPS -----------
class LinearModel:
    def __init__(self, weight):
        self.weight = weight

    def train(self, X, y):
        for x, target in zip(X, y):
            pred = self.weight * x
            error = target - pred
            self.weight += error * 0.1

model = LinearModel(1)
X = [1, 2, 3]
y = [2, 4, 6]

model.train(X, y)
print("Final Weight:", model.weight)


# ----------- 8. MULTIPLE OBJECTS (MULTIPLE MODELS) -----------
model1 = LinearModel(1)
model2 = LinearModel(5)

print("Model1 Weight:", model1.weight)
print("Model2 Weight:", model2.weight)


# ----------- 9. MODEL WITH CONFIG STYLE -----------
# Similar to ML frameworks (like sklearn)
class ModelConfig:
    def __init__(self, lr, epochs):
        self.lr = lr
        self.epochs = epochs

config = ModelConfig(0.01, 10)
print("Config:", config.lr, config.epochs)


# ----------- 10. USING MODEL + CONFIG -----------
class LinearModel:
    def __init__(self, weight, lr):
        self.weight = weight
        self.lr = lr

    def train(self, x, y):
        error = y - (self.weight * x)
        self.weight += error * self.lr

model = LinearModel(1, 0.1)
model.train(2, 4)
print("Updated Weight with LR:", model.weight)


# ----------- 11. MINI ML PIPELINE -----------
# Combining everything (model + data + training)
class LinearModel:
    def __init__(self, weight):
        self.weight = weight

    def predict(self, x):
        return self.weight * x

    def train(self, X, y):
        for x, target in zip(X, y):
            pred = self.predict(x)
            error = target - pred
            self.weight += error * 0.1

model = LinearModel(1)

X = [1, 2, 3]
y = [2, 4, 6]

model.train(X, y)
print("Trained Weight:", model.weight)
print("Prediction after training:", model.predict(5))