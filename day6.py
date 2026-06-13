# ==========================================
# DAY X: OOP Part 2 — ML System Design
# ==========================================

# ----------- 1. BASE MODEL (ABSTRACTION) -----------
# Blueprint for all models
class BaseModel:
    def train(self, X, y):
        print("Training base model...")

    def predict(self, x):
        raise NotImplementedError("Subclasses must implement predict")


# ----------- 2. LINEAR MODEL (INHERITANCE) -----------
class LinearModel(BaseModel):
    def __init__(self, weight, bias=0):
        self.__weight = weight     # encapsulation (private)
        self.bias = bias

    def predict(self, x):
        return self.__weight * x + self.bias

    def train(self, X, y):
        # simple training loop (learning simulation)
        for x_val, target in zip(X, y):
            pred = self.predict(x_val)
            error = target - pred
            self.__weight += error * 0.1   # update rule

    def get_weight(self):
        return self.__weight


# ----------- 3. ANOTHER MODEL (POLYMORPHISM) -----------
class SquareModel(BaseModel):
    def predict(self, x):
        return x ** 2


# ----------- 4. CONFIG-DRIVEN MODEL -----------
class ModelConfig:
    def __init__(self, **kwargs):
        self.lr = kwargs.get("lr", 0.1)
        self.epochs = kwargs.get("epochs", 5)

config = ModelConfig(lr=0.05, epochs=3)
print("Config:", config.lr, config.epochs)


# ----------- 5. PIPELINE DESIGN (REAL ML STYLE) -----------
class Pipeline:
    def __init__(self, model):
        self.model = model

    def preprocess(self, X):
        # simple transformation (like normalization/scaling)
        return [x * 2 for x in X]

    def run(self, X):
        X_processed = self.preprocess(X)
        return [self.model.predict(x) for x in X_processed]


# ----------- 6. MULTIPLE MODELS HANDLING -----------
models = [LinearModel(1), SquareModel()]

for m in models:
    print("Model Output:", m.predict(3))


# ----------- 7. MINI ML SYSTEM (FINAL BUILD 🔥) -----------

# Dataset
X = [1, 2, 3, 4]
y = [2, 4, 6, 8]

# Step 1: Create model
model = LinearModel(weight=1)

# Step 2: Train model
model.train(X, y)

# Step 3: Use pipeline
pipeline = Pipeline(model)

# Step 4: Run predictions
output = pipeline.run([5, 6, 7])

print("Trained Weight:", model.get_weight())
print("Final Predictions:", output)


# ----------- 8. OLD VS NEW COMPARISON -----------

# Old ML style (no structure ❌)
weight = 2
def old_predict(x):
    return weight * x

print("Old Style Output:", old_predict(5))

# New OOP ML style (structured ✔)
print("OOP Model Output:", model.predict(5)