# ==========================================
# DAY 3: Functions + Data Thinking (ML Core)
# ==========================================

# ----------- 1. BASIC FUNCTIONS -----------
def add(a, b):
    return a + b

print("Sum:", add(5, 10))


def square(x):
    return x * x

print("Square:", square(4))


def is_even(num):
    return num % 2 == 0

print("Is Even (10):", is_even(10))
print("Is Even (7):", is_even(7))


# ----------- 2. RECURSION (FACTORIAL) -----------
def factorial(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)

print("Factorial:", factorial(5))


# ----------- 3. *args (Dynamic Inputs) -----------
def add_numbers(*args):
    total = 0
    for num in args:
        total += num
    return total

print("Sum using *args:", add_numbers(1, 2, 3, 4, 5))


def average(*numbers):
    return sum(numbers) / len(numbers)

print("Average:", average(10, 20, 30, 40))


# ----------- 4. **kwargs (Flexible Config) -----------
def train_model(**params):
    lr = params.get("lr", 0.01)
    epochs = params.get("epochs", 5)
    print(f"Training with lr={lr}, epochs={epochs}")

train_model(lr=0.001, epochs=10)


# ----------- 5. COMBINING args + kwargs -----------
def demo(a, *args, **kwargs):
    print("a:", a)
    print("args:", args)
    print("kwargs:", kwargs)

demo(1, 2, 3, x=10, y=20)


# ----------- 6. UNPACKING (IMPORTANT IN ML) -----------
numbers = [1, 2, 3]
print("Unpacked list:", *numbers)


# ----------- 7. SCOPE (LEGB RULE) -----------
def outer():
    x = 10

    def inner():
        print("Accessing enclosing scope:", x)

    inner()

outer()


# ----------- 8. CLOSURE (ML TRANSFORMATION) -----------
def make_scaler(factor):
    def scale(x):
        return x * factor
    return scale

scale_by_10 = make_scaler(10)
print("Scaled value:", scale_by_10(5))


# ----------- 9. LAMBDA FUNCTIONS -----------
add_lambda = lambda a, b: a + b
print("Lambda Add:", add_lambda(2, 3))


# ----------- 10. SORTING USING LAMBDA -----------
data = [(1, 5), (2, 3), (4, 1)]
sorted_data = sorted(data, key=lambda x: x[1])
print("Sorted Data:", sorted_data)


# ----------- 11. MAP (TRANSFORMATION) -----------
nums = [1, 2, 3]
squared = list(map(lambda x: x**2, nums))
print("Squared:", squared)


# ----------- 12. FILTER (SELECTION) -----------
nums = [1, 2, 3, 4]
even = list(filter(lambda x: x % 2 == 0, nums))
print("Even Numbers:", even)


# ----------- 13. MINI ML-LIKE PIPELINE -----------
# Inputs (features) and outputs (labels)
X = [1, 2, 3]
y = [2, 4, 6]

# Model (function)
def model(x, w):
    return x * w

# Training simulation
w = 1

for x_val, target in zip(X, y):
    pred = model(x_val, w)
    error = target - pred
    w += error * 0.1

print("Trained Weight:", w)