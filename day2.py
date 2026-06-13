# ==========================================
# DAY 2: Python Logic → ML Thinking
# ==========================================

# ----------- 1. INPUT + TYPE CONVERSION -----------
a, b = input("Enter two numbers: ").split()
a = float(a)
b = float(b)

# ----------- 2. ARITHMETIC OPERATORS -----------
print("Sum:", a + b)
print("Subtraction:", a - b)
print("Multiplication:", a * b)
print("Division:", a / b)
print("Modulus:", a % b)
print("Exponent:", a ** b)        # Fixed
print("Floor Division:", a // b)

# ----------- 3. COMPARISON OPERATORS -----------
print("a > b:", a > b)
print("a < b:", a < b)
print("a >= b:", a >= b)
print("a <= b:", a <= b)
print("a == b:", a == b)
print("a != b:", a != b)

print("===========================")

# ----------- 4. LOGICAL OPERATORS -----------
s = True
r = False

print("AND:", s and r)
print("OR:", s or r)
print("NOT r:", not r)
print("NOT s:", not s)

print("===========================")

# ----------- 5. BITWISE OPERATORS -----------
a = 10
b = 4

print("AND:", a & b)
print("OR:", a | b)
print("NOT:", ~a)
print("XOR:", a ^ b)
print("Right Shift:", a >> 2)
print("Left Shift:", a << 2)

print("===========================")

# ----------- 6. ASSIGNMENT OPERATORS -----------
print("Initial b:", b)
b += a
print("b += a:", b)

b -= a
print("b -= a:", b)

b *= a
print("b *= a:", b)

# ----------- 7. CONDITIONAL STATEMENTS -----------
age = int(input("Enter your age: "))

if age >= 18:
    print("Eligible to vote")
else:
    print("Not eligible")

# Nested condition
age = 70
is_member = True

if age >= 60:
    if is_member:
        print("30% senior discount")
    else:
        print("20% senior discount")
else:
    print("No discount")

# Multiple conditions
marks = int(input("Enter marks: "))

if marks >= 80:
    print("Grade A")
else:
    print("Grade B")

# Age classification
age = int(input("Enter age: "))

if age <= 12:
    print("Child")
elif age <= 19:
    print("Teenager")
elif age <= 35:
    print("Young Adult")
else:
    print("Adult")

# ----------- 8. LOOPS -----------
# For loop
for i in range(10):
    print("For Loop:", i)

# While loop
i = 0
while i < 5:
    print("While Loop:", i)
    i += 1

# ----------- 9. ENUMERATE -----------
fruits = ["orange", "apple", "banana", "mango"]

for index, fruit in enumerate(fruits, start=1):
    print(index, fruit)

# ----------- 10. ZIP -----------
names = ["Sai", "Ram", "John"]
marks = [90, 85, 88]

for name, mark in zip(names, marks):
    print(name, "scored", mark)

# ----------- 11. EARLY ML-LIKE LOGIC -----------
# Simple linear combination (core ML idea)

weights = [0.5, 0.3]
inputs = [2, 4]

output = 0
for w, x in zip(weights, inputs):
    output += w * x

print("Model Output:", output)

# Simulated training loop
predicted = output
actual = 10

for epoch in range(5):
    error = actual - predicted
    weights[0] += error * 0.01
    weights[1] += error * 0.01

print("Updated Weights:", weights)