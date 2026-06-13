'''
# -------------------------------
# DAY 1: Python Foundations
# -------------------------------

# 1. Printing output
print("Hello World")  # Basic output statement


# 2. Taking user input
name = input("What is your name? ")
print("Hello " + name)  # Concatenating string with input


# 3. Taking another input
age = input("What is your age? ")
print("Your age is " + age)


# 4. Variables and Data Types
name = "Sai"          # String
age = 20              # Integer
city = "Bellary"      # String

# Printing multiple variables
print("My name is " + name + " and my age is " + str(age) + " and I am from " + city)


# 5. Checking data types
print(type(name))     # <class 'str'>
print(type(age))      # <class 'int'>
print(type(city))     # <class 'str'>


# 6. Taking two numbers as input
print("ENTER THE TWO NUMBERS")

num1 = int(input("Enter the first number: "))   # Convert input to integer
num2 = int(input("Enter the second number: "))

# Performing addition
sum = num1 + num2
print("Sum of num1 and num2 is " + str(sum))


# 7. Type Conversion
a = 12
print(float(a))       # Convert int → float

b = 3.21122
print(int(b))         # Convert float → int (cuts decimal)


# 8. String operations
a = "Sai"
print(len(a))         # Length of string
print(type(a))        # Type of variable
print(id(a))          # Memory address (advanced concept)

# 9. Mini Logic (Even or Odd)
num = int(input("Enter a number: "))

if num % 2 == 0:
    print("Even number")
else:
    print("Odd number")
    '''
 

