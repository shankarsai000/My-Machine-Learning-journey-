# ==========================================
# CLI Expense Tracker (OOP + File I/O)
# ==========================================

import json
import os

# ----------- 1. EXPENSE CLASS -----------
class Expense:
    def __init__(self, amount, category, note=""):
        self.amount = amount
        self.category = category
        self.note = note

    def to_dict(self):
        return {
            "amount": self.amount,
            "category": self.category,
            "note": self.note
        }


# ----------- 2. TRACKER CLASS -----------
class ExpenseTracker:
    def __init__(self, file_name="expenses.json"):
        self.file_name = file_name
        self.expenses = self.load_data()

    # Load data from file
    def load_data(self):
        if os.path.exists(self.file_name):
            with open(self.file_name, "r") as f:
                return json.load(f)
        return []

    # Save data to file
    def save_data(self):
        with open(self.file_name, "w") as f:
            json.dump(self.expenses, f, indent=4)

    # Add expense
    def add_expense(self, expense):
        self.expenses.append(expense.to_dict())
        self.save_data()
        print("✅ Expense added")

    # Show all expenses
    def show_expenses(self):
        if not self.expenses:
            print("No expenses found")
            return
        
        for i, exp in enumerate(self.expenses, 1):
            print(f"{i}. ₹{exp['amount']} | {exp['category']} | {exp['note']}")

    # Summary by category
    def summary(self):
        summary = {}

        for exp in self.expenses:
            cat = exp["category"]
            summary[cat] = summary.get(cat, 0) + exp["amount"]

        print("\n📊 Expense Summary:")
        for cat, total in summary.items():
            print(f"{cat}: ₹{total}")


# ----------- 3. CLI MENU -----------
def run():
    tracker = ExpenseTracker()

    while True:
        print("\n--- Expense Tracker ---")
        print("1. Add Expense")
        print("2. Show Expenses")
        print("3. Summary")
        print("4. Exit")

        choice = input("Enter choice: ")

        if choice == "1":
            amount = float(input("Amount: "))
            category = input("Category: ")
            note = input("Note: ")

            exp = Expense(amount, category, note)
            tracker.add_expense(exp)

        elif choice == "2":
            tracker.show_expenses()

        elif choice == "3":
            tracker.summary()

        elif choice == "4":
            print("Goodbye 👋")
            break

        else:
            print("Invalid choice")


# ----------- 4. START APP -----------
if __name__ == "__main__":
    run()