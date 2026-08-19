import pandas as pd
import numpy as np
import os

# =======================================================
# 1. template for company data
# =======================================================

Zenaki_columns = ['2023', '2024', '2025', '2026', 'latest']
Super_rows = ['Metric', 'Revenue', 'EPS', 'Net income', 'FCF', 'Debt', 'Cash', 'Market Cap']

Company_template = pd.DataFrame(index = Super_rows, columns = Zenaki_columns)

# =======================================================
# 2. Check if company file exists, if not create a new one
# =======================================================

User_input = input("Enter or Add company:")

def Search_validator():
    return(
        len(User_input) <= 5 and
        all(char.isalpha() for char in User_input) and User_input.isupper())

if Search_validator():
    if os.path.exists("Analysis (fund)/" + User_input + ".csv"):
        print("Company already exists. Loading data...")
        Company_template = pd.read_csv("Analysis (fund)/" + User_input + ".csv", index_col=0)
    else:
        print("Company file does not exist. Creating new file...")
        Company_template.to_csv("Analysis (fund)/" + User_input + ".csv")

# =======================================================
# 3. Allowing user to input data for the company
# =======================================================

