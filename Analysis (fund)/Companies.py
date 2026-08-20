import pandas as pd
import numpy as np
import os
import curses

# =======================================================
# 1. template for company data
# =======================================================

Zenaki_columns = ['2023', '2024', '2025', '2026', 'latest']
Super_rows = ['Metric', 'Revenue', 'EPS', 'Net income', 'FCF', 'Debt', 'Cash', 'Market Cap']

Company_template = pd.DataFrame(index = Super_rows, columns = Zenaki_columns)

# ==============================================================================================================
# 2 / 3 . Check if company file exists, if not create a new one / Allowing user to input data for the company
# ==============================================================================================================

User_input = input("Enter or Add company:")

# search validator function to check if the user input is valid
def Search_validator():
    return(
        len(User_input) <= 5 and
        all(char.isalpha() for char in User_input) and User_input.isupper())

# check if user already has a file for the company, if not create a new one
if Search_validator():
    if os.path.exists("Analysis (fund)/" + User_input + ".csv"):
        print("Company already exists. Loading data...")
        Company_template = pd.read_csv("Analysis (fund)/" + User_input + ".csv", index_col=0)

        # show the current data to the user
        print(Company_template)

        # allow the user to edit the data
        row = input("Enter the row you want to edit (Metric, Revenue, EPS, Net income, FCF, Debt, Cash, Market Cap): ")
        col = input("Enter the column you want to edit (2023, 2024, 2025, 2026, latest): ")
        new_value = input("Enter the new value: ")

        # update the value in the DataFrame
        Company_template.loc[row, col] = new_value
        Company_template.to_csv("Analysis (fund)/" + User_input + ".csv")

    else:
        print("Company file does not exist. Creating new file...")
        Company_template.to_csv("Analysis (fund)/" + User_input + ".csv")

