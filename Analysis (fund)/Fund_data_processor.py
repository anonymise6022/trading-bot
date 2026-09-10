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
spreadsheet_storage = "Analysis (fund)/" + User_input + ".csv"

# search validator function to check if the user input is valid
def Search_validator():
    return(
        len(User_input) <= 5 and
        all(char.isalpha() for char in User_input) and User_input.isupper())

# check if user already has a file for the company, if not create a new one
if Search_validator():
    if os.path.exists(spreadsheet_storage):
        print("Company already exists. Loading data...")
        Company_template = pd.read_csv(spreadsheet_storage, index_col=0)
    else:
        print("Creating new company file...")
        Company_template.to_csv(spreadsheet_storage)

# =======================================================
# 4. setting up for curses
# =======================================================

def draw_grid(stdscr, df, current_row, current_col, edit_buffer, editing):
    """Renders the DataFrame onto the curses window as an Excel-style grid."""
    stdscr.clear()
    height, width = stdscr.getmaxyx()
    
    # Grid formatting configuration
    cell_width = 12
    header_height = 2
    row_label_width = 5
    
    # 1. Render Column Headers
    stdscr.attron(curses.A_REVERSE)
    stdscr.addstr(0, 0, f"{'':>{row_label_width}}|")
    for c_idx, col_name in enumerate(df.columns):
        stdscr.addstr(0, row_label_width + 1 + (c_idx * cell_width), f"{str(col_name)[:cell_width-1]:^{cell_width-1}}|")
    stdscr.attroff(curses.A_REVERSE)
    
    # 2. Render Data Grid Matrix
    for r_idx in range(len(df)):
        if r_idx + header_height >= height - 2:  # Prevent rendering past screen bounds
            break
            
        # Draw Row Sidebar label
        stdscr.attron(curses.A_REVERSE)
        stdscr.addstr(r_idx + header_height, 0, f"{r_idx:^{row_label_width}}|")
        stdscr.attroff(curses.A_REVERSE)
        
        # Draw cells per row
        for c_idx in range(len(df.columns)):
            x_pos = row_label_width + 1 + (c_idx * cell_width)
            y_pos = r_idx + header_height
            
            # Extract content string
            val = str(df.iloc[r_idx, c_idx])
            
            # Handle Active Cursor Highlight vs Standard Text
            if r_idx == current_row and c_idx == current_col:
                if editing:
                    # While editing, show temporary buffer text with an underline
                    display_text = edit_buffer
                    stdscr.attron(curses.A_UNDERLINE)
                else:
                    display_text = val
                    stdscr.attron(curses.A_REVERSE)
            else:
                display_text = val
                
            # Crop string formatting to prevent grid bleed overflow
            formatted_text = f"{display_text[:cell_width-1]:<{cell_width-1}}|"
            stdscr.addstr(y_pos, x_pos, formatted_text)
            
            # Reset visual masks
            stdscr.attroff(curses.A_REVERSE)
            stdscr.attroff(curses.A_UNDERLINE)
    # 3. Render Bottom Status Guide Bar
    status_y = height - 1
    if editing:
        guide = "[MODE: EDITING] Type value. Press ENTER to submit, ESC to discard."
    else:
        guide = "[MODE: NAVIGATE] Arrows to move | ENTER to edit | Q to save & exit"
    stdscr.addstr(status_y, 0, guide[:width-1], curses.A_DIM)
    
    stdscr.refresh()
def excel_book(stdscr):
    # Initialize basic curses operational mechanics
    curses.curs_set(1)  # Enable visible blinking terminal cursor
    stdscr.keypad(True) # Translate escape sequences for arrow keys natively
    
    df = Company_template.copy()
    
    # State tracking indexes
    current_row = 0
    current_col = 0
    editing = False
    edit_buffer = ""
    
    while True:
        draw_grid(stdscr, df, current_row, current_col, edit_buffer, editing)
        key = stdscr.getch() # Stand by and poll for next input character
        
        if not editing:
            # --- NAVIGATION CONTROLS ---
            if key in [curses.KEY_UP, ord('w')]:
                current_row = max(0, current_row - 1)
            elif key in [curses.KEY_DOWN, ord('s')]:
                current_row = min(len(df) - 1, current_row + 1)
            elif key in [curses.KEY_LEFT, ord('a')]:
                current_col = max(0, current_col - 1)
            elif key in [curses.KEY_RIGHT, ord('d')]:
                current_col = min(len(df.columns) - 1, current_col + 1)
            elif key in [10, 13]:  # ENTER Key registry
                editing = True
                edit_buffer = str(df.iloc[current_row, current_col]) # Pre-fill buffer
            elif key in [ord('q'), ord('Q')]:
                Company_template.to_csv(spreadsheet_storage)
            break # Exit program loops
                
        else:
            # --- IN-CELL EDIT CONTROLS ---
            if key in [10, 13]:  # ENTER Key: commit text changes to pandas structure
                # Attempt numerical typecast fallback natively, otherwise leave string
                try:
                    if '.' in edit_buffer:
                        df.iloc[current_row, current_col] = float(edit_buffer)
                    else:
                        df.iloc[current_row, current_col] = int(edit_buffer)
                except ValueError:
                    df.iloc[current_row, current_col] = edit_buffer
                editing = False
                edit_buffer = ""
            elif key == 27:  # ESC Key: Abort operation and discard updates
                editing = False
                edit_buffer = ""
            elif key in [curses.KEY_BACKSPACE, 127, 8]: # Backspace handlers across terminal specs
                edit_buffer = edit_buffer[:-1]
            elif 32 <= key <= 126: # Accept printable ASCII character modifications
                edit_buffer += chr(key)
                
    return df
if __name__ == "__main__":
    # The curses wrapper safely boots the environment and handles terminal cleanup on crash
    print("\nTemplate being loaded into editor:")
    print(Company_template)

    input("\nPress ENTER to open the editor...")

    final_df = curses.wrapper(excel_book)
    
    # Print the resulting modified dataset back out to standard terminal output stream
    print("\n--- Final Saved DataFrame Output ---")
    print(final_df)





