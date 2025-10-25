# -------------------------------
# Author: Priyadharshni G
# Project: Simple Calculator using Tkinter
# Description: A basic calculator GUI that performs arithmetic operations.
# -------------------------------

import tkinter as tk  # Importing the Tkinter module for GUI creation

# Function to handle number and operator button presses
def press(key):
    entry_var.set(entry_var.get() + str(key))  # Append the pressed key to the display

# Function to evaluate the expression and display the result
def equal():
    try:
        expr = entry_var.get()  # Get the current expression from the entry field
        if expr.strip() == "":  # Check if entry is empty
            return
        result = str(eval(expr))  # Evaluate the mathematical expression
        entry_var.set(result)     # Display the result
    except:
        entry_var.set("Error")    # Display error if invalid expression

# Function to clear the calculator display
def clear():
    entry_var.set("")  # Reset the entry field

# Initialize the main Tkinter window
root = tk.Tk()
root.title("Simple Calculator")     # Window title
root.geometry("320x450")            # Window size
root.resizable(True, True)          # Allow resizing

entry_var = tk.StringVar()  # Variable to hold text displayed in the entry field

# Entry widget to display input and output
entry = tk.Entry(root, textvariable=entry_var, font=('Helvetica', 24), bd=5,
                 relief=tk.RIDGE, justify='right')
entry.grid(row=0, column=0, columnspan=4, pady=20, padx=10, ipadx=5, ipady=10)

# Calculator button layout
buttons = [
    ['7', '8', '9', '/'],
    ['4', '5', '6', '*'],
    ['1', '2', '3', '-'],
    ['0', '.', '=', '+'],
]

# Creating and placing all buttons dynamically
for i, row in enumerate(buttons, start=1):
    for j, button in enumerate(row):
        # Different color for operation buttons
        color = 'orange' if button in ['/', '*', '-', '+', '='] else 'lightgray'
        fg_color = 'white' if color == 'orange' else 'black'

        # Assign appropriate function for each button
        action = equal if button == '=' else lambda key=button: press(key)

        # Create button widget
        b = tk.Button(root, text=button, font=('Helvetica', 20), bg=color,
                      fg=fg_color, command=action, bd=1, relief=tk.RAISED)
        b.grid(row=i, column=j, sticky='nsew', padx=5, pady=5, ipadx=5, ipady=15)

# Clear button spanning all columns
clear_button = tk.Button(root, text='C', font=('Helvetica', 20), bg='red',
                         fg='white', command=clear)
clear_button.grid(row=5, column=0, columnspan=4, sticky='nsew', padx=5, pady=10, ipady=15)

# Adjusting row and column weights for responsive layout
for i in range(5):
    root.grid_rowconfigure(i, weight=1)
for j in range(4):
    root.grid_columnconfigure(j, weight=1)

# Run the main event loop
root.mainloop()
