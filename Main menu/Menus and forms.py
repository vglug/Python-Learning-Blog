def show_menu():
    print("\n===== MAIN MENU =====")
    print("1. Sign Up")
    print("2. Login")
    print("3. Exit")

# Function for user sign-up
def signup_form():
    print("\n--- Sign Up Form ---")
    name = input("Enter your name: ")        # Get user name
    email = input("Enter your email: ")      # Get user email
    password = input("Enter your password: ")# Get user password
    print(f"\n✅ Account created successfully for {name}!")  # Confirmation message

# Function for user login
def login_form():
    print("\n--- Login Form ---")
    email = input("Enter your email: ")      # Get login email
    password = input("Enter your password: ")# Get login password
    print(f"\n🔓 Welcome back, {email}!")    # Welcome message

# Main function to control program flow
def main():
    while True:  # Keep showing the menu until user exits
        show_menu()  # Display menu
        choice = input("Enter your choice (1-3): ")  # Get user's menu choice

        # Perform action based on user choice
        if choice == "1":
            signup_form()  # Call signup function
        elif choice == "2":
            login_form()   # Call login function
        elif choice == "3":
            print("\n👋 Exiting program. Goodbye!")  # Exit message
            break  # Stop the loop and exit program
        else:
            print("\n❌ Invalid choice, please try again.")  # Error message

# Run the program
if __name__ == "__main__":
    main()
