def show_menu():
    print("\n===== MAIN MENU =====")
    print("1. Sign Up")
    print("2. Login")
    print("3. Exit")

def signup_form():
    print("\n--- Sign Up Form ---")
    name = input("Enter your name: ")
    email = input("Enter your email: ")
    password = input("Enter your password: ")
    print(f"\n✅ Account created successfully for {name}!")

def login_form():
    print("\n--- Login Form ---")
    email = input("Enter your email: ")
    password = input("Enter your password: ")
    print(f"\n🔓 Welcome back, {email}!")

def main():
    while True:
        show_menu()
        choice = input("Enter your choice (1-3): ")

        if choice == "1":
            signup_form()
        elif choice == "2":
            login_form()
        elif choice == "3":
            print("\n👋 Exiting program. Goodbye!")
            break
        else:
            print("\n❌ Invalid choice, please try again.")
if __name__ == "__main__":
    main()
