def shop():
    price = int(input("Enter a price:"))  # Take price input from user

    if price >= 1000 and price <= 2000:    # Check if price is between 1000 and 2000
        print("You got a 200 rs gift voucher")
    elif price >= 2000 and price <= 3000:  # Check if price is between 2000 and 3000
        print("You got a 400 rs gift voucher")
    elif price >= 3000 and price <= 4000:  # Check if price is between 3000 and 4000
        print("You got a 600 rs gift voucher")
    else:                                  # If price is outside all ranges
        print("No gift voucher")
        
shop()  # Call the function
