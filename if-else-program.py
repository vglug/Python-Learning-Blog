def shop():
    price = int(input("Enter a price:")) # Ask the user to enter the price of the item

    if price >= 1000 and price <= 2000: # If the price is between 1000 and 2000, give a 200 rs voucher
        print("You got a 200 rs gift voucher")
    
    elif price >= 2000 and price <= 3000:  # If the price is between 2000 and 3000, give a 400 rs voucher
        print("You got a 400 rs gift voucher")
    
    elif price >= 3000 and price <= 4000:   # If the price is between 3000 and 4000, give a 600 rs voucher
        print("You got a 600 rs gift voucher")
    
    else:  # If the price is not in any of the above, no voucher
        print("No gift voucher")

shop() #run the shop function
