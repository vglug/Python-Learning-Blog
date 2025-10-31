def shop():
    price = int(input("Enter a Number: "))
    if price >= 1000 and price <= 2000:
        print("you got a 200 rs gift voucher")
    elif price >= 2000 and price <= 3000:
        print("you got a 400 rs gift voucher")
    elif price >= 3000 and price <= 4000:
        print("you got a 600 rs gift voucher")
    else:
        print("no offer")

shop()
