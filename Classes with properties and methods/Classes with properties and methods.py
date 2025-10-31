
class Car:
    
    def __init__(self, brand, model, year):
        self.brand = brand
        self.model = model
        self.year = year
        self.mileage = 0

    
    def display_details(self):
        print(f"Brand: {self.brand}")
        print(f"Model: {self.model}")
        print(f"Year: {self.year}")
        print(f"Mileage: {self.mileage} km")

    
    def update_mileage(self, new_mileage):
        self.mileage = new_mileage
        print(f"Mileage updated to {self.mileage} km")


def main():
    # Create an instance of the Car class
    my_car = Car('Toyota', 'Innova', 2020)
    my_car.display_details()
    my_car.update_mileage(5000)
    my_car.display_details()

if __name__ == "__main__":
    main()
