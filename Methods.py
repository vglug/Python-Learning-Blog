class Car:

    def __init__(self, brand, model, year):
        self.brand = brand    
        self.model = model    
        self.year = year   


    def display_details(self):
        print(f"Car Brand: {self.brand}")
        print(f"Model: {self.model}")
        print(f"Year: {self.year}")

    
    def start_engine(self):
        print(f"The {self.brand} {self.model}'s engine has started!")

    
    def stop_engine(self):
        print(f"The {self.brand} {self.model}'s engine has stopped.")


car1 = Car("Toyota", "Innova", 2020)
car2 = Car("Hyundai", "Creta", 2023)


car1.display_details()
car1.start_engine()
car1.stop_engine()

print()  

car2.display_details()
car2.start_engine()
car2.stop_engine()