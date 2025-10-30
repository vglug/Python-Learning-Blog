class Car:
    wheels = 4
    def __init__(self, brand, model, year):
        self.brand = brand
        self.model = model
        self.year = year
        self._mileage = 0  
    def drive(self, km):
        """Increase mileage when the car is driven."""
        self._mileage += km
        print(f"{self.brand} {self.model} drove {km} km.")
    @property
    def mileage(self):
        """Get the car's mileage."""
        return self._mileage
    @mileage.setter
    def mileage(self, value):
        """Set the car's mileage, but prevent lowering it."""
        if value < self._mileage:
            print("Error: mileage cannot be decreased!")
        else:
            self._mileage = value
    @classmethod
    def change_wheels(cls, number):
        """Change number of wheels for all cars."""
        cls.wheels = number

    @staticmethod
    def honk():
        print("Beep beep! ")


car1 = Car("Toyota", "Camry", 2022)
car2 = Car("Tesla", "Model 3", 2023)

car1.drive(100)
car2.drive(50)

print(car1.mileage) 
car1.mileage = 200   
car1.mileage = 50   

print(f"All cars have {Car.wheels} wheels.")

Car.change_wheels(6)
print(f"Now All cars have {Car.wheels} wheels.")

Car.honk()
