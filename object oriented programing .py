# Task #32: Inheritance and Method Overriding Example

class Animal:
    def __init__(self, name):
        self.name = name

    def sound(self):
        return "Some generic animal sound"

# Dog inherits from Animal
class Dog(Animal):
    def sound(self):
        return "Woof! Woof!"

# Cat inherits from Animal
class Cat(Animal):
    def sound(self):
        return "Meow!"

# Demonstration
animals = [Dog("Buddy"), Cat("Kitty"), Animal("Creature")]

for animal in animals:
    print(f"{animal.name} says: {animal.sound()}")
