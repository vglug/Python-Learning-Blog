class Animal:
    def __init__(self, name):
        self.name = name
    def speak(self):
        return "The animal makes a sound."
class Dog(Animal):
    def speak(self):
        return f"{self.name} says Woof!"
class Cat(Animal):
    def speak(self):
        return f"{self.name} says Meow!"
class Cow(Animal):
    def speak(self):
        return f"{self.name} says Moo!"
def main():
    dog = Dog("Buddy")
    cat = Cat("Whiskers")
    cow = Cow("Daisy")
    animals = [dog, cat, cow]
    for animal in animals:
        print(animal.speak())
if __name__ == "__main__":
    main()
