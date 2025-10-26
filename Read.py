# Program to read and write small text files

# Writing to a text file
with open("sample.txt", "w") as file:
    file.write("Hello, this is a simple text file.\n")
    file.write("We are learning how to read and write files in Python.\n")
    file.write("File handling is easy and powerful!\n")

print("Data has been written to sample.txt successfully.\n")

# Reading from the same text file
with open("sample.txt", "r") as file:
    content = file.read()

print("Contents of the file are:\n")
print(content)
