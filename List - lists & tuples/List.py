my_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
print("Original list:", my_list)

print("\n---- List Slicing ----")

print("Slice from index 2 to 5:", my_list[2:6])

print("Slice from index 5 to end:", my_list[5:])

print("Slice from beginning to index 3:", my_list[:4])

print("Slice from index 1 to 8 with step 2:", my_list[1:9:2])

print("Reversed list:", my_list[::-1])

list_copy = my_list[:]
print("Shallow copy of the list:", list_copy)

print("\n--- Basic List Manipulations ---")

my_list[2:5] = [35, 45, 55]
print("After modifying elements at indices 2-4:", my_list)

my_list[5:5] = [58, 62]
print("After inserting elements at index 5:", my_list)

my_list[7:9] = []
print("After deleting elements at indices 7-8:", my_list)

my_list.append(110)
print("After appending 110:", my_list)

my_list.remove(20)
print("After removing 20:", my_list)

popped = my_list.pop(0)
print("After popping element at index 0:", my_list)
print("Popped element:", popped)

my_list.insert(2, 42)
print("After inserting 42 at index 2:", my_list)
