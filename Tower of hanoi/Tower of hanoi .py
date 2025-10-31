def tower_of_hanoi(n, source, destination, auxiliary):
    if n == 1:
        print(f"Move disk 1 from {source} to {destination}")
        return
    tower_of_hanoi(n-1, source, auxiliary, destination)
    print(f"Move disk {n} from {source} to {destination}")
    tower_of_hanoi(n-1, auxiliary, destination, source)

def main():
    n = 3 
    tower_of_hanoi(n, 'A', 'C', 'B')
    print(f"Tower of Hanoi solved for {n} disks.")

if __name__ == "__main__":
    main()
