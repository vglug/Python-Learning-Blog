import socket
import sys
from datetime import datetime
import argparse


def scan_port(target_ip, port, timeout=1):
    """
    Attempt to connect to a specific port on the target IP.
    
    Args:
        target_ip (str): The IP address to scan
        port (int): The port number to check
        timeout (float): Connection timeout in seconds
    
    Returns:
        bool: True if port is open, False otherwise
    """
    try:
        # Create a socket object
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)

        # Attempt to connect to the port
        result = sock.connect_ex((target_ip, port))
        sock.close()

        # Return True if connection successful (port is open)
        return result == 0
    except socket.gaierror:
        print(f"Error: Hostname could not be resolved")
        return False
    except socket.error:
        print(f"Error: Could not connect to server")
        return False


def get_service_name(port):
    """
    Try to get the common service name for a port number.
    
    Args:
        port (int): Port number
    
    Returns:
        str: Service name or "Unknown"
    """
    try:
        return socket.getservbyport(port)
    except:
        return "Unknown"


def scan_ports(target, port_range, timeout=1):
    """
    Scan a range of ports on the target host.
    
    Args:
        target (str): Target hostname or IP address
        port_range (tuple): Tuple of (start_port, end_port)
        timeout (float): Connection timeout in seconds
    """
    start_port, end_port = port_range

    # Resolve target hostname to IP
    try:
        target_ip = socket.gethostbyname(target)
    except socket.gaierror:
        print(f"Error: Cannot resolve hostname '{target}'")
        return

    print("-" * 60)
    print(f"Scanning Target: {target}")
    print(f"Target IP: {target_ip}")
    print(f"Port Range: {start_port}-{end_port}")
    print(f"Scan started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)

    open_ports = []

    # Scan each port in the range
    for port in range(start_port, end_port + 1):
        if scan_port(target_ip, port, timeout):
            service = get_service_name(port)
            print(f"Port {port:5d} - OPEN    [{service}]")
            open_ports.append((port, service))

    print("-" * 60)
    print(f"Scan completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total open ports found: {len(open_ports)}")
    print("-" * 60)


def main():
    """Main function to parse arguments and run the scanner."""
    parser = argparse.ArgumentParser(
        description="Simple Port Scanner for Educational Use",
        epilog="WARNING: Only scan systems you own or have permission to test!"
    )

    parser.add_argument(
        "target",
        help="Target hostname or IP address to scan"
    )

    parser.add_argument(
        "-s", "--start",
        type=int,
        default=1,
        help="Starting port number (default: 1)"
    )

    parser.add_argument(
        "-e", "--end",
        type=int,
        default=1024,
        help="Ending port number (default: 1024)"
    )

    parser.add_argument(
        "-t", "--timeout",
        type=float,
        default=1.0,
        help="Connection timeout in seconds (default: 1.0)"
    )

    args = parser.parse_args()

    # Validate port range
    if args.start < 1 or args.end > 65535 or args.start > args.end:
        print("Error: Invalid port range. Ports must be between 1-65535")
        sys.exit(1)

    # Display warning
    print("\n" + "=" * 60)
    print("EDUCATIONAL PORT SCANNER")
    print("=" * 60)
    print("WARNING: Unauthorized port scanning may be illegal.")
    print("Only scan systems you own or have explicit permission to test.")
    print("=" * 60 + "\n")

    try:
        scan_ports(args.target, (args.start, args.end), args.timeout)
    except KeyboardInterrupt:
        print("\n\nScan interrupted by user. Exiting...")
        sys.exit(0)


if __name__ == "__main__":
    main()
