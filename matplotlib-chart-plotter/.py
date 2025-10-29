"""
Matplotlib Chart Plotter - Task #89
A comprehensive GUI application for plotting various charts from user data using matplotlib.
Supports multiple chart types: Line, Bar, Scatter, Pie, Histogram, and Box plots.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import json


class ChartPlotterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Matplotlib Chart Plotter - Task #89")
        self.root.geometry("1200x800")
        
        # Data storage
        self.data = {}
        self.current_chart = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Left panel - Controls
        control_frame = ttk.LabelFrame(main_frame, text="Chart Controls", padding="10")
        control_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Chart type selection
        ttk.Label(control_frame, text="Chart Type:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.chart_type = ttk.Combobox(control_frame, state="readonly", width=20)
        self.chart_type['values'] = ('Line Chart', 'Bar Chart', 'Scatter Plot', 
                                       'Pie Chart', 'Histogram', 'Box Plot')
        self.chart_type.current(0)
        self.chart_type.grid(row=0, column=1, pady=5)
        self.chart_type.bind('<<ComboboxSelected>>', self.on_chart_type_change)
        
        # Data input section
        ttk.Label(control_frame, text="Data Input", font=('', 10, 'bold')).grid(
            row=1, column=0, columnspan=2, sticky=tk.W, pady=(15, 5))
        
        # X-axis data
        ttk.Label(control_frame, text="X-axis data:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.x_data_entry = ttk.Entry(control_frame, width=25)
        self.x_data_entry.grid(row=2, column=1, pady=5)
        self.x_data_entry.insert(0, "1,2,3,4,5")
        
        # Y-axis data
        ttk.Label(control_frame, text="Y-axis data:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.y_data_entry = ttk.Entry(control_frame, width=25)
        self.y_data_entry.grid(row=3, column=1, pady=5)
        self.y_data_entry.insert(0, "2,4,6,8,10")
        
        # Labels section
        ttk.Label(control_frame, text="Labels & Titles", font=('', 10, 'bold')).grid(
            row=4, column=0, columnspan=2, sticky=tk.W, pady=(15, 5))
        
        # Chart title
        ttk.Label(control_frame, text="Title:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.title_entry = ttk.Entry(control_frame, width=25)
        self.title_entry.grid(row=5, column=1, pady=5)
        self.title_entry.insert(0, "My Chart")
        
        # X-axis label
        ttk.Label(control_frame, text="X-axis label:").grid(row=6, column=0, sticky=tk.W, pady=5)
        self.xlabel_entry = ttk.Entry(control_frame, width=25)
        self.xlabel_entry.grid(row=6, column=1, pady=5)
        self.xlabel_entry.insert(0, "X Values")
        
        # Y-axis label
        ttk.Label(control_frame, text="Y-axis label:").grid(row=7, column=0, sticky=tk.W, pady=5)
        self.ylabel_entry = ttk.Entry(control_frame, width=25)
        self.ylabel_entry.grid(row=7, column=1, pady=5)
        self.ylabel_entry.insert(0, "Y Values")
        
        # Color selection
        ttk.Label(control_frame, text="Color:").grid(row=8, column=0, sticky=tk.W, pady=5)
        self.color_combo = ttk.Combobox(control_frame, state="readonly", width=20)
        self.color_combo['values'] = ('blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta')
        self.color_combo.current(0)
        self.color_combo.grid(row=8, column=1, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=9, column=0, columnspan=2, pady=20)
        
        ttk.Button(button_frame, text="Plot Chart", command=self.plot_chart).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear", command=self.clear_plot).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save", command=self.save_chart).pack(side=tk.LEFT, padx=5)
        
        # Load/Save data buttons
        data_button_frame = ttk.Frame(control_frame)
        data_button_frame.grid(row=10, column=0, columnspan=2, pady=5)
        
        ttk.Button(data_button_frame, text="Load Data", command=self.load_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(data_button_frame, text="Save Data", command=self.save_data).pack(side=tk.LEFT, padx=5)
        
        # Help text
        help_text = ("Enter comma-separated values.\n"
                    "Example: 1,2,3,4,5\n"
                    "For pie charts, use Y-axis only.\n"
                    "For histograms, use X-axis only.")
        help_label = ttk.Label(control_frame, text=help_text, font=('', 8), 
                              foreground='gray', justify=tk.LEFT)
        help_label.grid(row=11, column=0, columnspan=2, pady=10, sticky=tk.W)
        
        # Right panel - Chart display
        chart_frame = ttk.LabelFrame(main_frame, text="Chart Display", padding="10")
        chart_frame.grid(row=0, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Matplotlib figure
        self.figure = plt.Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def on_chart_type_change(self, event=None):
        """Update UI hints based on chart type"""
        chart_type = self.chart_type.get()
        
        if chart_type == 'Pie Chart':
            self.x_data_entry.delete(0, tk.END)
            self.x_data_entry.insert(0, "A,B,C,D")
            self.y_data_entry.delete(0, tk.END)
            self.y_data_entry.insert(0, "30,25,20,25")
        elif chart_type == 'Histogram':
            self.x_data_entry.delete(0, tk.END)
            self.x_data_entry.insert(0, "1,2,2,3,3,3,4,4,5")
        else:
            self.x_data_entry.delete(0, tk.END)
            self.x_data_entry.insert(0, "1,2,3,4,5")
            self.y_data_entry.delete(0, tk.END)
            self.y_data_entry.insert(0, "2,4,6,8,10")
    
    def parse_data(self, data_string):
        """Parse comma-separated data string"""
        try:
            # Try to parse as numbers
            return [float(x.strip()) for x in data_string.split(',') if x.strip()]
        except ValueError:
            # Return as strings if not numbers
            return [x.strip() for x in data_string.split(',') if x.strip()]
    
    def plot_chart(self):
        """Plot the selected chart type with user data"""
        try:
            chart_type = self.chart_type.get()
            
            # Parse data
            x_data = self.parse_data(self.x_data_entry.get())
            y_data = self.parse_data(self.y_data_entry.get())
            
            # Get labels and settings
            title = self.title_entry.get()
            xlabel = self.xlabel_entry.get()
            ylabel = self.ylabel_entry.get()
            color = self.color_combo.get()
            
            # Clear previous plot
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            # Plot based on chart type
            if chart_type == 'Line Chart':
                if not y_data:
                    raise ValueError("Y-axis data required for line chart")
                ax.plot(x_data if len(x_data) == len(y_data) else range(len(y_data)), 
                       y_data, color=color, marker='o', linewidth=2, markersize=6)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.3)
                
            elif chart_type == 'Bar Chart':
                if not y_data:
                    raise ValueError("Y-axis data required for bar chart")
                x_pos = x_data if len(x_data) == len(y_data) else range(len(y_data))
                ax.bar(x_pos, y_data, color=color, alpha=0.7, edgecolor='black')
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.3, axis='y')
                
            elif chart_type == 'Scatter Plot':
                if not y_data:
                    raise ValueError("Y-axis data required for scatter plot")
                if len(x_data) != len(y_data):
                    raise ValueError("X and Y data must have same length for scatter plot")
                ax.scatter(x_data, y_data, color=color, s=100, alpha=0.6, edgecolors='black')
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.3)
                
            elif chart_type == 'Pie Chart':
                if not y_data:
                    raise ValueError("Y-axis data required for pie chart")
                labels = x_data if x_data else [f'Item {i+1}' for i in range(len(y_data))]
                ax.pie(y_data, labels=labels, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                
            elif chart_type == 'Histogram':
                if not x_data:
                    raise ValueError("X-axis data required for histogram")
                ax.hist(x_data, bins=10, color=color, alpha=0.7, edgecolor='black')
                ax.set_xlabel(xlabel)
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3, axis='y')
                
            elif chart_type == 'Box Plot':
                if not y_data:
                    y_data = x_data
                ax.boxplot([y_data], labels=['Data'], patch_artist=True,
                          boxprops=dict(facecolor=color, alpha=0.7))
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.3, axis='y')
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            self.figure.tight_layout()
            self.canvas.draw()
            
            messagebox.showinfo("Success", "Chart plotted successfully!")
            
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid data: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot chart: {str(e)}")
    
    def clear_plot(self):
        """Clear the current plot"""
        self.figure.clear()
        self.canvas.draw()
    
    def save_chart(self):
        """Save the current chart as an image"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), 
                          ("SVG files", "*.svg"), ("All files", "*.*")]
            )
            if filename:
                self.figure.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Chart saved to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save chart: {str(e)}")
    
    def save_data(self):
        """Save current data configuration to JSON file"""
        try:
            data = {
                'chart_type': self.chart_type.get(),
                'x_data': self.x_data_entry.get(),
                'y_data': self.y_data_entry.get(),
                'title': self.title_entry.get(),
                'xlabel': self.xlabel_entry.get(),
                'ylabel': self.ylabel_entry.get(),
                'color': self.color_combo.get()
            }
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if filename:
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=4)
                messagebox.showinfo("Success", f"Data saved to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save data: {str(e)}")
    
    def load_data(self):
        """Load data configuration from JSON file"""
        try:
            filename = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if filename:
                with open(filename, 'r') as f:
                    data = json.load(f)
                
                # Set values
                self.chart_type.set(data.get('chart_type', 'Line Chart'))
                self.x_data_entry.delete(0, tk.END)
                self.x_data_entry.insert(0, data.get('x_data', ''))
                self.y_data_entry.delete(0, tk.END)
                self.y_data_entry.insert(0, data.get('y_data', ''))
                self.title_entry.delete(0, tk.END)
                self.title_entry.insert(0, data.get('title', ''))
                self.xlabel_entry.delete(0, tk.END)
                self.xlabel_entry.insert(0, data.get('xlabel', ''))
                self.ylabel_entry.delete(0, tk.END)
                self.ylabel_entry.insert(0, data.get('ylabel', ''))
                self.color_combo.set(data.get('color', 'blue'))
                
                messagebox.showinfo("Success", f"Data loaded from {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")


def main():
    root = tk.Tk()
    app = ChartPlotterGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
