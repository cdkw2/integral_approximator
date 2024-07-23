import tkinter as tk
from tkinter import ttk, messagebox
import math
from typing import Callable, Dict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
from scipy import special

class IntegralApproximationGUI:
    def __init__(self, master):
        self.master = master
        master.title("Integral Approximation")
        master.geometry("900x700")

        self.functions = self.get_functions()

        self.create_widgets()
        self.create_graph()

    def create_widgets(self):
        input_frame = ttk.Frame(self.master)
        input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        ttk.Label(input_frame, text="Lower limit (a):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.a_entry = ttk.Entry(input_frame)
        self.a_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="Upper limit (b):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.b_entry = ttk.Entry(input_frame)
        self.b_entry.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="Number of subintervals (n):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.n_entry = ttk.Entry(input_frame)
        self.n_entry.grid(row=2, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="Select function:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.function_var = tk.StringVar()
        self.function_dropdown = ttk.Combobox(input_frame, textvariable=self.function_var)
        self.function_dropdown['values'] = list(self.functions.keys())
        self.function_dropdown.grid(row=3, column=1, padx=5, pady=5)
        self.function_dropdown.current(0)

        self.calculate_button = ttk.Button(input_frame, text="Calculate", command=self.calculate)
        self.calculate_button.grid(row=4, column=0, columnspan=2, pady=10)

        self.result_label = ttk.Label(input_frame, text="Result:")
        self.result_label.grid(row=5, column=0, columnspan=2, pady=5)

        self.result_value = ttk.Label(input_frame, text="")
        self.result_value.grid(row=6, column=0, columnspan=2, pady=5)

    def create_graph(self):
        graph_frame = ttk.Frame(self.master)
        graph_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, graph_frame)
        self.toolbar.update()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

    def get_functions(self) -> Dict[str, Callable[[float], float]]:
        return {
            # Elementary functions
            "x^2": lambda x: x ** 2,
            "x^3": lambda x: x ** 3,
            "sin(x)": self.sin_taylor,
            "cos(x)": self.cos_taylor,
            "ln(x)": self.ln_approximation,
            "e^x": self.exp_taylor,
            
            # Non-elementary functions
            "erf(x)": special.erf,  # Error function
            "Γ(x)": math.gamma,  # Gamma function
            "Ai(x)": special.airy,  # Airy function
            "Ei(x)": special.expi,  # Exponential integral
            "Li(x)": special.spence,  # Dilogarithm
            "Bessel J₀(x)": special.j0,  # Bessel function of the first kind, order 0
            "Bessel Y₀(x)": special.y0,  # Bessel function of the second kind, order 0
            "sinc(x)": np.sinc,  # Sinc function
            "W(x)": special.lambertw,  # Lambert W function
            "zeta(x)": special.zeta,  # Riemann zeta function
        }

    def calculate(self):
        try:
            a = float(self.a_entry.get())
            b = float(self.b_entry.get())
            n = int(self.n_entry.get())

            if n <= 0:
                raise ValueError("n must be a positive integer")

            selected_function = self.functions[self.function_var.get()]
            result = self.trapezoidal_approximation(selected_function, a, b, n)
            self.result_value.config(text=f"{result:.7f}")

            self.plot_function(selected_function, a, b, n)
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def plot_function(self, func, a, b, n):
        self.ax.clear()
        
        def safe_func(x):
            try:
                return func(x)
            except (ValueError, ZeroDivisionError, OverflowError):
                return np.nan

        x = np.linspace(a, b, 1000)
        y = [safe_func(xi) for xi in x]
        
        x = x[np.isfinite(y)]
        y = np.array(y)[np.isfinite(y)]
        
        if len(x) > 0:
            self.ax.plot(x, y, 'b-', label='Function')
            self.draw_trapezoids(func, a, b, n)
            self.ax.set_title(f"Graph of {self.function_var.get()} with Trapezoidal Approximation")
            self.ax.set_xlabel("x")
            self.ax.set_ylabel("y")
            self.ax.grid(True)
            self.ax.legend()
            self.ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            
            y_range = np.ptp(y)
            y_mean = np.mean(y)
            self.ax.set_ylim(y_mean - 2*y_range, y_mean + 2*y_range)
        else:
            self.ax.text(0.5, 0.5, "Function not defined in this range", 
                         ha='center', va='center', transform=self.ax.transAxes)
        
        self.canvas.draw()

    def draw_trapezoids(self, func, a, b, n):
        h = (b - a) / n
        x = np.linspace(a, b, n+1)
        y = [func(xi) for xi in x]
        
        for i in range(n):
            y1, y2 = y[i], y[i+1]
            x1, x2 = x[i], x[i+1]
            
            if np.isnan(y1) or np.isnan(y2) or np.isinf(y1) or np.isinf(y2):
                continue
            
            if y1 >= 0 and y2 >= 0:
                color = 'g'
            elif y1 < 0 and y2 < 0:
                color = 'r'
            else:
                x_intersect = x1 + (0 - y1) * (x2 - x1) / (y2 - y1)
                self.ax.fill([x1, x1, x_intersect, x_intersect], [0, y1, 0, 0], 'g' if y1 > 0 else 'r', alpha=0.3)
                self.ax.fill([x_intersect, x_intersect, x2, x2], [0, 0, y2, 0], 'g' if y2 > 0 else 'r', alpha=0.3)
                continue
            
            self.ax.fill([x1, x1, x2, x2], [0, y1, y2, 0], color, alpha=0.3)
        
        self.ax.plot(x, y, 'bo', markersize=3)

    def trapezoidal_approximation(self, f: Callable[[float], float], a: float, b: float, n: int) -> float:
        h = (b - a) / n
        integral_sum = sum(f(a + i * h) for i in range(1, n))
        return (h / 2) * (f(a) + 2 * integral_sum + f(b))

    def ln_approximation(self, x: float) -> float:
        if x <= 0:
            raise ValueError("ln(x) is undefined for x <= 0")
        a = 10000
        return a * (x ** (1/a)) - a

    def sin_taylor(self, x: float, terms: int = 10) -> float:
        result = 0
        for n in range(terms):
            result += ((-1)**n * x**(2*n+1)) / math.factorial(2*n+1)
        return result

    def cos_taylor(self, x: float, terms: int = 10) -> float:
        result = 0
        for n in range(terms):
            result += ((-1)**n * x**(2*n)) / math.factorial(2*n)
        return result

    def exp_taylor(self, x: float, terms: int = 10) -> float:
        result = 0
        for n in range(terms):
            result += x**n / math.factorial(n)
        return result

if __name__ == "__main__":
    root = tk.Tk()
    app = IntegralApproximationGUI(root)
    root.mainloop()
