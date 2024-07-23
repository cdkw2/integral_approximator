import tkinter as tk
from tkinter import ttk, messagebox
import math
from typing import Callable, Dict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

TAYLOR_TERMS = 20

class IntegralApproximationGUI:
    def __init__(self, master):
        self.master = master
        master.title("Integral Approximation")
        master.geometry("900x600")

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
        self.function_dropdown['values'] = ["x^3", "x^4", "x^11", "x^23", "sin(x)", "cos(x)", "tan(x)", "ln(x)", "e^x"]
        self.function_dropdown.grid(row=3, column=1, padx=5, pady=5)
        self.function_dropdown.current(0)

        self.calculate_button = ttk.Button(input_frame, text="Calculate", command=self.calculate)
        self.calculate_button.grid(row=4, column=0, columnspan=2, pady=10)

        self.result_label = ttk.Label(input_frame, text="Result:")
        self.result_label.grid(row=5, column=0, columnspan=2, pady=5)

        self.result_value = ttk.Label(input_frame, text="")
        self.result_value.grid(row=6, column=0, columnspan=2, pady=5)

    def create_graph(self):
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        

    def get_functions(self) -> Dict[str, Callable[[float], float]]:
        return {
            "x^3": lambda x: x ** 3,
            "x^4": lambda x: x ** 4,
            "x^11": lambda x: x ** 11,
            "x^23": lambda x: x ** 23,
            "sin(x)": self.sin_taylor,
            "cos(x)": self.cos_taylor,
            "tan(x)": self.tan_taylor,
            "ln(x)": self.ln_taylor,
            "e^x": self.exp_taylor
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

            self.plot_function(selected_function, a, b)
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def plot_function(self, func, a, b):
        self.ax.clear()
        x = np.linspace(a, b, 1000)
        y = [func(xi) for xi in x]
        self.ax.plot(x, y)
        self.ax.set_title(f"Graph of {self.function_var.get()}")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.grid(True)
        self.canvas.draw()

    def trapezoidal_approximation(self, f: Callable[[float], float], a: float, b: float, n: int) -> float:
        h = (b - a) / n
        integral_sum = sum(f(a + i * h) for i in range(1, n))
        return (h / 2) * (f(a) + 2 * integral_sum + f(b))

    def sin_taylor(self, x: float) -> float:
        return sum((-1) ** i * (x ** (2 * i + 1)) / math.factorial(2 * i + 1) for i in range(TAYLOR_TERMS))

    def cos_taylor(self, x: float) -> float:
        return sum((-1) ** i * (x ** (2 * i)) / math.factorial(2 * i) for i in range(TAYLOR_TERMS))

    def tan_taylor(self, x: float) -> float:
        return self.sin_taylor(x) / self.cos_taylor(x)

    def ln_taylor(self, x: float) -> float:
        if x <= 0:
            raise ValueError("ln(x) is undefined for x <= 0")
        return sum((-1) ** (i - 1) * (x - 1) ** i / i for i in range(1, TAYLOR_TERMS + 1))

    def exp_taylor(self, x: float) -> float:
        return sum(x ** i / math.factorial(i) for i in range(TAYLOR_TERMS))

if __name__ == "__main__":
    root = tk.Tk()
    app = IntegralApproximationGUI(root)
    root.mainloop()
