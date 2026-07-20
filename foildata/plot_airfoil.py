import tkinter as tk
from tkinter import filedialog, messagebox
import os
import re

class AirfoilPlotter:
    def __init__(self, root):
        self.root = root
        self.root.title("翼型可视化工具")
        self.root.geometry("1200x900")

        self.open_button = tk.Button(
            self.root,
            text="打开 .dat 文件",
            command=self.open_airfoil_file,
            font=("Microsoft YaHei", 12),
        )
        self.open_button.pack(pady=(10, 0))

        self.canvas = tk.Canvas(self.root, background="white", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        self.label = tk.Label(self.root, text="请选择 .dat 翼型文件", font=("Microsoft YaHei", 12), pady=10)
        self.label.pack()
        self.coords = None
        self.airfoil_name = None
        self.filename = None

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        self.root.destroy()
        import sys
        sys.exit(0)  # 显式告诉系统：我彻底走人了

    def parse_selig(self, file_path):
        """解析 Selig 格式的 .dat 文件"""
        coords = []
        name = "Unknown Airfoil"
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                if not lines:
                    return None, None
                
                # 第一行通常是名称
                name = lines[0].strip()
                
                # 正则匹配浮点数对
                coord_pattern = re.compile(r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$")
                
                for line in lines[1:]:
                    match = coord_pattern.match(line)
                    if match:
                        x = float(match.group(1))
                        y = float(match.group(2))
                        coords.append((x, y))
            
            if not coords:
                return None, name
                
            return coords, name
        except Exception as e:
            print(f"解析错误: {e}")
            return None, name

    def open_airfoil_file(self):
        """选择并加载一个 Selig 格式翼型文件。"""
        file_path = filedialog.askopenfilename(
            title="选择翼型文件",
            filetypes=[("Airfoil data", "*.dat"), ("All files", "*.*")],
        )
        if not file_path:
            return

        coords, name = self.parse_selig(file_path)
        filename = os.path.basename(file_path)
        if coords:
            self.draw_airfoil(coords, name, filename)
            return

        messagebox.showerror("格式错误", f"无法解析文件: {filename}\n请确保它是 Selig 格式的坐标文件。")

    def draw_airfoil(self, coords, name, filename):
        """保存翼型数据并绘制。"""
        self.coords = coords
        self.airfoil_name = name
        self.filename = filename
        self.render_airfoil()
        self.label.config(text=f"当前显示: {filename}")

    def on_canvas_resize(self, event):
        if self.coords:
            self.render_airfoil()

    def render_airfoil(self):
        """将当前翼型缩放到 Canvas 坐标系。"""
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        if width < 160 or height < 160:
            return

        margin_x = 70
        margin_y = 60
        x_min, x_max = -0.05, 1.05
        ys = [point[1] for point in self.coords]
        y_limit = max(abs(min(ys)), abs(max(ys))) * 1.5 + 0.1
        y_min, y_max = -y_limit, y_limit
        plot_width = width - 2 * margin_x
        plot_height = height - 2 * margin_y

        def to_canvas(x, y):
            canvas_x = margin_x + (x - x_min) / (x_max - x_min) * plot_width
            canvas_y = height - margin_y - (y - y_min) / (y_max - y_min) * plot_height
            return canvas_x, canvas_y

        self.canvas.delete("all")
        for index in range(11):
            x = x_min + (x_max - x_min) * index / 10
            canvas_x, _ = to_canvas(x, 0)
            self.canvas.create_line(canvas_x, margin_y, canvas_x, height - margin_y, fill="#d9d9d9", dash=(3, 3))
        for index in range(9):
            y = y_min + (y_max - y_min) * index / 8
            _, canvas_y = to_canvas(0, y)
            self.canvas.create_line(margin_x, canvas_y, width - margin_x, canvas_y, fill="#d9d9d9", dash=(3, 3))

        axis_x, _ = to_canvas(0, 0)
        _, axis_y = to_canvas(0, 0)
        self.canvas.create_line(axis_x, margin_y, axis_x, height - margin_y, fill="#666666")
        self.canvas.create_line(margin_x, axis_y, width - margin_x, axis_y, fill="#666666")

        points = [to_canvas(x, y) for x, y in self.coords]
        self.canvas.create_line(points, fill="#1565c0", width=2)
        for point_x, point_y in points:
            self.canvas.create_oval(point_x - 2, point_y - 2, point_x + 2, point_y + 2, fill="#d32f2f", outline="")

        self.canvas.create_text(width / 2, 24, text=f"Airfoil: {self.airfoil_name}", font=("Arial", 16))
        self.canvas.create_text(width / 2, height - 22, text="X/c", font=("Arial", 12))
        self.canvas.create_text(22, height / 2, text="Y/c", font=("Arial", 12), angle=90)

if __name__ == "__main__":
    root = tk.Tk()
    app = AirfoilPlotter(root)
    root.mainloop()
