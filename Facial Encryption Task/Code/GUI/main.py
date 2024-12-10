import tkinter as tk
from tkinter import simpledialog, filedialog, messagebox
from PIL import Image, ImageTk

# 全局变量
password = ""  # 存储设置的密码
input_image_path = ""  # 输入图片路径（输入1）
output_image_setup_path = ""  # 设置时的输出图片路径（输出1）
output_image_correct_path = ""  # 验证正确的输出图片路径（输出2）
output_image_wrong_path = ""  # 验证错误的输出图片路径（输出3）

# 设置密码和图片
def setup():
    global password, input_image_path, output_image_setup_path, output_image_correct_path, output_image_wrong_path
    input_image_path = filedialog.askopenfilename(title="选择输入图片")
    output_image_setup_path = filedialog.askopenfilename(title="选择设置时的输出图片（输出1）")
    output_image_correct_path = filedialog.askopenfilename(title="选择密码正确时的输出图片（输出2）")
    output_image_wrong_path = filedialog.askopenfilename(title="选择密码错误时的输出图片（输出3）")
    password = simpledialog.askstring("设置", "设置密码")
    if input_image_path and output_image_setup_path and password:
        show_image(input_image_path, 'left')
        show_image(output_image_setup_path, 'right')
        messagebox.showinfo("设置完成", "账户设置成功！")
    else:
        messagebox.showerror("设置失败", "请提供密码和所有必要的图片。")

# 验证密码并显示相应的图片
def verify():
    input_password = simpledialog.askstring("验证", "请输入密码")
    show_image(input_image_path, 'left')  # 再次展示输入图片（输入1）
    if input_password == password:
        show_image(output_image_correct_path, 'right')
        messagebox.showinfo("验证", "密码正确！")
    else:
        show_image(output_image_wrong_path, 'right')
        messagebox.showerror("验证", "密码错误！")

# 显示图片
def show_image(path, side):
    img = Image.open(path)
    img = img.resize((500, 500), Image.Resampling.LANCZOS)  # 更新图片大小调整方法
    img = ImageTk.PhotoImage(img)
    if side == 'left':
        panel_left.config(image=img)
        panel_left.image = img
    else:
        panel_right.config(image=img)
        panel_right.image = img

# 创建GUI
root = tk.Tk()
root.title("密码设置和验证")
root.geometry("1200x600")  # 设置窗口初始大小

# 按钮区域
frame_buttons = tk.Frame(root)
frame_buttons.pack(side="top", fill="x")

# 图片显示区域
frame_images = tk.Frame(root)
frame_images.pack(expand=True, fill="both")

# 左右图片区域
frame_left = tk.Frame(frame_images, width=600, height=600)  # 指定左侧框架大小
frame_left.pack(side="left", expand=True, fill="both")

frame_right = tk.Frame(frame_images, width=600, height=600)  # 指定右侧框架大小
frame_right.pack(side="right", expand=True, fill="both")

# 设置和验证按钮
setup_btn = tk.Button(frame_buttons, text="设置", command=setup)
setup_btn.pack(side="left", padx=10, pady=10)

verify_btn = tk.Button(frame_buttons, text="验证", command=verify)
verify_btn.pack(side="left", padx=10, pady=10)

# 图片面板
panel_left = tk.Label(frame_left)
panel_left.pack()

panel_right = tk.Label(frame_right)
panel_right.pack()

root.mainloop()
