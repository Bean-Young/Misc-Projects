import os
from PIL import Image, ImageDraw

# 原始图像所在的根目录
input_root = "/Users/youngbean/Downloads/Data/Result"
# 裁剪后的图像保存目录
output_root = "/Users/youngbean/Downloads/Data/Cropped"
# 原图带框的对比图保存目录
marked_root = "/Users/youngbean/Downloads/Data/Marked"
os.makedirs(output_root, exist_ok=True)
os.makedirs(marked_root, exist_ok=True)

# 设置两个要裁剪的区域坐标：(left, upper, right, lower)
crop_region_1 = (1600, 1600, 1800, 1800)
crop_region_2 = (1000, 300, 1200, 500)

# 支持的图像扩展名
image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

# 遍历文件夹
for root, dirs, files in os.walk(input_root):
    for file in files:
        ext = os.path.splitext(file)[1].lower()
        if ext in image_exts:
            image_path = os.path.join(root, file)
            rel_path = os.path.relpath(root, input_root)

            save_crop_dir = os.path.join(output_root, rel_path)
            save_marked_dir = os.path.join(marked_root, rel_path)
            os.makedirs(save_crop_dir, exist_ok=True)
            os.makedirs(save_marked_dir, exist_ok=True)

            try:
                with Image.open(image_path) as img:
                    # 裁剪区域1
                    cropped1 = img.crop(crop_region_1)
                    name1 = os.path.splitext(file)[0] + "_crop1" + ext
                    cropped1.save(os.path.join(save_crop_dir, name1))

                    # 裁剪区域2
                    cropped2 = img.crop(crop_region_2)
                    name2 = os.path.splitext(file)[0] + "_crop2" + ext
                    cropped2.save(os.path.join(save_crop_dir, name2))

                    # 创建带框图像
                    marked_img = img.copy()
                    draw = ImageDraw.Draw(marked_img)
                    draw.rectangle(crop_region_1, outline="red", width=5)
                    draw.rectangle(crop_region_2, outline="green", width=5)

                    marked_name = os.path.splitext(file)[0] + "_marked" + ext
                    marked_img.save(os.path.join(save_marked_dir, marked_name))

                    print(f"Processed and marked: {image_path}")
            except Exception as e:
                print(f"Failed to process {image_path}: {e}")