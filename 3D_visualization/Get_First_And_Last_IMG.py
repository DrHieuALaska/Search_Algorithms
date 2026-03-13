from PIL import Image

func_name = ["ABC", "Cuckoo", "DE", "Firefly", "PSO", "GA", "TLBO"]
for i in range(len(func_name)):
    gif = Image.open(f"3D_Gif/{func_name[i]}.gif")

    # Lấy frame đầu tiên
    gif.seek(0)
    gif.save(f"{func_name[i]}_first.png")

    # Lấy frame cuối cùng
    gif.seek(gif.n_frames - 1)
    gif.save(f"{func_name[i]}_last.png")
