from PIL import Image

func_name = ["abc", "cuckoo_search", "de", "firefly", "pso", "ga", "tlbo"]
for i in range(len(func_name)):
    gif = Image.open(f"3D_Gif/{func_name[i]}.gif")

    # Lấy frame đầu tiên
    gif.seek(0)
    gif.save(f"3D_IMG/{func_name[i]}_first.png")

    # Lấy frame cuối cùng
    gif.seek(gif.n_frames - 1)
    gif.save(f"3D_IMG/{func_name[i]}_last.png")
    print(f"Saved first and last images for {func_name[i]}")
