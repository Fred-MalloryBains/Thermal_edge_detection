all_files = []

for lwir_dir in thermal_paths:
    if lwir_dir.is_dir():
        files = list(lwir_dir.iterdir())
        all_files.extend(files)


img = cv2.imread(str(all_files[0]), cv2.IMREAD_GRAYSCALE)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
edges = clahe.apply(img)
edges = cv2.Canny(edges, 70, 130)
Image.fromarray(edges).save("../../outputs/edges.png")