import os

def read_data_from_directory(directory):
    texts, labels = [], []
    for label_dir in ["pos", "neg"]:
        label = 1 if label_dir == "pos" else 0
        folder = os.path.join(directory, label_dir)
        for filename in os.listdir(folder):
            with open(os.path.join(folder, filename), encoding="utf-8") as f:
                texts.append(f.read())
                labels.append(label)
    return texts, labels