import h5py
from collections import Counter

def extract_datasets(h5_path):
    items = []  # (name, shape)
    with h5py.File(h5_path, "r") as f:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                items.append((name, obj.shape))
        f.visititems(visitor)
    return items

def summarize(items):
    shapes = Counter([s for _, s in items])
    return {
        "num_datasets": len(items),
        "unique_shapes": len(shapes),
        "top_shapes": shapes.most_common(10),
    }

def main(ok_path, bad_path):
    ok = extract_datasets(ok_path)
    bad = extract_datasets(bad_path)

    ok_set = set(ok)
    bad_set = set(bad)

    print("OK:", ok_path)
    print(" ", summarize(ok))
    print("BAD:", bad_path)
    print(" ", summarize(bad))

    print("\n--- In OK but not in BAD (show 30) ---")
    for x in list(ok_set - bad_set)[:30]:
        print(" ", x)

    print("\n--- In BAD but not in OK (show 30) ---")
    for x in list(bad_set - ok_set)[:30]:
        print(" ", x)

if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2])