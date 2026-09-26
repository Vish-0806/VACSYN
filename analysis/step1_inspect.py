import os
import json

base_dir = r"c:\VACSYN"
student_dir = os.path.join(base_dir, "student_resource")
dataset_dir = os.path.join(student_dir, "dataset")

files_to_inspect = [
    os.path.join(dataset_dir, "train", "train_source1.tsv"),
    os.path.join(dataset_dir, "train", "train_source2.tsv"),
    os.path.join(dataset_dir, "train", "train_source3.tsv"),
    os.path.join(dataset_dir, "train", "train_ground_truth.tsv"),
    os.path.join(dataset_dir, "test", "test_source1.tsv"),
    os.path.join(dataset_dir, "test", "test_source2.tsv"),
    os.path.join(dataset_dir, "test", "test_source3.tsv"),
]

results = {}

for fpath in files_to_inspect:
    rel_path = os.path.relpath(fpath, base_dir)
    file_size_bytes = os.path.getsize(fpath)
    file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
    
    total_lines = 0
    header_line = None
    delimiter = "\t"
    has_tab = False
    has_comma = False
    
    # Stream line by line to prevent high memory usage
    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
        for idx, line in enumerate(f):
            total_lines += 1
            if idx == 0:
                header_line = line.rstrip("\r\n")
                has_tab = "\t" in header_line
                has_comma = "," in header_line

    cols = header_line.split("\t") if header_line and "\t" in header_line else [header_line]
    num_cols = len(cols)
    row_count = max(0, total_lines - 1)  # excluding header
    
    results[rel_path] = {
        "filename": os.path.basename(fpath),
        "relative_path": rel_path.replace("\\", "/"),
        "file_size_bytes": file_size_bytes,
        "file_size_mb": file_size_mb,
        "has_header": True,
        "delimiter": "\\t" if has_tab else ("," if has_comma else "unknown"),
        "column_names": cols,
        "number_of_columns": num_cols,
        "total_lines": total_lines,
        "data_row_count": row_count,
        "encoding": "utf-8"
    }

print(json.dumps(results, indent=2))
with open(r"c:\VACSYN\analysis\file_inspection.json", "w", encoding="utf-8") as out:
    json.dump(results, out, indent=2)
