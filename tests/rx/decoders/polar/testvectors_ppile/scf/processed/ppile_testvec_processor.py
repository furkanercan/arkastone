import os

RAW_DIR = "tests/rx/decoders/polar/testvectors_ppile/scf/raw"
PROCESSED_DIR = "tests/rx/decoders/polar/testvectors_ppile/scf/processed"
# RAW_DIR = "raw"
# PROCESSED_DIR = "."

os.makedirs(PROCESSED_DIR, exist_ok=True)

def process_file(file_path, filename):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    lines = lines[1:]  # skip header

    if len(lines) % 8 != 0:
        raise ValueError(f"{filename}: Malformed file, total lines not multiple of 4")

    llr_lines = []
    decoded_lines = []

    for i in range(0, len(lines), 8):
        llr = lines[i+3].strip()
        decoded = lines[i+5].strip()
        llr_lines.append(llr)
        decoded_lines.append(decoded)

    base_name = os.path.splitext(filename)[0]
    llr_out_path = os.path.join(PROCESSED_DIR, f"{base_name}.in")
    decoded_out_path = os.path.join(PROCESSED_DIR, f"{base_name}.out")

    with open(llr_out_path, 'w') as f:
        f.write("\n".join(llr_lines))
    with open(decoded_out_path, 'w') as f:
        f.write("\n".join(decoded_lines))

    print(f"Processed: {filename} → {base_name}.in/.out")



def main():
    for filename in os.listdir(RAW_DIR):
        if filename.endswith(".txt"):
            full_path = os.path.join(RAW_DIR, filename)
            process_file(full_path, filename)

if __name__ == "__main__":
    main()
