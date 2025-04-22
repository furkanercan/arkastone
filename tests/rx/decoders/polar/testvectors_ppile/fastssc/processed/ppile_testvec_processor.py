import os

RAW_DIR = "tests/rx/decoders/polar/testvectors_ppile/fastssc/raw"
PROCESSED_DIR = "tests/rx/decoders/polar/testvectors_ppile/fastssc/processed"
# RAW_DIR = "raw"
# PROCESSED_DIR = "."

os.makedirs(PROCESSED_DIR, exist_ok=True)

def process_file(file_path, filename):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    lines = lines[1:]  # skip header

    if len(lines) % 4 != 0:
        raise ValueError(f"{filename}: Malformed file, total lines not multiple of 4")

    llr_lines = []
    decoded_lines = []

    for i in range(0, len(lines), 4):
        llr = lines[i+2].strip()
        decoded = lines[i+3].strip()
        llr_lines.append(llr)
        decoded_lines.append(decoded)

    base_name = os.path.splitext(filename)[0]
    llr_out_path = os.path.join(PROCESSED_DIR, f"{base_name}.in")
    decoded_out_path = os.path.join(PROCESSED_DIR, f"{base_name}.out")

    with open(llr_out_path, 'w') as f:
        f.write("\n".join(llr_lines))
    with open(decoded_out_path, 'w') as f:
        f.write("\n".join(decoded_lines))

    # 💡 Instruction parsing if matching _instr file exists
    instr_base = base_name.replace("Q0", "instr").replace("Q1", "instr")
    instr_path = os.path.join(RAW_DIR, f"{instr_base}.txt")

    if os.path.exists(instr_path):
        with open(instr_path, "r") as f:
            instr_lines = f.readlines()

        vec_dec_sch = []
        vec_dec_sch_size = []
        vec_dec_sch_dir = []

        for line in instr_lines:
            parts = line.strip().split()
            if len(parts) != 3:
                continue  # skip malformed lines
            instr, size, direction = parts
            vec_dec_sch.append(instr)
            vec_dec_sch_size.append(size)
            vec_dec_sch_dir.append(direction)

        # Save them
        with open(os.path.join(PROCESSED_DIR, f"{base_name}.sch"), 'w') as f:
            f.write(" ".join(vec_dec_sch))

        with open(os.path.join(PROCESSED_DIR, f"{base_name}.size"), 'w') as f:
            f.write(" ".join(vec_dec_sch_size))

        with open(os.path.join(PROCESSED_DIR, f"{base_name}.dir"), 'w') as f:
            f.write(" ".join(vec_dec_sch_dir))

        print(f"Processed instructions: {instr_base}.txt → {base_name}.sch/.size/.dir")

    print(f"Processed: {filename} → {base_name}.in/.out")



def main():
    for filename in os.listdir(RAW_DIR):
        if filename.endswith(".txt"):
            full_path = os.path.join(RAW_DIR, filename)
            process_file(full_path, filename)

if __name__ == "__main__":
    main()
