import streamlit as st
from src.interface.runner import run_polar_sim_with_len_k
import matplotlib.pyplot as plt

st.title("Arkastone: Polar Simulation (Interactive)")

# Inputs
len_k = st.number_input("Set Polar Code len_k", min_value=8, max_value=2048, value=512, step=8)
save_output = st.checkbox("Save output to file", value=True)
seed = st.number_input("Random Seed", value=42)

# Run Button
if st.button("Run Simulation"):
    with st.spinner("Running simulation..."):
        results, terminal_log = run_polar_sim_with_len_k("configs/config_polar.json5", len_k, seed, save_output)

    # Show terminal log in expandable box
    with st.expander("📟 Terminal Output", expanded=False):
        st.code(terminal_log, language="text")

    # Plot results
    snrs = [r["snr"] for r in results]
    bers = [r["ber"] for r in results]
    fers = [r["fer"] for r in results]

    st.subheader("BER/FER vs SNR")
    fig, ax = plt.subplots()
    ax.plot(snrs, bers, label="BER", marker="o")
    ax.plot(snrs, fers, label="FER", marker="x")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Error Rate")
    ax.set_yscale("log")
    ax.legend()
    st.pyplot(fig)

    # Table
    st.subheader("Simulation Results")
    st.table(results)

    if save_output:
        # We assume the Simulation output path is accessible via the last result's metadata
        # If not, modify `run_polar_sim_with_len_k()` to return `output_dir` or `output_file_path` as well
        output_dir = results[0].get("output_dir", None)
        if output_dir:
            output_file_path = f"{output_dir}/results.txt"
            st.success(f"Output successfully saved to: `{output_file_path}`")

            # Read the file content for download
            with open(output_file_path, "r") as f:
                file_contents = f.read()

            st.download_button(
                label="Download Results File",
                data=file_contents,
                file_name="results.txt",
                mime="text/plain"
            )
        else:
            st.warning("Simulation finished, but output file path is not available.")

