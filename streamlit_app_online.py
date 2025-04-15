import streamlit as st
from src.utils.validation.config_loader import ConfigLoader
from src.utils.validation.validation_manager import validate_config

import plotly.graph_objects as go
import time
import requests

# Add a logo with reduced size
st.image("assets/arkastone_logo_transparent.png", width=200)

# Add a welcome sentence
st.title("the arkastone simulator")
st.markdown(
    """
    Welcome to the Arkastone Communication System Simulator! 
    This tool allows you to simulate and analyze 5G communication components interactively.
    """
)

choice = st.selectbox(
    "I want to ...",
    options=["", "simulate a 5G polar code", "simulate a 5G PUCCH PHY sequence"],
    format_func=lambda x: "Select an option" if x == "" else x
)

polar_file_map = {
    1024: "src/lib/ecc/polar/3gpp/n1024_3gpp.pc",
    512: "src/lib/ecc/polar/3gpp/n512_3gpp.pc",
    256: "src/lib/ecc/polar/3gpp/n256_3gpp.pc",
    128: "src/lib/ecc/polar/3gpp/n128_3gpp.pc",
    64: "src/lib/ecc/polar/3gpp/n64_3gpp.pc",
    32: "src/lib/ecc/polar/3gpp/n32_3gpp.pc",
}

if choice == "simulate a 5G polar code":
    default_config_path = "configs/config_polar.json5"
    config = ConfigLoader(default_config_path).get()

    st.subheader("5G Polar Code Simulation Configuration")

    st.sidebar.header("Code Configuration")
    len_N = st.sidebar.number_input("Set 5G Polar Code Length", min_value=16, max_value=1024, value=1024, step=16)
    len_k = st.sidebar.number_input("Set Polar Code len_k", min_value=8, max_value=2048, value=512, step=8)
    decoder_algorithm = st.sidebar.selectbox("Decoder Algorithm", options=["SC", "SC-List", "SC-Flip"], index=0)
    crc_enable = st.sidebar.checkbox("Enable CRC", value=False)
    crc_length = st.sidebar.number_input("CRC Length", min_value=0, max_value=32, value=8, step=1, disabled=not crc_enable)

    st.sidebar.header("Modulation Configuration")
    modulation_type = st.sidebar.selectbox("Modulation Type", options=["BPSK", "QPSK", "16QAM"], index=1)
    demod_type = st.sidebar.selectbox("Demodulation Type", options=["soft", "hard"], index=0)

    st.sidebar.header("OFDM Configuration")
    num_subcarriers = st.sidebar.number_input("Number of Subcarriers", min_value=1, max_value=200, value=16, step=1)
    cyclic_prefix_length = st.sidebar.number_input("Cyclic Prefix Length", min_value=0, max_value=32, value=4, step=1)

    st.sidebar.header("Simulation Configuration")
    sim_type = st.sidebar.selectbox("Sweep Type", options=["SNR", "EbN0"], index=0)
    snr_start = st.sidebar.number_input("SNR Start (dB)", min_value=-20.0, max_value=50.0, value=1.0, step=0.1)
    snr_end = st.sidebar.number_input("SNR End (dB)", min_value=-20.0, max_value=50.0, value=2.0, step=0.1)
    snr_step = st.sidebar.number_input("SNR Step (dB)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)

    st.sidebar.header("Simulation Loop Configuration")
    num_frames = st.sidebar.number_input("Number of Frames", min_value=1, max_value=1000000, value=1000, step=1)
    num_errors = st.sidebar.number_input("Number of Errors", min_value=0, max_value=1000000, value=0, step=1)
    max_frames = st.sidebar.number_input("Maximum Frames", min_value=1, max_value=1000000, value=10000, step=1)

    st.sidebar.header("Simulation Configuration")
    save_output = st.sidebar.checkbox("Save output to file", value=True)
    seed = st.sidebar.number_input("Random Seed", value=42)

    st.sidebar.header("Quantization Configuration")
    quantize_enable = st.sidebar.checkbox("Enable Quantization", value=False)
    bits_chnl = st.sidebar.number_input("Bits for Channel Quantization", 1, 16, 5, 1, disabled=not quantize_enable)
    bits_intl = st.sidebar.number_input("Bits for Internal Quantization", 1, 16, 6, 1, disabled=not quantize_enable)
    bits_frac = st.sidebar.number_input("Bits for Fractional Quantization", 0, 16, 1, 1, disabled=not quantize_enable)

    if st.button("Run Configuration"):
        with st.spinner("Submitting configuration to backend..."):
            config["code"]["polar"]["polar_file"] = polar_file_map[len_N]
            config["code"]["len_k"] = len_k
            config["code"]["polar"]["decoder"]["algorithm"] = decoder_algorithm
            config["code"]["polar"]["crc"]["enable"] = crc_enable
            config["code"]["polar"]["crc"]["length"] = crc_length
            config["code"]["polar"]["quantize"]["enable"] = quantize_enable
            config["code"]["polar"]["quantize"]["bits_chnl"] = bits_chnl
            config["code"]["polar"]["quantize"]["bits_intl"] = bits_intl
            config["code"]["polar"]["quantize"]["bits_frac"] = bits_frac
            config["mod"]["type"] = modulation_type
            config["mod"]["demod_type"] = demod_type
            config["ofdm"]["num_subcarriers"] = num_subcarriers
            config["ofdm"]["cyclic_prefix_length"] = cyclic_prefix_length
            config["sim"]["sweep_type"] = sim_type
            config["sim"]["sweep_vals"]["start"] = snr_start
            config["sim"]["sweep_vals"]["end"] = snr_end
            config["sim"]["sweep_vals"]["step"] = snr_step
            config["sim"]["loop"]["num_frames"] = num_frames
            config["sim"]["loop"]["num_errors"] = num_errors
            config["sim"]["loop"]["max_frames"] = max_frames
            config = validate_config(config)

            try:
                import numpy as np
                def make_json_serializable(obj):
                    if isinstance(obj, dict):
                        return {make_json_serializable(k): make_json_serializable(v) for k, v in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return [make_json_serializable(v) for v in obj]
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, (np.generic, np.integer, np.floating)):
                        return obj.item()
                    else:
                        return obj
                config = make_json_serializable(config)
                res = requests.post("http://localhost:8001/run_config", json=config)
                if res.status_code == 200:
                    st.success("Configuration submitted. Waiting for results...")
                else:
                    st.error("Failed to submit config to backend.")
                    st.stop()
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        terminal_output_placeholder = st.empty()
        plot_placeholder = st.empty()
        table_placeholder = st.empty()

        results_so_far = []
        previous_results_length = 0

        while True:
            try:
                r = requests.get("http://localhost:8001/get_progress")
                progress_data = r.json()
                # st.write("Progress data received:", progress_data)
                if progress_data and len(progress_data) > previous_results_length:
                    previous_results_length = len(progress_data)
                    results_so_far = progress_data

                    # Filter only permanent entries
                    # Keep only latest result per snr_point, prioritizing perm > temp
                    latest_by_snr = {}

                    for entry in results_so_far:
                        snr = entry.get("snr_point")
                        if snr is None:
                            continue

                        current = latest_by_snr.get(snr)

                        if current is None:
                            latest_by_snr[snr] = entry
                        elif current.get("type") == "temp" and entry.get("type") == "perm":
                            latest_by_snr[snr] = entry
                        elif entry.get("type") == "temp":
                            latest_by_snr[snr] = entry  # Replace temp with fresher temp

                    final_display = sorted(latest_by_snr.values(), key=lambda x: x["snr_point"])

                    clean_display = [{k: v for k, v in row.items() if k != "type"} for row in final_display]
                    table_placeholder.table(clean_display)

                    snrs = [r["snr_point"] for r in final_display]
                    bers = [r["ber"] for r in final_display]
                    fers = [r["bler"] for r in final_display]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=snrs, y=bers, mode='lines+markers', name='BER'))
                    fig.add_trace(go.Scatter(x=snrs, y=fers, mode='lines+markers', name='FER'))
                    fig.update_layout(
                        title="BER/FER vs SNR",
                        xaxis_title="SNR (dB)",
                        yaxis_title="Error Rate",
                        yaxis_type="log",
                        template="plotly_white"
                    )
                    plot_placeholder.plotly_chart(fig, use_container_width=True, key=f"plot_{time.time()}")


            except Exception as e:
                st.error(f"Error fetching progress: {e}")
                break

            try:
                fr = requests.get("http://localhost:8001/get_final_result")
                if fr.status_code == 200 and fr.json().get("status") == "done":
                    st.success("Simulation completed!")
                    break
            except Exception:
                pass

            time.sleep(3)

elif choice == "simulate a 5G PUCCH PHY sequence":
    st.subheader("PUCCH PHY Sequence Simulation")
    st.markdown("This feature is under development. Stay tuned!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: lightgray; font-size: small;">
       2025 © Furkan Ercan. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)