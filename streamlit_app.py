import streamlit as st
from src.interface.runner import run_polar_sim_with_len_k
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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

# Initial choice dropdown with no default selection
choice = st.selectbox(
    "I want to ...",
    options=["", "simulate a 5G polar code", "simulate a 5G PUCCH PHY sequence"],
    format_func=lambda x: "Select an option" if x == "" else x
)

if choice == "simulate a 5G polar code":
    # Inputs for Polar Code Simulation
    st.subheader("Polar Code Simulation Configuration")

    # Code Configuration
    st.sidebar.header("Code Configuration")
    len_N = st.sidebar.number_input("Set 5G Polar Code Length", min_value=16, max_value=4096, value=1024, step=16)
    len_k = st.sidebar.number_input("Set Polar Code len_k", min_value=8, max_value=2048, value=512, step=8)
    decoder_algorithm = st.sidebar.selectbox("Decoder Algorithm", options=["SC", "SCL", "SCFlip"], index=0)
    crc_enable = st.sidebar.checkbox("Enable CRC", value=False)
    crc_length = st.sidebar.number_input("CRC Length", min_value=0, max_value=32, value=8, step=1, disabled=not crc_enable)

    # Modulation Configuration
    st.sidebar.header("Modulation Configuration")
    modulation_type = st.sidebar.selectbox("Modulation Type", options=["BPSK", "QPSK", "16QAM"], index=1)
    demod_type = st.sidebar.selectbox("Demodulation Type", options=["soft", "hard"], index=0)

    # OFDM Configuration
    st.sidebar.header("OFDM Configuration")
    num_subcarriers = st.sidebar.number_input("Number of Subcarriers", min_value=1, max_value=200, value=16, step=1)
    cyclic_prefix_length = st.sidebar.number_input("Cyclic Prefix Length", min_value=0, max_value=32, value=4, step=1)

    # Channel Configuration
    st.sidebar.header("Channel Configuration")
    channel_type = st.sidebar.selectbox("Channel Type", options=["SNR", "AWGN"], index=0)
    snr_start = st.sidebar.number_input("SNR Start (dB)", min_value=-20.0, max_value=50.0, value=1.0, step=0.1)
    snr_end = st.sidebar.number_input("SNR End (dB)", min_value=-20.0, max_value=50.0, value=2.0, step=0.1)
    snr_step = st.sidebar.number_input("SNR Step (dB)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)

    # Simulation Loop Configuration
    st.sidebar.header("Simulation Loop Configuration")
    num_frames = st.sidebar.number_input("Number of Frames", min_value=1, max_value=1000000, value=1000, step=1)
    num_errors = st.sidebar.number_input("Number of Errors", min_value=0, max_value=1000000, value=0, step=1)
    max_frames = st.sidebar.number_input("Maximum Frames", min_value=1, max_value=1000000, value=10000, step=1)

    # Simulation Configuration
    st.sidebar.header("Simulation Configuration")
    save_output = st.sidebar.checkbox("Save output to file", value=True)
    seed = st.sidebar.number_input("Random Seed", value=42)

    # Quantization Configuration
    st.sidebar.header("Quantization Configuration")
    quantize_enable = st.sidebar.checkbox("Enable Quantization", value=False)
    bits_chnl = st.sidebar.number_input(
        "Bits for Channel Quantization",
        min_value=1,
        max_value=16,
        value=5,
        step=1,
        disabled=not quantize_enable
    )
    bits_intl = st.sidebar.number_input(
        "Bits for Internal Quantization",
        min_value=1,
        max_value=16,
        value=6,
        step=1,
        disabled=not quantize_enable
    )
    bits_frac = st.sidebar.number_input(
        "Bits for Fractional Quantization",
        min_value=0,
        max_value=16,
        value=1,
        step=1,
        disabled=not quantize_enable
    )

    # Run Button
    if st.button("Run Configuration"):
        with st.spinner("Running simulation..."):
            # Pass the advanced configuration parameters to the simulation function
            print("Running simulation with the following parameters:")
            print(f"len_N: {len_N}, len_k: {len_k}, seed: {seed}")
            print(f"save_output: {save_output}, decoder_algorithm: {decoder_algorithm}")
            print(f"crc_enable: {crc_enable}, crc_length: {crc_length}")
            print(f"quantize_enable: {quantize_enable}, bits_chnl: {bits_chnl}")
            print(f"bits_intl: {bits_intl}, bits_frac: {bits_frac}")
            print(f"modulation_type: {modulation_type}, demod_type: {demod_type}")
            print(f"num_subcarriers: {num_subcarriers}, cyclic_prefix_length: {cyclic_prefix_length}")
            print(f"channel_type: {channel_type}, snr_start: {snr_start}")
            print(f"snr_end: {snr_end}, snr_step: {snr_step}")
            print(f"num_frames: {num_frames}, num_errors: {num_errors}, max_frames: {max_frames}")
            
            results, terminal_log = run_polar_sim_with_len_k(
                "configs/config_polar.json5",
                len_N,
                len_k,
                seed,
                save_output,
                decoder_algorithm,
                crc_enable,
                crc_length,
                quantize_enable,
                bits_chnl,
                bits_intl,
                bits_frac,
                modulation_type,
                demod_type,
                num_subcarriers,
                cyclic_prefix_length,
                channel_type,
                snr_start,
                snr_end,
                snr_step,
                num_frames,
                num_errors,
                max_frames
            )

        # Show terminal log in expandable box
        with st.expander("📟 Terminal Output", expanded=False):
            st.code(terminal_log, language="text")

        # Extract results for plotting
        snrs = [r["snr"] for r in results]
        bers = [r["ber"] for r in results]
        fers = [r["fer"] for r in results]

        # Plot results using Plotly
        st.subheader("BER/FER vs SNR")

        # Create a Plotly figure
        fig = go.Figure()

        # Create a Plotly figure
        fig = go.Figure()

        # Add BER trace
        fig.add_trace(go.Scatter(
            x=snrs,
            y=bers,
            mode='lines+markers',
            name='BER',
            marker=dict(symbol='circle', size=8),
            line=dict(width=2)
        ))

        # Add FER trace
        fig.add_trace(go.Scatter(
            x=snrs,
            y=fers,
            mode='lines+markers',
            name='FER',
            marker=dict(symbol='x', size=8),
            line=dict(width=2)
        ))

        # Update layout
        fig.update_layout(
            title="BER/FER vs SNR",
            xaxis_title="SNR (dB)",
            yaxis_title="Error Rate",
            yaxis_type="log",  # Set y-axis to logarithmic scale
            template="plotly_white",
            legend=dict(
                title="Legend",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Render the Plotly chart in Streamlit
        st.plotly_chart(fig, use_container_width=True)

        # Table
        st.subheader("Simulation Results")
        st.table(results)

        if save_output:
            output_dir = results[0].get("output_dir", None)
            if output_dir:
                output_file_path = f"{output_dir}/results.txt"
                st.success(f"Output successfully saved to: `{output_file_path}`")
                with open(output_file_path, "r") as f:
                    file_contents = f.read()
                st.download_button(label="Download Results File", data=file_contents, file_name="results.txt", mime="text/plain")
            else:
                st.warning("Simulation finished, but output file path is not available.")
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