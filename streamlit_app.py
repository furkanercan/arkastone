import streamlit as st
from src.interface.runner import run_polar_sim_with_len_k
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time

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
    list_size = None  # Default to None if not applicable
    scf_iters = None  # Default to None if not applicable
    if decoder_algorithm == "SCL":
        list_size = st.sidebar.number_input("List Size for SCL", min_value=1, max_value=64, value=8, step=1)
    elif decoder_algorithm == "SCFlip":
        scf_iters = st.sidebar.number_input("SCFlip Iterations", min_value=1, max_value=200, value=10, step=1)
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
            # Create placeholders for dynamic updates
            terminal_output_placeholder = st.empty()
            plot_placeholder = st.empty()
            table_placeholder = st.empty()

            # Initialize a variable to track the length of results
            previous_results_length = 0

            # Poll for results
            for results, terminal_log in run_polar_sim_with_len_k(
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
            ):
                # Check if there are updates (every 3 seconds or when results are updated)
                if time.time() % 3 < 0.5 or len(results) != previous_results_length:

                    # Update the previous results length
                    previous_results_length = len(results)

                    # Update terminal log
                    terminal_output_placeholder.expander("📟 Terminal Output", expanded=False).code(terminal_log, language="text")

                    # Extract results for plotting
                    snrs = [r["snr"] for r in results]
                    bers = [r["ber"] for r in results]
                    fers = [r["fer"] for r in results]

                    # Check if BER or FER is 0
                    if (bers[-1] == 0):
                        # Only update the results table
                        table_placeholder.table(results)
                        continue

                    # Create a new Plotly figure for each iteration
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
                        yaxis_type="log",
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

                    # Refresh the placeholder with the updated figure
                    with plot_placeholder.container():
                        unique_key = f"plot_{time.time()}"  # Use the current timestamp for uniqueness
                        st.plotly_chart(fig, use_container_width=True, key=unique_key)

                    # Update results table
                    table_placeholder.table(results)

            st.success("Simulation completed!")
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