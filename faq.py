import streamlit as st

st.set_page_config(page_title="Arkastone FAQ", layout="centered")
st.title("❓ Frequently Asked Questions")

faq_items = [
    ("Why do I need to download a client?",
     "Arkastone is designed to run simulations directly on your machine to save server costs and allow for high-performance computation without needing a cloud GPU. This keeps it free and scalable."),

    ("What does BYOB mean?",
     "It stands for 'Bring Your Own Brains' — meaning your computer does the thinking (simulation), while Arkastone handles the rest (UI, config, coordination)."),

    ("Is this secure?",
     "Yes, all communication is routed through HTTPS. However, since it's an early version, the backend doesn't yet require authentication. We're adding session validation and protections soon."),

    ("Is the client safe to run?",
     "Yes. It's a standalone Python-based executable. We will provide open-source code and optional build instructions so you can verify or build it yourself."),

    ("Why isn't everything in the browser?",
     "Browser simulations would be slower, more limited, and expensive to host. Arkastone lets you run full-scale simulations without server load."),

    ("How is my session identified?",
     "Each time you use the simulator, a unique session ID is generated in your browser. This ID links the UI to your local client securely."),

    ("Can I run multiple simulations?",
     "Yes, but only one per session. You can refresh the page or open a new window to start another session."),

    ("Can two users collide?",
     "No — each session is isolated using its own session ID. Results and progress are separated for every user."),

    ("Do I have to enter the session ID manually?",
     "For now, yes. We plan to automate this by embedding it into the client during download or launching it with one click."),

    ("Can I see my simulation history?",
     "Not yet — but future versions will include a per-user history, especially when email login is enabled."),

    ("What platforms are supported?",
     "Currently Linux (.out) and Windows (.exe). macOS support is planned."),

    ("How often does the backend update?",
     "The backend updates in real-time during your simulation as the client sends progress."),

    ("Is this open source?",
     "Yes, the code will probably be made open-source for transparency and collaboration."),

    ("Will this always be free?",
     "Yes for the basic use. Advanced features or cloud-hosted modes may be added later with optional pricing. It really depends on where this is going. But the core functionalities will always be free."),

    ("How can I trust the executable?",
     "You’ll be able to verify it against the GitHub source code, or build your own from the same source."),

    ("What if my simulation gets stuck?",
     "You can stop the client and re-run it with the same session ID, or refresh and start over."),

    ("How does the backend work?",
     "It acts as a simple job relay: your config is sent from the UI, the client picks it up, runs it, and sends back results."),

    ("Will it work without internet?",
     "You need internet to load the UI and for the client to communicate, but all computation happens on your machine."),

    ("Can I run this on multiple cores?",
     "Yes, the client and simulation code can be expanded to utilize parallel computation depending on your machine."),

    ("What’s next for Arkastone?",
     "Session persistence, user authentication, cloud-based simulation mode, history tracking, and better UI polish. Honestly, there are a number of long to-do lists all over the place, waiting to be tackled."),
]

for question, answer in faq_items:
    with st.expander(question):
        st.write(answer)

st.markdown("---")
st.markdown("Built with ❤️ by the Arkastone project.")
