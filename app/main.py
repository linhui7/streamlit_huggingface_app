import streamlit as st
from model import HuggingFaceModel

# Set page configuration
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)


def initialize_session_state():
    """Initialize session state variables if they don't exist"""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "model" not in st.session_state:
        st.session_state.model = HuggingFaceModel()

    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False


def load_model():
    """Load the model and update session state"""
    with st.spinner("Loading model... This might take a minute."):
        success = st.session_state.model.load_model()
        st.session_state.model_loaded = success
        if success:
            st.success("Model loaded successfully!")
        else:
            st.error("Failed to load model. Please check the logs.")


def main():
    initialize_session_state()

    # Sidebar with model options
    st.sidebar.title("Model Options")
    model_options = {
        "Tiny Random LLaMA (Fast)": "HuggingFaceM4/tiny-random-LlamaForCausalLM",
        "TinyLlama-1.1B": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    }

    selected_model = st.sidebar.selectbox(
        "Select a model:", list(model_options.keys()), key="model_selection"
    )

    # Update model if changed
    model_name = model_options[selected_model]
    if st.session_state.model.model_name != model_name:
        st.session_state.model = HuggingFaceModel(model_name=model_name)
        st.session_state.model_loaded = False

    # Load model button
    if not st.session_state.model_loaded:
        st.sidebar.button("Load Model", on_click=load_model)
    else:
        st.sidebar.success("Model loaded and ready!")

    # Max response length slider
    max_length = st.sidebar.slider(
        "Max response length:", min_value=20, max_value=200, value=100, step=10
    )

    # App header
    st.title("💬 AI Chatbot")
    st.caption("Chat with an open-source language model from Hugging Face")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask something..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Generate response
        with st.chat_message("assistant"):
            if not st.session_state.model_loaded:
                st.warning(
                    "Please load the model first using the button in the sidebar."
                )
                response = "Model not loaded. Please load the model first."
            else:
                with st.spinner("Thinking..."):
                    message_placeholder = st.empty()
                    response = st.session_state.model.generate_response(
                        prompt, max_length=max_length
                    )
                    message_placeholder.write(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
