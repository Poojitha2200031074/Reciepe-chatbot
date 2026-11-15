import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI Recipe Chatbot", page_icon="🍳", layout="centered")

st.title("🍳 Local AI Recipe Assistant")
st.caption("Enter the ingredients you have on hand, and your locally fine-tuned LLM will suggest a recipe.")

# Sidebar for server status and generation controls
with st.sidebar:
    st.header("Server & Model Settings")
    try:
        health = requests.get(f"{API_URL}/health", timeout=2).json()
        st.success(f"API Connected ({health['device'].upper()})")
        st.info(f"**Active Model:**\n{health['model']}")
    except Exception:
        st.error("FastAPI Server Offline. Start `uvicorn api:app` first.")

    temperature = st.slider("Creativity (Temperature)", 0.1, 1.0, 0.4, 0.1)
    max_tokens = st.slider("Max Response Length", 100, 600, 300, 50)

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! Tell me what ingredients you have (for example: **Egg, Onions**), and I'll whip up a recipe for you."
        }
    ]

# Display existing chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle user input
if user_input := st.chat_input("Enter ingredients (e.g., Egg, Onions)..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Cooking up a recipe..."):
            try:
                response = requests.post(
                    f"{API_URL}/generate-recipe",
                    json={
                        "ingredients": user_input,
                        "temperature": temperature,
                        "max_new_tokens": max_tokens
                    },
                    timeout=120
                )
                if response.status_code == 200:
                    data = response.json()
                    reply = data["recipe"]
                else:
                    reply = f"⚠️ API Error ({response.status_code}): {response.text}"
            except requests.exceptions.ConnectionError:
                reply = "⚠️ Could not connect to the FastAPI backend. Make sure `uvicorn api:app --reload` is running."
            except Exception as e:
                reply = f"⚠️ Unexpected error: {str(e)}"

            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})