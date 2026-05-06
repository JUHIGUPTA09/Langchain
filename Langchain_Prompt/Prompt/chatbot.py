from langchain_community.chat_models import ChatOllama
import streamlit as st

# Initialize the model
model = ChatOllama(model="llama3")

st.header("Chatbot at your service")

# 1. Create a chat input area at the bottom of the screen
query = st.chat_input("Let me know your queries...")

if query:
    # 2. Display the User message
        with st.chat_message("You"):
            st.markdown(query)

    # 3. Get the AI response
    # We use a spinner so the user knows the AI is "thinking"
        with st.spinner("Thinking..."):
         response = model.invoke(query)
         result = response.content

    # 4. Display the AI response
         with st.chat_message("Agent"):
          st.markdown(f"**AI:** {result}")