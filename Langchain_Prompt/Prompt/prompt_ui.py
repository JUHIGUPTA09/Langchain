from langchain_community.chat_models import ChatOllama
import  streamlit as st

model=ChatOllama(model="llama3")
st.header("Molu Bot")
user_input=st.text_input("Enter the text")
if st.button("Summarize"):
    result=model.invoke(user_input)
    st.text(result.content)


