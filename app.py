import streamlit as st
from query import generate_answer

st.set_page_config(page_title="Home Inspection Chat Assistance", page_icon=":house:", layout="centered")

st.title("Home Inspection Report — Q&A")
st.write("Use this chatbot to ask a question about the inspection report and get a concise answer.")

with st.form(key="qa_form"):
    query = st.text_area("Your question", height=120, placeholder="e.g. What issues were found in the kitchen?")
    submit = st.form_submit_button("Ask")

if submit:
    q = (query or "").strip()
    if not q:
        st.warning("Please type a question you want to ask about the inspection report.")
    else:
        try:
            result = generate_answer(q)  
        except Exception as e:
            st.error(f"An error occurred while answering: {e}")
        else:
            answer = result.get("answer") 
            if not answer:
                st.info("There was not enough information to answer the query.")
            else:
                st.subheader("Answer")
                st.markdown(answer)
