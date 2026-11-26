
import streamlit as st
import re
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

st.set_page_config(
    page_title="Open Book Semantic LLM",
    page_icon="📚"
)

st.title("📚 Open Book Semantic LLM")
st.write("Upload your study material and ask questions!")

@st.cache_resource
def load_models():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    qa = pipeline(
        "text2text-generation",
        model="google/flan-t5-small"
    )
    return model, qa

with st.spinner("Loading AI models..."):
    model, qa = load_models()

uploaded_file = st.file_uploader(
    "Upload your paragraph (.txt file)",
    type=["txt"]
)

if uploaded_file:
    paragraph = uploaded_file.read().decode("utf-8")

    st.success("Paragraph uploaded successfully!")

    with st.expander("View uploaded paragraph"):
        st.write(paragraph)

    question = st.text_input("Enter your question:")

    if st.button("Get Answers"):
        if question.strip():

            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            sentences = [s.strip() for s in sentences if s.strip()]

            if not sentences:
                st.warning("The uploaded file is empty.")
            else:
                embeddings = model.encode(
                    sentences,
                    convert_to_tensor=True
                )

                question_embedding = model.encode(
                    question,
                    convert_to_tensor=True
                )

                scores = util.cos_sim(
                    question_embedding,
                    embeddings
                )[0]

                best_index = scores.argmax().item()
                retrieved_answer = sentences[best_index]

                st.subheader("1. Retrieved Answer")
                st.info(retrieved_answer)

                prompt = (
                    f"Answer the question using only the context. "
                    f"Give a clear and complete answer. "
                    f"Context: {paragraph} "
                    f"Question: {question}"
                )

                with st.spinner("Generating answer..."):
                    generated = qa(
                        prompt,
                        max_new_tokens=120,
                        do_sample=False
                    )

                st.subheader("2. Generative Answer")
                st.success(generated[0]["generated_text"])

        else:
            st.warning("Please enter a question.")

else:
    st.info("Please upload a .txt file to begin.")
