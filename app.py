
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
api_key = st.secrets['GROQ_API_KEY']
load_dotenv()

st.set_page_config(
    page_title="Website RAG Chatbot",
    page_icon="🌐",
    layout="wide"
)

st.title("🌐 Website RAG Chatbot")
st.write("Enter any website URL and ask questions about its content.")

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "website_url" not in st.session_state:
    st.session_state.website_url = ""

if "messages" not in st.session_state:
    st.session_state.messages = []

# URL input
url = st.text_input(
    "Enter Website URL",
    placeholder="https://example.com"
)

# Create database from the provided website
if st.button("Load Website", type="primary"):

    if not url.strip():
        st.error("Please enter a website URL.")

    elif not url.strip().startswith(("http://", "https://")):
        st.error("Please enter a valid URL starting with http:// or https://")

    else:
        try:
            with st.spinner("Loading website and creating database..."):

                # Load website
                loader = WebBaseLoader(url.strip())
                docs = loader.load()

                if not docs or not any(
                    doc.page_content.strip() for doc in docs
                ):
                    st.error("No readable content found on this website.")
                    st.stop()

                # Split documents
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = splitter.split_documents(docs)

                # Create embeddings
                embedding_model = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

                # Create vector database in memory
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embedding_model
                )

                # Save in session state
                st.session_state.vectorstore = vectorstore
                st.session_state.website_url = url.strip()
                st.session_state.messages = []

            st.success(
                f"Website loaded successfully! Created {len(chunks)} chunks."
            )

        except Exception as e:
            st.error(f"Error loading website: {e}")

# Chat interface
if st.session_state.vectorstore is not None:

    st.divider()

    st.subheader("💬 Ask Questions")
    st.caption(f"Website: {st.session_state.website_url}")

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    query = st.chat_input("Ask something about this website...")

    if query:
        # Display user question
        st.session_state.messages.append(
            {"role": "user", "content": query}
        )

        with st.chat_message("user"):
            st.markdown(query)

        try:
            with st.chat_message("assistant"):
                with st.spinner("Finding the answer..."):

                    # MMR retriever
                    retriever = (
                        st.session_state.vectorstore.as_retriever(
                            search_type="mmr",
                            search_kwargs={
                                "k": 4,
                                "fetch_k": 10,
                                "lambda_mult": 0.5
                            }
                        )
                    )

                    # Retrieve relevant documents
                    docs = retriever.invoke(query)

                    context = "\n\n".join(
                        doc.page_content for doc in docs
                    )

                    # LLM
                    llm = ChatGroq(
                        model="qwen/qwen3.8-27b",api_key=api_key
                    )

                    # Prompt template
                    prompt = ChatPromptTemplate.from_messages([
                        (
                            "system",
                            """You are a helpful AI assistant.
                            Use only the provided context to answer.
                            If the answer is not present in the context,
                            say: I could not find answer in the document."""
                        ),
                        (
                            "human",
                            "Context:\n{context}\n\nQuestion:\n{question}"
                        )
                    ])

                    final_prompt = prompt.invoke({
                        "context": context,
                        "question": query
                    })

                    response = llm.invoke(final_prompt)
                    answer = response.content

                    st.markdown(answer)

                    # Show retrieved sources
                    with st.expander("View retrieved context"):
                        for i, doc in enumerate(docs, start=1):
                            st.markdown(f"**Source chunk {i}**")
                            st.write(doc.page_content)
                            st.caption(str(doc.metadata))

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )

        except Exception as e:
            st.error(f"Error generating answer: {e}")

else:
    st.info("Enter a website URL and click 'Load Website' to get started.")
