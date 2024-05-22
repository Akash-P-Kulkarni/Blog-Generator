import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

# Function to get response
def generateBlog(input_text, no_words, blog_style):
    # Initialize the model
    try:
        llm = CTransformers(
            model='models\llama-2-7b-chat.ggmlv3.q8_0.bin',
            model_type='llama',
            config={
                'max_new_tokens': 256,
                'temperature': 0.5
            }
        )
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # Define the prompt template
    template = '''Write a blog for {blog_style} job profile 
    for a topic {input_text} 
    within {no_words} words'''

    prompt = PromptTemplate(
        input_variables=['blog_style', 'input_text', 'no_words'],
        template=template
    )

    formatted_prompt = prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words)
    
    # Generate response using the invoke method
    try:
        response = llm.invoke(formatted_prompt)
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return

    return response

# Streamlit app configuration
st.set_page_config(
    page_title='Generate Blog',
    layout='centered',
    initial_sidebar_state='collapsed'
)

st.header('Blog Generator')

# User inputs
input_text = st.text_input('Enter the blog topic', placeholder='Blog topic')
col1, col2 = st.columns([2, 5])

with col1:
    no_words = st.number_input('Number of words', value=100, min_value=100, step=1)
with col2:
    blog_style = st.selectbox('Writing the blog for', ('Researchers', 'Data Scientist', 'Common People'), index=0)

submit = st.button('Generate')

# Generate and display the blog
if submit:
    if input_text and no_words and blog_style:
        response = generateBlog(input_text, no_words, blog_style)
        if response:
            st.write(response)
    else:
        st.error("Please fill all the input fields")
