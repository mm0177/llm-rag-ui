import streamlit as st
from PIL import Image
import rag as raglib # Reference to local lib script

# Setting page configuration and headers
st.set_page_config(page_title="UI GENERATION APP")
st.markdown("<h1 style='text-align: center;'>Give the Prompts Here and Stable Diffusion will seamlessly generate </h1>", unsafe_allow_html=True)
st.markdown("""
<style>
.big-font {
    font-size:20px !important;
}
</style>
""", unsafe_allow_html=True)



# Step descriptions
st.markdown("Step 1: Searching")
# ... (rest of the markdown content)

# Checking and creating vector index
if 'vector_index' not in st.session_state:
    with st.spinner("Indexing document..."):
        st.session_state.vector_index = raglib.get_index()

# Main interactive section for prompt improvement and image generation
with st.form('Provide Prompts and Image Generation'):
    st.subheader("Prompt Improvement and Image Generation")
    
    # User input for original prompt
    original_prompt = st.text_input("Give your prompt for Stable Diffusion Model:")
    
    # Form submission button
    if st.form_submit_button("Improve and Generate Image"):
        if original_prompt:
            # Semantic search and display of related prompts
            list_prompts = raglib.semantic_search(index=st.session_state.vector_index, original_prompt=original_prompt)
            st.markdown("**Some of the Similar Prompts**")
            for i, prompt in enumerate(list_prompts):
                st.write(f"{i}: {prompt}")
            
            # Prompt selection for further improvement
            number_selected = st.number_input('Select a Prompt from the list for improving the Given Prompt', min_value=0, max_value=len(list_prompts)-1, value=0)
            selected_prompt = list_prompts[number_selected]
            new_prompt = raglib.get_rag_response(original_prompt, selected_prompt)
            st.markdown("Prompt generated from LLM:")
            st.write(new_prompt)

            # Image generation based on the improved prompt
            st.markdown(f"Generating an image using **stable-diffusion-v1-4** with the prompt: *{new_prompt}*")
            with st.spinner('Generating image...'):
                generated_image = raglib.get_image_response(prompt_content=new_prompt)
                st.success('Image generated successfully')

            # Display the generated image
            st.image(generated_image, caption=new_prompt)