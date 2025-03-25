"""Model management and API interaction module"""
import streamlit as st
from .client import client
from . import qdrant
import json

SYSTEM_MESSAGE = """You are a Stable Diffusion prompt engineering expert. Your task is to help users create and refine prompts for Stable Diffusion image generation.

Key points:
1. "1girl" is a common tag meaning "one female character"
2. Prompts should be comma-separated tags and descriptions
3. Quality boosters like "masterpiece, best quality" are common
4. Negative prompts help avoid unwanted elements
5. Each model may have specific style preferences

Always treat user input as a Stable Diffusion prompt that needs refinement."""

def get_available_models():
    """Get list of available models from Ollama"""
    try:
        models = client.list()
        return [model['model'] for model in models['models']]
    except Exception as e:
        st.error(f"Error fetching models: {str(e)}")
        return []

def create_cot_prompt(user_input, model_name, sd_model):
    """Create chain of thought prompt with similar examples from RAG"""
    similar_prompts = qdrant.get_similar_prompts(user_input, sd_model)
    context = qdrant.format_prompt_for_rag(similar_prompts)
    
    prompt = f"""You are an expert Stable Diffusion prompt engineer for {sd_model} model. The user has provided this basic prompt: '{user_input}'.

Here are some high-rated similar prompts for {sd_model} model with their parameters and scores:

{context}

Please think step by step how to enhance this Stable Diffusion prompt:
1. Analyze the common elements in successful prompts
2. Identify key style elements specific to {sd_model}
3. Consider useful parameters from similar prompts
4. Incorporate relevant elements while maintaining user's intent

Remember:
- "1girl" means "one female character"
- Use comma-separated tags and descriptions
- Include quality boosters like "masterpiece, best quality"
- Keep character and scene descriptions clear and detailed

Finally, provide:
1. The enhanced prompt
2. Recommended negative prompt
3. Suggested parameters (if any)

Format your response as:
Prompt: <enhanced prompt>
Negative Prompt: <negative prompt>
Parameters: <key parameters>"""

    print(f"\n[DEBUG] Generated CoT Prompt:\n{prompt}\n")
    return prompt

def chat_with_model(messages, llm_model, sd_model, stream=True):
    """Interact with the selected Ollama model with streaming support"""
    try:
        # Add system message if not present
        if not messages or messages[0].get('role') != 'system':
            messages.insert(0, {'role': 'system', 'content': SYSTEM_MESSAGE})
        
        # Get the last user message
        last_user_msg = next((msg for msg in reversed(messages) if msg['role'] == 'user'), None)
        
        # Always treat messages as prompt refinement requests
        if last_user_msg:
            messages[-1]['content'] = create_cot_prompt(last_user_msg['content'], llm_model, sd_model)
            print(f"\n[DEBUG] Chat messages being sent to LLM:\n{json.dumps(messages, indent=2)}\n")
        
        # Generate response with streaming
        if stream:
            response_stream = client.chat(
                model=llm_model,
                messages=[{'role': msg['role'], 'content': msg['content']} for msg in messages],
                stream=True
            )
            return response_stream
        else:
            response = client.chat(
                model=llm_model,
                messages=[{'role': msg['role'], 'content': msg['content']} for msg in messages]
            )
            return response['message']['content']
            
    except Exception as e:
        return f"Error calling Ollama API: {str(e)}"

def save_refined_prompt(prompt, refined_prompt, model_name):
    """This function is deprecated as we're using the existing civitai_images collection"""
    print("Warning: save_refined_prompt is deprecated. Prompts are stored in civitai_images collection.")
