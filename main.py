import streamlit as st
#import torch
import pdb
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import certifi
print(certifi.where())
# pdb.set_trace()
import os
local_model_directory = r""
local_tokenizer_directory = r""
# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "roneneldan/TinyStories-33M",
    #torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-33M")
# model = AutoModelForCausalLM.from_pretrained(
#     local_model_directory,
#     trust_remote_code=True
#     #torch_dtype=torch.bfloat16,
# )
# tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_directory)
# Create a pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=100000,
    do_sample=False
)
st.info("Gen AI small use case! story generation")
st.header('Generate a story with small statements')
a=st.chat_input(placeholder='your inital story statement')
st.header('Output:')
if a:
    st.success('Story generated')
    st.write(f"{a} {pipe(a)[0]['generated_text']}")
