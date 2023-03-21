import streamlit as st
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast, pipeline

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                    bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                                                    pad_token='<pad>', mask_token='<mask>')

# Set up text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
generation_args = dict(
    num_beams=4,
    repetition_penalty=2.0,
    no_repeat_ngram_size=4,
    eos_token_id=375,  # \n
    max_new_tokens=64,
    do_sample=True,
    top_k=50,
    early_stopping=True
)

st.title("Korean Text Generator")
input_text = st.text_input("Enter a starting text:")
generate_button = st.button("Generate")

if generate_button:
    if input_text:
        generated_text = generator(input_text, **generation_args)
        st.write(generated_text[0]['generated_text'])
    else:
        st.write("Please enter some text to generate.")
