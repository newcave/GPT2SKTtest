import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Config, PreTrainedTokenizerFast, pipeline

st.title("Korean GPT-2 Example")

tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                    bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                                                    pad_token='<pad>', mask_token='<mask>')

model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

default_text = ""
text = st.text_area("Input Text:", value=default_text)

num_beams = st.slider("Num Beams", 1, 10, 4, step=1)
repetition_penalty = st.slider("Repetition Penalty", 1.0, 3.0, 2.0, step=0.1)
no_repeat_ngram_size = st.slider("No Repeat Ngram Size", 1, 10, 4, step=1)
eos_token_id = st.slider("EOS Token ID", 1, 1000, 375, step=1)
max_new_tokens = st.slider("Max New Tokens", 1, 128, 64, step=1)
do_sample = st.checkbox("Do Sample", value=True)
top_k = st.slider("Top K", 1, 100, 50, step=1)
early_stopping = st.checkbox("Early Stopping", value=True)

punct = ('!', '?', '.')

generated = ""

if st.button("Generate"):
    if text:
        st.markdown("## Predict")
        with st.spinner('processing..'):
            print(f'input > {text}') 
            input_ids = tokenizer(text)['input_ids']
            gen_ids = model.generate(torch.tensor([input_ids]),
                                      max_length=128,
                                      num_beams=num_beams,
                                      repetition_penalty=repetition_penalty,
                                      no_repeat_ngram_size=no_repeat_ngram_size,
                                      eos_token_id=eos_token_id,
                                      max_new_tokens=max_new_tokens,
                                      do_sample=do_sample,
                                      top_k=top_k,
                                      early_stopping=early_stopping)
            generated = tokenizer.decode(gen_ids[0,:].tolist()).strip()
            if generated != '' and generated[-1] not in punct:
                for i in reversed(range(len(generated))):
                    if generated[i] in punct:
                        break
                generated = generated[:(i+1)]
            print(f'KoGPT > {generated}')
        st.write(generated)

if st.button("Copy Text"):
    if generated:
        st.code(generated)
        st.write("Please manually copy the text above.")

if st.button("Reset Input"):
    st.text_area("Input Text:", value=default_text)
