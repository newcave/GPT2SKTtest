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
                                repetition_penalty=2.0)
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
        st.markdown(f"<textarea readonly>{generated}</textarea>", unsafe_allow_html=True)
        st.write("Text copied to clipboard.")

if st.button("Reset Input"):
    text = st.text_area("Input Text:", value="")
