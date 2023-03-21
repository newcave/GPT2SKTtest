import streamlit as st
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast, pipeline

st.title("한국어 GPT-2 텍스트 생성기")

# Load tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                    bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                                                    pad_token='<pad>', mask_token='<mask>')

# Load model
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

# Set up text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
generation_args = dict(
    num_beams=4,
    repetition_penalty=2.0,
    no_repeat_ngram_size=4,
    eos_token_id=375, # \n
    max_new_tokens=64,
    do_sample=True,
    top_k=50,
    early_stopping=True
)

input_text = st.text_area("텍스트를 입력하세요:", value="0 :", max_chars=300)

if st.button("텍스트 생성"):
    generated_text = generator(input_text, **generation_args)[0]['generated_text']
    st.write(f"생성된 텍스트:\n{generated_text}")
