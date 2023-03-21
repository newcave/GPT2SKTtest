import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Config, PreTrainedTokenizerFast, pipeline

st.title("Korean GPT-2 Example")

tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                    bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                                                    pad_token='<pad>', mask_token='<mask>')

model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
generation_args = dict(
    num_beams=4,
    repetition_penalty=2.0,
    no_repeat_ngram_size=4,
    eos_token_id=tokenizer.eos_token_id,
    max_length=64,
    do_sample=True,
    top_k=50,
    early_stopping=True
)

if st.button("Generate"):
    examples = [
        "0 : **는 게임 좋아하니\n1 :",
        "0 : 어제 강남에서 살인사건 났대 ㅜㅜ 너무 무서워\n1 : 헐 왜? 무슨 일 있었어?\n0 : 사진보니까 막 피흘리는 사람있고 경찰들이 떠서 제압하고 난리도 아니었다던데??\n1 :",
        "0 : 자기야 어제는 나한테 왜 그랬어?\n1 : 뭔 일 있었어?\n0 : 어떻게 나한테 말도 없이 그럴 수 있어? 나 진짜 실망했어\n1 : "
    ]
    
    generated_text = generator(examples, **generation_args)
    
    for idx, text in enumerate(generated_text):
        st.write(f"Example {idx + 1}:")
        st.write(text['generated_text'])
