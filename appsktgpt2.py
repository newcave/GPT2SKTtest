import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Config, PreTrainedTokenizerFast, pipeline

st.title("Korean GPT-2 Example")

tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                    bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                                                    pad_token='<pad>', mask_token='<mask>')

model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

default_text = "안녕하세요? 한국수자원공사입니다."
text = st.text_area("Input Text:", value=default_text)

punct = ('!', '?', '.')

max_length = st.slider("최대 문장 길이", 64, 256, 128, step=64)
top_k = st.slider("Top K", 10, 100, 50, step=5)
#repetition_penalty = st.slider("Repetition Penalty", 2, 4, 2, step=1)
#no_repeat_ngram_size = st.slider("No Repeat Ngram Size", 1, 10, 4, step=1)
eos_token_id = st.slider("EOS Token ID", 1, 1000, 375, step=1)
max_new_tokens = st.slider("Max New Tokens", 1, 128, 64, step=1)
#do_sample = st.checkbox("Do Sample", value=True)
early_stopping = st.checkbox("Early Stopping", value=True)

if st.button("Generate"):
    if text:
        st.markdown("## 결과를 표출합니다. ")
        with st.spinner('잠시만 더 기다려주세요~!'):
            print(f'input > {text}') 
            input_ids = tokenizer(text)['input_ids']
            gen_ids = model.generate(torch.tensor([input_ids]),
                                      max_length=max_length,
                                      num_beams=4,
#                                      repetition_penalty=2,
                                      no_repeat_ngram_size=3,
                                      eos_token_id=eos_token_id,
                                      max_new_tokens=max_new_tokens,
                                      # do_sample=true,
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

generated = ""
        

# GPT환경변수 설정입니다.

st.write("| Parameter | 주요 GPT환경변수 설명입니다. |")
st.write("| ----------------------------------------|")
st.write("| `max_length` | 생성할 최대 길이를 결정합니다. |")
st.write("| `top_k` | 다음 단어의 후보군을 결정할 때 사용되는 상위 k 개의 단어를 선택합니다. |")
st.write("| `repetition_penalty` | 반복을 방지하기 위해 사용되는 패널티 값입니다. 이 값이 높을수록 모델은 중복 단어를 생성하지 않으려고 노력합니다. |")
st.write("| `eos_token_id` | 문장의 끝을 나타내는 토큰 ID입니다. |")
st.write("| `max_new_tokens` | 생성할 최대 토큰 수입니다. 이 값을 초과하는 경우 모델은 생성을 중지합니다. |")
st.write("| `early_stopping` | 생성된 문장이 `eos_token_id`를 포함하면 조기 종료합니다. |")


        
if st.button("Copy Text"):
    if generated:
        st.code(generated)
        st.write("Please manually copy the text above.")

if st.button("Reset Input"):
    st.text_area("Input Text:", value=default_text)
