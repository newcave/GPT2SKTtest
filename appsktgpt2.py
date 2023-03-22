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

                
num_beams = st.slider("Num Beams", 1, 10, 4, step=1)
repetition_penalty = st.slider("Repetition Penalty", 1.0, 3.0, 2.0, step=0.1)
no_repeat_ngram_size = st.slider("No Repeat Ngram Size", 1, 10, 4, step=1)
eos_token_id = st.slider("EOS Token ID", 1, 1000, 375, step=1)
max_new_tokens = st.slider("Max New Tokens", 1, 128, 64, step=1)
do_sample = st.checkbox("Do Sample", value=True)
top_k = st.slider("Top K", 1, 100, 50, step=1)
early_stopping = st.checkbox("Early Stopping", value=True)
       

# 첫 번째 행
st.write("| Parameter | Description |")
st.write("| --- | --- |")
st.write("| `num_beams` | Beam search를 사용하여 생성할 문장의 수를 결정합니다. 높은 값일수록 생성된 문장이 다양해지지만, 실행 속도가 느려질 수 있습니다. |")
st.write("| `repetition_penalty` | 반복을 방지하기 위해 사용되는 패널티 값입니다. 이 값이 높을수록 모델은 중복 단어를 생성하지 않으려고 노력합니다. |")
st.write("| `no_repeat_ngram_size` | 중복되지 않도록 유지할 n-gram의 최대 크기를 결정합니다. 예를 들어, `no_repeat_ngram_size`가 3이면, 3-gram 중복이 제거됩니다. |")
st.write("| `eos_token_id` | 문장의 끝을 나타내는 토큰 ID입니다. |")
st.write("| `max_new_tokens` | 생성할 최대 토큰 수입니다. 이 값을 초과하는 경우 모델은 생성을 중지합니다. |")

# 두 번째 행
st.write("| Parameter | Description |")
st.write("| --- | --- |")
st.write("| `do_sample` | 샘플링 방법을 선택하는 옵션입니다. 이 값을 `True`로 설정하면 확률적으로 샘플링하여 다양한 문장을 생성할 수 있습니다. |")
st.write("| `top_k` | 다음 단어의 후보군을 결정할 때 사용되는 상위 k 개의 단어를 선택합니다. |")
st.write("| `early_stopping` | 생성된 문장이 `eos_token_id`를 포함하면 조기 종료합니다. |")
st.write("| `max_length` | 생성할 최대 길이를 결정합니다. |")
st.write("| `repetition_penalty` | 중복 단어에 대한 패널티를 결정합니다. 이 값이 높을수록 모델은 중복 단어를 생성하지 않으려고 노력합니다. |")



num_beams = st.slider("Num Beams", 1, 10, 4, step=1)
repetition_penalty = st.slider("Repetition Penalty", 1.0, 3.0, 2.0, step=0.1)
no_repeat_ngram_size = st.slider("No Repeat Ngram Size", 1, 10, 4, step=1)
eos_token_id = st.slider("EOS Token ID", 1, 1000, 375, step=1)
max_new_tokens = st.slider("Max New Tokens", 1, 128, 64, step=1)
do_sample = st.checkbox("Do Sample", value=True)
top_k = st.slider("Top K", 1, 100, 50, step=1)
early_stopping = st.checkbox("Early Stopping", value=True)

        
if st.button("Copy Text"):
    if generated:
        st.code(generated)
        st.write("Please manually copy the text above.")

if st.button("Reset Input"):
    st.text_area("Input Text:", value=default_text)
