import torch
import string
#import streamlit as st
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast


@st.cache(allow_output_mutation=True)
def get_model():
    model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
    model.eval()
    return model

tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                    bos_token='</s>',
                                                    eos_token='</s>',
                                                    unk_token='<unk>',
                                                    pad_token='<pad>',
                                                    mask_token='<mask>')


default_text = "현대인들은 왜 항상 불안해 할까?"

N_SENT = 3

model = get_model()
st.title("KoGPT2 Demo Page(ver 2.0)")

st.markdown("""
### 모델
| Model       |  # of params |   Type   | # of layers  | # of heads | ffn_dim | hidden_dims | 
|--------------|:----:|:-------:|--------:|--------:|--------:|--------------:|
| `KoGPT2` |  125M  |  Decoder |   12     | 12      | 3072    | 768 | 
### 샘플링 방법
- greedy sampling
- 최대 출력 길이 : 128/1,024
## Conditional Generation
""")

text = st.text_area("Input Text:", value=default_text)
st.write(text)
punct = ('!', '?', '.')

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

