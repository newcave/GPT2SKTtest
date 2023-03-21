import streamlit as st
import string
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast, pipeline

# Load model and tokenizer
def get_model():
    model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
    model.eval()
    return model

# model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
model = get_model()
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                    bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                                                    pad_token='<pad>', mask_token='<mask>')

# Set up text generation pipeline
#generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
#generation_args = dict(
#    num_beams=4,
#    repetition_penalty=2.0,
#    no_repeat_ngram_size=4,
#    eos_token_id=375,  # \n
#    max_new_tokens=64,
#    do_sample=True,
#    top_k=50,
#    early_stopping=True
#)

st.title("한국어 텍스트 생성기")
#input_text = st.text_input("시작 텍스트를 입력하세요:")
#generate_button = st.button("생성")

#if generate_button:
#    if input_text:
#        generated_text = generator(input_text, **generation_args)
#        st.write(generated_text[0]['generated_text'])
#    else:
#        st.write("생성할 텍스트를 입력하세요.")
        
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
        
