import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Config, PreTrainedTokenizerFast, pipeline
from streamlit.ReportThread import get_report_ctx
from streamlit.server.Server import Server

def main():
    st.title("Korean GPT-2 Example")

    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                        bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                                                        pad_token='<pad>', mask_token='<mask>')

    model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

    default_text = "대한민국 만세?"
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
            st.code(generated)
            st.write("Please manually copy the text above.")

    if st.button("Reset Input"):
        ctx = get_report_ctx()
        this_session = None
        session_infos = Server.get_current()._session_infos.values()

        for session_info in session_infos:
            s = session_info.session
            if ctx.session_id == s.id:
                this_session = s

        if this_session is None:
            raise RuntimeError("Could not get a handle on the current Streamlit session.")

        this_session.request_rerun()

if __name__ == "__main__":
    main()
