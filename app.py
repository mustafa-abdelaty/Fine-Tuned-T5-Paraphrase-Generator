import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Path to your saved model
MODEL_PATH = "saved_t5_model"

# Load model only once
@st.cache_resource
def load_model():
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return tokenizer, model, device

tokenizer, model, device = load_model()


def generate_paraphrases(
    sentence,
    num_return_sequences=4,
    num_beams=5,
    max_length=128
):

    input_text = "paraphrase: " + sentence

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length + 20,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        top_k=100,
        top_p=0.9,
        temperature=1.0,
        early_stopping=True
    )

    paraphrases = []

    for output in outputs:
        text = tokenizer.decode(output, skip_special_tokens=True)

        if text not in paraphrases:
            paraphrases.append(text)

    return paraphrases


st.set_page_config(
    page_title="T5 Paraphrase Generator",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 T5 Paraphrase Generator")

st.write("Generate multiple paraphrases using your fine-tuned T5 model.")

sentence = st.text_area(
    "Enter a sentence",
    height=150
)

if st.button("Generate"):

    if sentence.strip() == "":
        st.warning("Please enter a sentence.")

    else:

        with st.spinner("Generating..."):

            results = generate_paraphrases(sentence)

        st.success("Done!")

        for i, result in enumerate(results, 1):
            st.subheader(f"Paraphrase {i}")
            st.write(result)