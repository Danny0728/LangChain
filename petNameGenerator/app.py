import langchain_helper as lch
import streamlit as st

st.title("Pets name generator")
animal_type = st.sidebar.selectbox("What is your pet?",("Cat","Dog","Hamster","Cow","Rat"))

pet_color = st.sidebar.text_area(max_chars=15,label=f"What color is your {animal_type}?")
if pet_color:
    st.text(lch.generate_pet_name(animal_type=animal_type,pet_color=pet_color))