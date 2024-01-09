import streamlit as st
import langchain_helper

st.title("Best Menu Generator")

# for try
# st.sidebar.selectbox("Pick a Cuisine", ("Indian", "Italian", "Mexican", "Arabic", "American"))

culture = st.sidebar.selectbox("Select a food culture", ("Kurdish", "Turkish", "Indian", "Fine Dining", "Mexican", "Europe", "American"))

location = st.sidebar.selectbox("Pick a Cuisine", ("Turkey", "India", "Germany", "Saudi Arabia", "USA"))


# # for try
# def generate_restaurant_name_and_items(culture, location):
#     return {
#         'city_name':'Ahmedabad',
#         'restaurant_name':'Curry Delight',
#         'menu_items':'samosa, paneer tikka'
#          }
# if culture and location:
#     response = generate_restaurant_name_and_items(culture, location)
#     st.header(response['restaurant_name'])
#     st.write("in")
#     st.header(response['city_name'])
#     menu_items = response['menu_items'].split(",")
#     st.write("---Menu Items---")
#     for item in menu_items:
#         st.write("-", item)

if culture and location:
    response = langchain_helper.generate_restaurant_name_and_items(culture, location)
    st.header(response['restaurant_name'])
    menu_items = response['menu_items'].strip().split(",")
    st.write("**Menu Items**")
    for item in menu_items:
        st.write("-", item)

