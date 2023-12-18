from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from secret_key import openapi_key

import os
os.environ['OPENAI_API_KEY'] = openapi_key

llm = OpenAI(temperature=0.7)

def generate_restaurant_name_and_items(culture, location):
    # best city_name for X culture in Y location
    prompt_template_city_name = PromptTemplate(
        input_variables = ['culture', 'location'],
        template = "I want to open a {culture} restaurant in {location}, Suggest a fancy city for this."
    )
    city_name_chain = LLMChain(llm=llm, prompt=prompt_template_city_name, output_key="city_name")

    # retaurant_name for that city
    prompt_template_res_name = PromptTemplate(
        input_variables = ['city_name'],
        template = "Suggest a restaurant name in {city_name}."
    )
    retaurant_name_chain = LLMChain(llm=llm, prompt=prompt_template_res_name, output_key="restaurant_name")

    # sugested menu_items for that restaurant
    prompt_template_menu_items = PromptTemplate(
        input_variables = ['restaurant_name'],
        template="Suggest some menu items for {restaurant_name}."
    )
    food_items_chain =LLMChain(llm=llm, prompt=prompt_template_menu_items, output_key="menu_items")

    # generate chain
    chain = SequentialChain(
        chains = [city_name_chain, retaurant_name_chain, food_items_chain],
        input_variables = ['culture', 'location'],
        output_variables = ['city_name', 'restaurant_name', "menu_items"]
    )

    response  = chain({"culture": culture, "location":location})
    
    return response

if __name__ == "__main__":
    print(generate_restaurant_name_and_items("Italian", "India"))
