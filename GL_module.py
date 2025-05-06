import pandas as pd
import numpy as np
import ast
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from bs4 import BeautifulSoup
import pandas as pd
import time
from thefuzz import fuzz
from thefuzz import process

# No longer used
def load_usda_data(path, name):
    try:
        usda_df = pd.read_csv(path)
        print(f"USDA {name} data loaded successfully.")
        return usda_df
    except Exception as e:
        print(f"Error loading USDA {name} data:", e)
        return pd.DataFrame()
    
# No longer used
def merge_usda_data(usda_food_df, usda_food_nutrient_df, usda_nutrient_df):
    carb_id = [1005]
    food_carbs_df = usda_food_nutrient_df[usda_food_nutrient_df['nutrient_id'].isin(carb_id)]
    return pd.merge(food_carbs_df, usda_food_df, on='fdc_id', how='left')

def get_GI_data():
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)
    
    # University of Sydney GI data
    url = "https://glycemicindex.com/gi-search/" 
    driver.get(url)
    
    wait = WebDriverWait(driver, 10)
    wait.until(EC.presence_of_element_located((By.ID, "tablepress-1")))
    
    dropdown = Select(driver.find_element(By.NAME, "tablepress-1_length"))
    dropdown.select_by_value("100")  
    time.sleep(2)
    
    all_entries = []
    
    # Loop through all pages of the table
    while True:
        time.sleep(2)
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        
        table = soup.find("table", {"id": "tablepress-1"})
        if not table:
            print("Table not found!")
            break
        
        header_row = table.find("thead").find("tr")
        headers = [th.get_text(strip=True) for th in header_row.find_all("th")]
        
        tbody = table.find("tbody")
        rows = tbody.find_all("tr")

        for row in rows:
            cells = [td.get_text(strip=True) for td in row.find_all("td")]
            if cells:
                all_entries.append(cells)
        
        try:
            next_button = driver.find_element(By.ID, "tablepress-1_next")
            if "disabled" in next_button.get_attribute("class"):
                print("Reached the last page.")
                break
            else:
                next_button.click()
                time.sleep(2)
        except Exception as e:
            print("Next button not found or error occurred:", e)
            break
            
    driver.quit()
    return pd.DataFrame(all_entries, columns=headers)

def fuzzy_match(text, choices):
    # Check for negligible food items
    if fuzz.token_set_ratio(text, 'salt') > 90:
        return 'salt', -1
    
    match = process.extractOne(text, choices, scorer=fuzz.token_set_ratio, score_cutoff=50)
    if match is None:
        return None, None
    else: 
        return match[0], choices.index(match[0])

def create_recipes_GL_data(foodcom_recipes_df, usda_df, GI_df, csv_path):
    gl_list = []
     
    GI_food = GI_df['Food Name']
    usda_food = usda_df['description'] # No longer used

    GI_food_lower = GI_df['Food Name'].str.lower().tolist()
    usda_food_lower = usda_df['description'].str.lower().tolist() # No longer used

    # Iterate through each recipe in the Food.com dataset and calculate GL
    for idx, recipe in foodcom_recipes_df.iterrows():
        print('recipe')
        recipe_carbs = float(ast.literal_eval(recipe.get('nutrition'))[6])/100.0 * 275.0
        ingr_str = recipe.get('ingredients')
        if pd.isna(ingr_str):
            gl_list.append(np.nan)
            continue

        ingredients = [ing.strip().lower() for ing in ingr_str.split(',')]
        ingr_gi = []
        total_carbs = 0.0
        ingr_carbs = []
        
        # Iterate through each ingredient in the recipe and get GI and carb values
        for ingredient in ingredients:
            # Fuzzy string match against the GI dataset.
            gi_match, gi_idx = fuzzy_match(ingredient, GI_food_lower)
            gi_val = 50 # default gi
            carb_val = 25 # default carb value
            if gi_match is not None:
                if gi_idx == -1:
                    gi_val = 0
                    carb_val = 0
                else:
                    gi_val = float(GI_df.loc[GI_df['Food Name'] == GI_food[gi_idx], 'GI'].values[0])
                    if str(GI_df.loc[GI_df['Food Name'] == GI_food[gi_idx], 'Carbohydrate portion (g) or Average Carbohydrate portion (g)'].values[0]).strip() != '':
                        carb_val = float(GI_df.loc[GI_df['Food Name'] == GI_food[gi_idx], 'Carbohydrate portion (g) or Average Carbohydrate portion (g)'].values[0])

            # # Fuzzy match against the USDA carbonhydrate dataset.
            # usda_match, usda_idx = fuzzy_match(ingredient, usda_food_lower)
            # carb_val = 50
            # if usda_match is not None:
            #     if usda_idx == -1:
            #         carb_val = 0
            #     else:
            #         carb_val = float(usda_df.loc[usda_df['description'] == usda_food[usda_idx], 'amount'].values[0])

            total_carbs += carb_val
            ingr_carbs.append(carb_val)
            ingr_gi.append(gi_val)
        
        # Calculate GL for the recipe
        total_gl = 0.0
        for i in range(len(ingr_carbs)):
            total_gl += ((ingr_carbs[i]/total_carbs)*recipe_carbs) * ingr_gi[i]
        gl_list.append(total_gl/100.0)

    foodcom_recipes_df['GL'] = gl_list
    foodcom_recipes_df.to_csv(csv_path, index=False)
    print(f"New recipe .csv file with GL data saved to {csv_path}.")

