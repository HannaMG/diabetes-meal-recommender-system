from GL_module import create_recipes_GL_data, get_GI_data, load_usda_data, merge_usda_data
from doc2vec_module import load_foodcom_data


def main():
    # Datasets
    usda_food_path = 'Datasets/USDA/food.csv'
    usda_food_nutrient_path = 'Datasets/USDA/food_nutrient.csv'
    usda_nutrient_path = 'Datasets/USDA/nutrient.csv'
    foodcom_recipes_path = 'Datasets/Food.com/RAW_recipes.csv'
    recipes_with_GL_path = 'Datasets/recipes_with_GL.csv'
    
    usda_food_df = load_usda_data(usda_food_path, "food")
    usda_food_nutrient_df = load_usda_data(usda_food_nutrient_path, "food nutrient")
    usda_nutrient_df = load_usda_data(usda_nutrient_path, "nutrient")
    foodcom_recipes_df = load_foodcom_data(foodcom_recipes_path, "recipes")

    usda_df = merge_usda_data(usda_food_df, usda_food_nutrient_df, usda_nutrient_df)
    print("USDA data obtained.")
    print("Columns in the dataset:", usda_df.columns.tolist())
    print(f"Data: {usda_df}")

    GI_df = get_GI_data()
    print("University of Sydney GI data obtained")

    create_recipes_GL_data(foodcom_recipes_df, usda_df, GI_df, recipes_with_GL_path)


if __name__ == '__main__':
    main()