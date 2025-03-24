from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import json
from collections import defaultdict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store precomputed results
precomputed_results = {}

# Function to save data to JSON
def save_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

# Function to train the model and precompute results
def train_and_precompute():
    # Load dataset
    cafeteria_data = pd.read_csv("Cafeteria_Combined_Dataset_Food_Raw_Material_Nationality.csv")

    # Train the model
    cafeteria_data['Waste_Percentage'] = cafeteria_data['Wasted_Quantity'] / cafeteria_data['Prepared_Quantity']
    X = pd.get_dummies(cafeteria_data[['Day', 'Meal_Time', 'Food_Item', 'Nationality', 'Waste_Percentage']])
    y = cafeteria_data['Consumed_Quantity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, "food_demand_forecast_model.pkl")

    # Constants
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    meal_times = ["Breakfast", "Lunch", "Dinner"]
    food_items = cafeteria_data['Food_Item'].unique()
    nationalities = cafeteria_data['Nationality'].unique()

    # Precompute Waste Summary
    food_waste = cafeteria_data.groupby('Food_Item').agg({
        'Prepared_Quantity': 'sum',
        'Wasted_Quantity': 'sum'
    }).reset_index()
    food_waste['Waste_Percentage'] = round((food_waste['Wasted_Quantity'] / food_waste['Prepared_Quantity']) * 100, 2)

    raw_waste = cafeteria_data.groupby('Raw_Material').agg({
        'Material_Used_kg': 'sum',
        'Material_Wasted_kg': 'sum'
    }).reset_index()
    raw_waste['Waste_Percentage'] = round((raw_waste['Material_Wasted_kg'] / raw_waste['Material_Used_kg']) * 100, 2)

    waste_summary = {
        "food_waste": food_waste.to_dict(orient='records'),
        "raw_waste": raw_waste.to_dict(orient='records')
    }

    # Precompute Weekly Forecast
    predictions = []
    for day in weekdays:
        for meal in meal_times:
            for food in food_items:
                total_pred = 0
                for nat in nationalities:
                    input_data = pd.DataFrame([{
                        "Day": day,
                        "Meal_Time": meal,
                        "Food_Item": food,
                        "Nationality": nat,
                        "Waste_Percentage": 0.1  # dummy feature for AI (can later be dynamic)
                    }])
                    features = pd.get_dummies(input_data).reindex(columns=model.feature_names_in_, fill_value=0)
                    pred = model.predict(features)[0]
                    total_pred += pred

                predictions.append({
                    "Day": day,
                    "Meal_Time": meal,
                    "Food_Item": food,
                    "Recommended_Quantity": int(round(total_pred / 10) * 10)
                })

    # Raw material requirement summary
    raw_summary = cafeteria_data.groupby('Raw_Material').agg({
        'Material_Used_kg': 'sum'
    }).reset_index()
    raw_summary['Required_kg'] = raw_summary['Material_Used_kg'].round(2)

    # Top 5 nationalities expected
    top_nats = cafeteria_data['Nationality'].value_counts().reset_index().rename(
        columns={'index': 'Nationality', 'Nationality': 'Count'}
    ).head(3)

    weekly_forecast = {
        "recommended_food": predictions,
        "raw_materials_required": raw_summary.to_dict(orient='records'),
        "top_nationalities": top_nats.to_dict(orient='records')
    }

    # Precompute Weekly Menu Grid
    grouped = defaultdict(lambda: defaultdict(list))
    for _, row in cafeteria_data.iterrows():
        grouped[row['Day']][row['Meal_Time']].append({
            "Food_Item": row['Food_Item'],
            "Recommended_Quantity": int(row['Prepared_Quantity'] * 0.8)
        })

    grid = []
    for day in weekdays:
        day_obj = {"Day": day, "Meals": []}
        for meal in meal_times:
            day_obj["Meals"].append({
                "Meal_Time": meal,
                "Dishes": grouped[day][meal]
            })
        grid.append(day_obj)

    weekly_menu_grid = grid

    # Save all precomputed results
    global precomputed_results
    precomputed_results = {
        "waste_summary": waste_summary,
        "weekly_forecast": weekly_forecast,
        "weekly_menu_grid": weekly_menu_grid
    }

    # Save to JSON file
    save_to_json(precomputed_results, "precomputed_results.json")
    print("Model trained and results precomputed and saved to precomputed_results.json")

# Train and precompute when the server starts
train_and_precompute()

# Backend root
@app.get("/")
async def root():
    return {"message": "Backend running"}

#  Waste summary dropdown toggle
@app.get("/waste_summary")
async def waste_summary(waste_type: str):
    if waste_type == "food":
        return precomputed_results["waste_summary"]["food_waste"]
    elif waste_type == "raw":
        return precomputed_results["waste_summary"]["raw_waste"]
    else:
        return {"error": "Invalid waste_type, use 'food' or 'raw'"}

# AI-driven weekly forecast with smart model
@app.get("/weekly_forecast")
async def weekly_forecast():
    return precomputed_results["weekly_forecast"]

# Weekly collapsible grid API
@app.get("/weekly_menu_grid")
async def weekly_menu_grid():
    return precomputed_results["weekly_menu_grid"]

# Load student nutrition dataset
student_df = pd.read_csv("Student_Nutrition_History.csv")

# Dummy cafeteria menu (today's menu)
today_menu = [
    {"Food_Item": "Grilled Chicken", "Calories": 400, "Protein": 35, "Carbs": 20, "Fats": 15},
    {"Food_Item": "Garden Salad", "Calories": 150, "Protein": 5, "Carbs": 15, "Fats": 7},
    {"Food_Item": "Rice Bowl", "Calories": 500, "Protein": 15, "Carbs": 60, "Fats": 10},
    {"Food_Item": "Fish Tacos", "Calories": 450, "Protein": 25, "Carbs": 30, "Fats": 18},
]

# Ideal macros for students (customize as needed)
ideal_macros = {"Protein": 100, "Carbs": 250, "Fats": 70}


@app.get("/student_recommendation")
async def student_recommendation(student_id: int):
    student = student_df[student_df["Student_ID"] == student_id]
    if student.empty:
        return {"error": "Student not found"}
    
    student = student.iloc[0]
    
    # Calculate nutrient gaps
    gap_protein = ideal_macros["Protein"] - student["Protein (g)"]
    gap_carbs = ideal_macros["Carbs"] - student["Carbs (g)"]
    gap_fats = ideal_macros["Fats"] - student["Fats (g)"]

    nutrient_gaps = {
        "Protein": gap_protein,
        "Carbs": gap_carbs,
        "Fats": gap_fats
    }

    # Sort gaps by largest deficiency
    nutrient_gaps = dict(sorted(nutrient_gaps.items(), key=lambda x: x[1], reverse=True))

    recommended = []
    total_added = {"Calories": 0, "Protein": 0, "Carbs": 0, "Fats": 0}

    # Greedy selection based on gaps
    for nutrient, gap in nutrient_gaps.items():
        if gap <= 0:
            continue
        for item in today_menu:
            if item in recommended:
                continue  # Avoid duplicates
            if item[nutrient] > 0:
                recommended.append(item)
                total_added["Calories"] += item["Calories"]
                total_added["Protein"] += item["Protein"]
                total_added["Carbs"] += item["Carbs"]
                total_added["Fats"] += item["Fats"]
                break  # Pick only one item for each nutrient gap

    # Final totals after meal
    total_calories = student["Calories_Consumed"] + total_added["Calories"]
    total_protein = student["Protein (g)"] + total_added["Protein"]
    total_carbs = student["Carbs (g)"] + total_added["Carbs"]
    total_fats = student["Fats (g)"] + total_added["Fats"]

    return {
        "student_id": int(student_id),
        "recommended_dishes": recommended,
        "nutrition_after_meal": {
            "Calories": int(total_calories),
            "Protein": int(total_protein),
            "Carbs": int(total_carbs),
            "Fats": int(total_fats)
        }
    }


# Optional: root route for testing server
@app.get("/")
async def root():
    return {"message": "Smart Cafeteria Student Recommendation API"}
