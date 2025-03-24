from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
from collections import defaultdict

app = FastAPI()

# ✅ Allow frontend to access backend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load Dataset + AI Model
cafeteria_data = pd.read_csv("Cafeteria_Combined_Dataset__Food___Raw_Material___Nationality_.csv")
food_demand_model = joblib.load("food_demand_forecast_model.pkl")

# Constants
weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
meal_times = ["Breakfast", "Lunch", "Dinner"]
food_items = cafeteria_data['Food_Item'].unique()
nationalities = cafeteria_data['Nationality'].unique()

# ✅ Backend root
@app.get("/")
async def root():
    return {"message": "Backend running ✅"}

# 1️⃣ Waste summary dropdown toggle
@app.get("/waste_summary")
async def waste_summary(waste_type: str):
    if waste_type == "food":
        summary = cafeteria_data.groupby('Food_Item').agg({
            'Prepared_Quantity': 'sum',
            'Wasted_Quantity': 'sum'
        }).reset_index()
        summary['Waste_Percentage'] = round((summary['Wasted_Quantity'] / summary['Prepared_Quantity']) * 100, 2)
        return summary.to_dict(orient='records')

    elif waste_type == "raw":
        summary = cafeteria_data.groupby('Raw_Material').agg({
            'Material_Used_kg': 'sum',
            'Material_Wasted_kg': 'sum'
        }).reset_index()
        summary['Waste_Percentage'] = round((summary['Material_Wasted_kg'] / summary['Material_Used_kg']) * 100, 2)
        return summary.to_dict(orient='records')

    else:
        return {"error": "Invalid waste_type, use 'food' or 'raw'"}

# 2️⃣ AI-driven weekly forecast with smart model
@app.get("/weekly_forecast")
async def weekly_forecast():
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
                    features = pd.get_dummies(input_data).reindex(columns=food_demand_model.feature_names_in_, fill_value=0)
                    pred = food_demand_model.predict(features)[0]
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
    ).head(5)

    return {
        "recommended_food": predictions,
        "raw_materials_required": raw_summary.to_dict(orient='records'),
        "top_nationalities": top_nats.to_dict(orient='records')
    }

# 3️⃣ Weekly collapsible grid API
@app.get("/weekly_menu_grid")
async def weekly_menu_grid():
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

    return grid
