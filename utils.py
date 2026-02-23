import os
import pickle
import pandas as pd
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# .env se API keys load karna
load_dotenv()

class DiseaseRiskAnalyzer:
    def __init__(self):
        self.models = {}
        self.feature_lists = {}
        self.diseases = ['lung', 'pancreatic', 'ovarian']
        
        # Models aur Columns load karna
        for d in self.diseases:
            try:
                with open(f'models/{d}_model.pkl', 'rb') as f:
                    self.models[d] = pickle.load(f)
                with open(f'models/{d}_cols.pkl', 'rb') as f:
                    self.feature_lists[d] = pickle.load(f)
            except Exception as e:
                print(f"⚠️ Warning: Could not load {d} model: {e}")

    def get_risk_analysis(self, disease_type, user_input_df):
        model = self.models[disease_type]
        cols = self.feature_lists[disease_type]
        
        # 1. Prediction Logic
        if disease_type == 'ovarian':
            # Ovarian ke liye 3 classes hain (0, 1, 2)
            prediction = int(model.predict(user_input_df)[0])
            risk_map = {0: "Low", 1: "Moderate", 2: "High"}
            risk_level = risk_map.get(prediction, "Unknown")
            # Probability nikalne ke liye
            probs = model.predict_proba(user_input_df)[0]
            risk_score = round(max(probs) * 100, 2)
        else:
            # Lung aur Pancreatic Binary hain (0, 1)
            risk_prob = model.predict_proba(user_input_df)[0][1]
            risk_score = round(risk_prob * 100, 2)
            risk_level = "High" if risk_score > 70 else "Moderate" if risk_score > 35 else "Low"

        # 2. Key Factors (Top 3 features from user input)
        top_factors = list(user_input_df.columns[:3]) 
        
        # 3. AI Explanation using Groq (Naya Model)
        explanation = self.generate_ai_explanation(risk_score, risk_level, top_factors, disease_type)
        
        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "top_factors": top_factors,
            "explanation": explanation
        }

    def generate_ai_explanation(self, score, level, factors, disease):
        try:
            # Groq model update
            llm = ChatGroq(
                model_name="llama-3.3-70b-versatile", 
                groq_api_key=os.getenv("GROQ_API_KEY"),
                temperature=0.3
            )
            
            prompt = (
                f"As an AI Medical Researcher, analyze this {disease} screening result: "
                f"Risk Score {score}%, Level {level}. Influencing factors: {factors}. "
                f"Provide a 2-sentence empathetic summary. Avoid direct diagnosis."
            )
            
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"AI analysis temporarily unavailable. Risk Level: {level}."