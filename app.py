
import streamlit as st
import pandas as pd
import joblib

# Định nghĩa lại class InfectionPredictor để load mô hình
class InfectionPredictor:
    def __init__(self, pathogen_model, abx_models, feature_cols):
        self.pathogen_model = pathogen_model
        self.abx_models = abx_models
        self.feature_cols = feature_cols

    def predict_pathogen(self, input_data):
        return self.pathogen_model.predict(input_data)[0]

    def suggest_antibiotics(self, input_data):
        suggestions = []
        for abx_name, model in self.abx_models.items():
            if model.predict(input_data)[0] == 1:
                suggestions.append(abx_name)
        return suggestions

    def predict_all(self, input_data):
        pathogen = self.predict_pathogen(input_data)
        antibiotics = self.suggest_antibiotics(input_data)
        return {"pathogen": pathogen, "antibiotics": antibiotics}

# Load mô hình gộp
predictor = joblib.load("infection_model.pkl")
feature_cols = predictor.feature_cols

st.title("AI Chẩn đoán Tác nhân & Gợi ý Kháng sinh")

st.markdown("### Nhập dữ liệu bệnh nhân:")

user_input = {}
for col in feature_cols:
    if col == "Tuoi":
        user_input[col] = st.number_input("Tuổi (năm)", min_value=0.0, max_value=100.0, step=1.0)
    elif col in ["Nhiet do", "Bach cau", "CRP", "Nhip tho", "Mach"]:
        user_input[col] = st.number_input(f"{col}", value=0.0)
    else:
        user_input[col] = st.selectbox(f"{col}", ["Không", "Có"]) == "Có"

# Dự đoán khi người dùng nhấn nút
if st.button("Dự đoán"):
    input_df = pd.DataFrame([user_input])
    for col in input_df.columns:
        if isinstance(input_df[col].iloc[0], bool):
            input_df[col] = input_df[col].astype(int)

    result = predictor.predict_all(input_df)

    st.success(f"✅ Tác nhân gây bệnh dự đoán: **{result['pathogen']}**")

    st.markdown("### 💊 Kháng sinh gợi ý nên sử dụng:")
    if result["antibiotics"]:
        for abx in result["antibiotics"]:
            st.write(f"- **{abx}**")
    else:
        st.info("Không có kháng sinh nào được gợi ý.")
