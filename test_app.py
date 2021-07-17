from fastapi.testclient import TestClient
from main import app


# def test_ping():
#     with TestClient(app) as client:
#         response = client.get("/ping")
#         assert response.status_code == 200
#         assert response.json() == {"ping":"pong"}

def test_pred_virginica():
    payload = {
        "Status_of_existing_checking_account": "<0 DM",
        "Duration_in_month": 6,
        "Credit_history": "critical account",
        "Purpose": "radio/television",
        "Credit_amount": 1169,
        "Savings_account_bonds": "no savings account",
        "Present_employment_since": ">=7 years",
        "Installment_rate_in_percentage_of_disposable_income": 4,
        "Personal_status_and_sex": "male:single",
        "Other_debtors_guarantors": "none",
        "Present_residence_since": 4,
        "Property": "real estate",
        "Age_in_years": 67,
        "Other_installment_plans": "none",
        "Housing": "own",
        "Number_of_existing_credits_at_this_bank": 2,
        "Job": "skilled employee / official",
        "Number_of_people_being_liable_to_provide_maintenance_for": 1,
        "Telephone": "yes",
        "foreign_worker": "yes"
    }
    with TestClient(app) as client:
        response = client.post('/predict_credit_risk', json=payload)
        print(response.json())
        assert response.status_code == 200

