from joblib import dump, load
import copy
import pandas as pd


class CreditScoreModel:
    def __init__(self):
        self.model = load("./data_model/models/credit_risk.model")

    classes = ("Good Risk", "Bad Risk")

    def numerate_data(self, query_data):
        x = [query_data.dict()]
        data = {k:[] for k in self.column_data.keys()}
        for obj in x:
            for key in obj.keys():
                if type(obj[key]) != int:
                    data[self.key_map[key]].append(self.column_data[self.key_map[key]].index(obj[key]))
                else:
                    data[self.key_map[key]].append(obj[key])
        return pd.DataFrame(data)

    
    def predict(self, query_data):
        x = self.numerate_data(query_data)
        prediction = self.model.predict(x)[0]
        return self.classes[prediction]

    # function to retrain the model as part of the feedback loop
    def retrain(self, data):
        # pull out the relevant X and y from the FeedbackIn object
        X = [list(d.dict().values())[:-1] for d in data]
        y = [list(d.risk_class) for d in data]

        # fit the classifier again based on the new data obtained
        self.model.fit(X, y)

    column_data = {
        "Status of existing checking account": [
            "<0 DM",
            "0 <= <200 DM",
            "no checking account",
            ">= 200 DM "
        ],
        "Credit history": [
            "critical account",
            "existing credits paid back duly till now",
            "delay in paying off",
            "no credits taken",
            "all credits at this bank paid back duly"
        ],
        "Purpose": [
            "radio/television",
            "education",
            "furniture/equipment",
            "car (new)",
            "car (used)",
            "business",
            "domestic appliances",
            "repairs",
            "others",
            "retraining"
        ],
        "Savings account/bonds": [
            "no savings account",
            "<100 DM",
            "500 <= < 1000 DM",
            ">= 1000 DM",
            "100 <= <500 DM"
        ],
        "Present employment since": [
            ">=7 years",
            "1<= < 4 years",
            "4<= <7 years",
            "unemployed",
            "<1 years"
        ],
        "Personal status and sex": [
            "male:single",
            "female:divorced/separated/married",
            "male:divorced/separated",
            "male:married/widowed"
        ],
        "Other debtors / guarantors": [
            "none",
            "guarantor",
            "co-applicant"
        ],
        "Property": [
            "real estate",
            "savings agreement/life insurance",
            "unknown / no property",
            "car or other"
        ],
        "Other installment plans": [
            "none",
            "bank",
            "store"
        ],
        "Housing": [
            "own",
            "for free",
            "rent"
        ],
        "Job": [
            "skilled employee / official",
            "unskilled - resident",
            "management/ highly qualified employee",
            "unemployed/ unskilled  - non-resident"
        ],
        "Telephone": [
            "yes",
            "none"
        ],
        "foreign worker": [
            "yes",
            "no"
        ],
        "Duration in month": [],
        "Credit amount": [],
        "Installment rate in percentage of disposable income": [],
        "Present residence since": [],
        "Age in years": [],
        "Number of existing credits at this bank": [],
        "Number of people being liable to provide maintenance for": [],
    }

    key_map = {
        "Status_of_existing_checking_account": "Status of existing checking account",
        "Credit_history": "Credit history",
        "Purpose": "Purpose",
        "Savings_account_bonds": "Savings account/bonds",
        "Present_employment_since": "Present employment since",
        "Personal_status_and_sex": "Personal status and sex",
        "Other_debtors_guarantors": "Other debtors / guarantors",
        "Property": "Property",
        "Other_installment_plans": "Other installment plans",
        "Housing": "Housing",
        "Job": "Job",
        "Telephone": "Telephone",
        "foreign_worker": "foreign worker",
        "Duration_in_month": "Duration in month",
        "Credit_amount": "Credit amount",
        "Installment_rate_in_percentage_of_disposable_income": "Installment rate in percentage of disposable income",
        "Present_residence_since": "Present residence since",
        "Age_in_years": "Age in years",
        "Number_of_existing_credits_at_this_bank": "Number of existing credits at this bank",
        "Number_of_people_being_liable_to_provide_maintenance_for": "Number of people being liable to provide maintenance for",
    }
