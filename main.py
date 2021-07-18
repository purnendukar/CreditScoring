import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from ml_utils import CreditScoreModel
from typing import List

app = FastAPI(
    title="Credit Risk",
    docs_url="/"
)

templates = Jinja2Templates(directory="templates/")

class QueryIn(BaseModel):
    Status_of_existing_checking_account: str
    Credit_history: str
    Purpose: str
    Savings_account_bonds: str
    Present_employment_since: str
    Personal_status_and_sex:str
    Other_debtors_guarantors:str
    Property:str
    Other_installment_plans:str
    Housing:str
    Job: str
    Telephone: str
    foreign_worker: str
    Duration_in_month: int
    Credit_amount: int
    Installment_rate_in_percentage_of_disposable_income: int
    Present_residence_since: int
    Age_in_years: int
    Number_of_existing_credits_at_this_bank: int
    Number_of_people_being_liable_to_provide_maintenance_for: int

class QueryOut(BaseModel):
    credit_risk: str

class FeedbackIn(BaseModel):
    Status_of_existing_checking_account: str
    Credit_history: str
    Purpose: str
    Savings_account_bonds: str
    Present_employment_since: str
    Personal_status_and_sex:str
    Other_debtors_guarantors:str
    Property:str
    Other_installment_plans:str
    Housing:str
    Job: str
    Telephone: str
    foreign_worker: str
    Duration_in_month: int
    Credit_amount: int
    Installment_rate_in_percentage_of_disposable_income: int
    Present_residence_since: int
    Age_in_years: int
    Number_of_existing_credits_at_this_bank: int
    Number_of_people_being_liable_to_provide_maintenance_for: int
    risk_class: str

@app.get("/ping")
def ping():
    return {"ping": "pong"}

@app.get("/home")
def get_form(request: Request):
    status = CreditScoreModel.column_data['Status of existing checking account']
    history = CreditScoreModel.column_data['Credit history']
    purpose = CreditScoreModel.column_data['Purpose']
    account = CreditScoreModel.column_data['Savings account/bonds']
    emp = CreditScoreModel.column_data['Present employment since']
    personal_status = CreditScoreModel.column_data['Personal status and sex']
    guarantors = CreditScoreModel.column_data['Other debtors / guarantors']
    property = CreditScoreModel.column_data['Property']
    installments = CreditScoreModel.column_data['Other installment plans']
    housing = CreditScoreModel.column_data['Housing']
    job = CreditScoreModel.column_data['Job']
    tel = CreditScoreModel.column_data['Telephone']
    foreign = CreditScoreModel.column_data['foreign worker']

    result = "Fill above details and click on Predict"
    return templates.TemplateResponse('home.html', context={'request': request, 'result': result, 'status': status, 'history': history, 'purpose' : purpose, 'account' : account, 'emp': emp, 'personal_status': personal_status, 'guarantors': guarantors, 'property': property, 'installments': installments, 'housing': housing, 'job': job, 'tel': tel, 'foreign': foreign})

@app.post("/home")
def predict_credit_risk(request: Request, Status_of_existing_checking_account: str = Form(...), Credit_history: str = Form(...), Purpose: str = Form(...), Savings_account_bonds: str = Form(...), Present_employment_since: str = Form(...), Personal_status_and_sex: str = Form(...), Other_debtors_guarantors: str = Form(...), Property: str = Form(...), Other_installment_plans: str = Form(...), Housing: str = Form(...), Job: str = Form(...), Telephone: str = Form(...), foreign_worker: str = Form(...), Duration_in_month: int = Form(...), Credit_amount: int = Form(...), Installment_rate_in_percentage_of_disposable_income: int = Form(...), Present_residence_since: int = Form(...), Age_in_years: int = Form(...), Number_of_existing_credits_at_this_bank: int = Form(...),  Number_of_people_being_liable_to_provide_maintenance_for: int = Form(...)):
    status = CreditScoreModel.column_data['Status of existing checking account']
    history = CreditScoreModel.column_data['Credit history']
    purpose = CreditScoreModel.column_data['Purpose']
    account = CreditScoreModel.column_data['Savings account/bonds']
    emp = CreditScoreModel.column_data['Present employment since']
    personal_status = CreditScoreModel.column_data['Personal status and sex']
    guarantors = CreditScoreModel.column_data['Other debtors / guarantors']
    property = CreditScoreModel.column_data['Property']
    installments = CreditScoreModel.column_data['Other installment plans']
    housing = CreditScoreModel.column_data['Housing']
    job = CreditScoreModel.column_data['Job']
    tel = CreditScoreModel.column_data['Telephone']
    foreign = CreditScoreModel.column_data['foreign worker']

    query_data = QueryIn(Status_of_existing_checking_account=status[int(Status_of_existing_checking_account)], Credit_history=history[int(Credit_history)], Purpose=purpose[int(Purpose)], Savings_account_bonds=account[int(Savings_account_bonds)], Present_employment_since=emp[int(Present_employment_since)], Personal_status_and_sex=personal_status[int(Personal_status_and_sex)], Other_debtors_guarantors=guarantors[int(Other_debtors_guarantors)], Property=property[int(Property,)], Other_installment_plans=installments[int(Other_installment_plans)], Housing=housing[int(Housing)], Job=job[int(Job)], Telephone=tel[int(Telephone)], foreign_worker=foreign[int(foreign_worker)], Duration_in_month= Duration_in_month, Credit_amount=Credit_amount, Installment_rate_in_percentage_of_disposable_income=Installment_rate_in_percentage_of_disposable_income, Present_residence_since=Present_residence_since, Age_in_years=Age_in_years, Number_of_existing_credits_at_this_bank=Number_of_existing_credits_at_this_bank, Number_of_people_being_liable_to_provide_maintenance_for=Number_of_people_being_liable_to_provide_maintenance_for)
    model = CreditScoreModel()
    output = {'credit_risk': model.predict(query_data)}
    return templates.TemplateResponse('home.html', context={'request': request, 'result': output, 'status': status, 'history': history, 'purpose' : purpose, 'account' : account, 'emp': emp, 'personal_status': personal_status, 'guarantors': guarantors, 'property': property, 'installments': installments, 'housing': housing, 'job': job, 'tel': tel, 'foreign': foreign})

@app.post("/predict_credit_risk", response_model=QueryOut, status_code=200)
def predict_credit_risk(query_data: QueryIn):
    model = CreditScoreModel()
    output = {'credit_risk': model.predict(query_data)}
    return output


@app.get("/feedback_loop", status_code=200)
# Route to further train the model based on user input in form of feedback loop
# Payload: FeedbackIn containing the parameters and correct flower class
# Response: Dict with detail confirming success (200)
def feedback_loop(request: Request):
    status = CreditScoreModel.column_data['Status of existing checking account']
    history = CreditScoreModel.column_data['Credit history']
    purpose = CreditScoreModel.column_data['Purpose']
    account = CreditScoreModel.column_data['Savings account/bonds']
    emp = CreditScoreModel.column_data['Present employment since']
    personal_status = CreditScoreModel.column_data['Personal status and sex']
    guarantors = CreditScoreModel.column_data['Other debtors / guarantors']
    property = CreditScoreModel.column_data['Property']
    installments = CreditScoreModel.column_data['Other installment plans']
    housing = CreditScoreModel.column_data['Housing']
    job = CreditScoreModel.column_data['Job']
    tel = CreditScoreModel.column_data['Telephone']
    foreign = CreditScoreModel.column_data['foreign worker']

    result = "Fill above details and click on Predict"
    return templates.TemplateResponse('home.html', context={'request': request, 'result': result, 'status': status, 'history': history, 'purpose' : purpose, 'account' : account, 'emp': emp, 'personal_status': personal_status, 'guarantors': guarantors, 'property': property, 'installments': installments, 'housing': housing, 'job': job, 'tel': tel, 'foreign': foreign})


@app.post("/feedback_loop", status_code=200)
# Route to further train the model based on user input in form of feedback loop
# Payload: FeedbackIn containing the parameters and correct flower class
# Response: Dict with detail confirming success (200)
def feedback_loop(data: List[FeedbackIn]):
    model = CreditScoreModel()
    model.retrain(data)
    #print(data)
    return {"detail": "Feedback loop successful"}

if __name__ == "__main__":
    uvicorn.run("main:app", host='0.0.0.0', port=8888, reload=True)

