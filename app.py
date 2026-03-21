import streamlit as st
import requests
import json
import os
import pandas as pd
import plotly.express as px
import re
import numpy as np
import random
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Interview Rejection Insights and Hiring Prediction")

# ---------------------------
# USER DATABASE
# ---------------------------

USER_FILE="users.json"

if not os.path.exists(USER_FILE):
    with open(USER_FILE,"w") as f:
        json.dump({},f)

with open(USER_FILE,"r") as f:
    users=json.load(f)

# ---------------------------
# SESSION
# ---------------------------

if "page" not in st.session_state:
    st.session_state.page="login"

# ---------------------------
# DOMAIN KEYWORDS
# ---------------------------

DOMAIN_KEYWORDS={
"Software Development":["developer","software","engineer","programmer"],
"Data Science":["data","machine learning","ai","analytics"],
"Web Development":["frontend","backend","web"],
"Mobile Development":["android","ios","flutter"],
"Cyber Security":["security","cyber"],
"Cloud Computing":["cloud","aws","azure"],
"Artificial Intelligence":["ai","deep learning"],
"Digital Marketing":["marketing","seo"],
"Finance":["finance","accountant"],
"Human Resources":["hr","recruiter"],
"Sales":["sales"],
"Networking":["network"],
"Database":["database","sql"],
"Testing":["testing","qa"],
"Product Management":["product manager"],
"UI UX Design":["ui","ux","design"],
"Healthcare":["medical","health"],
"Education":["teacher","education"],
"Customer Support":["support","customer"],
"Content Writing":["writer","content"]
}

# ---------------------------
# SKILL DATABASE
# ---------------------------

SKILL_DB=[
"python","java","sql","machine learning","deep learning","html","css",
"javascript","react","node","django","flask","aws","azure","docker",
"kubernetes","linux","excel","powerbi","tableau","data analysis",
"communication","sales","marketing","finance","accounting",
"project management","testing","automation","networking",
"cybersecurity","cloud","c++","angular","mongodb","mysql",
"statistics","nlp","big data"
]

# ---------------------------
# CLEAN HTML
# ---------------------------

def clean_html(text):
    text=re.sub("<.*?>"," ",text)
    return text.lower()

# ---------------------------
# TRAIN ML MODEL
# ---------------------------

def train_model():
    url="https://remotive.com/api/remote-jobs"
    response=requests.get(url).json()
    jobs=response["jobs"]

    X=[]
    y=[]

    for job in jobs[:200]:
        desc=job["description"].lower()
        skill_count=sum(skill in desc for skill in SKILL_DB)
        experience=np.random.randint(0,10)
        domain_match=np.random.randint(0,2)
        selected=1 if skill_count>3 else 0

        X.append([skill_count,experience,domain_match])
        y.append(selected)

    model=LogisticRegression()
    model.fit(X,y)

    return model

model=train_model()

# ---------------------------
# LOGIN PAGE
# ---------------------------

if st.session_state.page=="login":

    st.title("Interview Rejection Insights and Hiring Prediction")

    username=st.text_input("Username")
    password=st.text_input("Password",type="password")

    if st.button("Login"):
        if username in users and users[username]==password:
            st.success("Login successful")
            st.session_state.page="domain"
            st.rerun()
        else:
            st.error("Invalid username or password")

    if st.button("Go to Register"):
        st.session_state.page="register"
        st.rerun()

# ---------------------------
# REGISTER PAGE
# ---------------------------

elif st.session_state.page=="register":

    st.title("Register")

    username=st.text_input("Create Username")
    password=st.text_input("Create Password",type="password")

    if st.button("Create Account"):
        users[username]=password
        with open(USER_FILE,"w") as f:
            json.dump(users,f)
        st.success("Account created successfully")

    if st.button("Go to Login"):
        st.session_state.page="login"
        st.rerun()

# ---------------------------
# DOMAIN PAGE
# ---------------------------

elif st.session_state.page=="domain":

    st.title("Select Domain")

    domain=st.selectbox("Choose Domain",list(DOMAIN_KEYWORDS.keys()))

    if st.button("Load Job Roles"):

        url="https://remotive.com/api/remote-jobs"
        response=requests.get(url).json()
        jobs=response["jobs"]

        roles=[]
        desc=[]

        keywords=DOMAIN_KEYWORDS[domain]

        for job in jobs:
            title=job["title"].lower()

            if any(k in title for k in keywords):
                roles.append(job["title"])
                desc.append(job["description"])

            if len(roles)>=20:
                break

        st.session_state.roles=roles
        st.session_state.desc=desc

    if "roles" in st.session_state:

        role=st.selectbox("Job Role",st.session_state.roles)

        index=st.session_state.roles.index(role)
        description=st.session_state.desc[index]
        clean_desc=clean_html(description)

        st.subheader("Job Description")
        st.write(clean_desc[:800])

        st.session_state.job_description=clean_desc

        if st.button("Next"):
            st.session_state.page="skills"
            st.rerun()

# ---------------------------
# SKILLS PAGE
# ---------------------------

elif st.session_state.page=="skills":

    st.title("Skill Analysis")

    text=st.session_state.job_description

    extracted=[]

    for skill in SKILL_DB:
        if skill in text:
            extracted.append(skill)

    extracted=list(set(extracted))
    st.session_state.required_skills=extracted

    st.subheader("Required Skills")

    for s in extracted:
        st.write("•",s)

    candidate_skills=st.text_input("Enter your skills (comma separated)")
    experience=st.number_input("Experience (years)",0,20)

    if st.button("Check Skill Match"):

        req=[s.lower() for s in extracted]
        cand=[c.strip().lower() for c in candidate_skills.split(",")]

        match=len(set(req).intersection(cand))
        total=len(req)

        score=(match/total)*100 if total>0 else 0
        missing=list(set(req)-set(cand))

        st.session_state.match=match
        st.session_state.missing=missing
        st.session_state.score=score
        st.session_state.exp=experience

        st.write("Skill Match Score:",round(score,2),"%")

        st.subheader("Missing Skills")

        for m in missing:
            st.write("•",m)

    if "score" in st.session_state:
        if st.button("Prediction"):
            st.session_state.page="prediction"
            st.rerun()

# ---------------------------
# PREDICTION PAGE
# ---------------------------

elif st.session_state.page=="prediction":

    st.title("Hiring Prediction Result")

    skills_count=st.session_state.match
    experience=st.session_state.exp
    domain_match=1

    pred=model.predict([[skills_count,experience,domain_match]])
    prob=model.predict_proba([[skills_count,experience,domain_match]])

    probability=round(prob[0][1]*100,2)

    if pred[0]==1:
        st.success("High Chance of Selection")
    else:
        st.error("Low Chance of Selection")

    st.write("Selection Probability:",probability,"%")

    # ---------------------------
    # CHARTS
    # ---------------------------

    st.header("Hiring Insights Dashboard")

    years=[2019,2020,2021,2022,2023,2024,2025]
    selected=[random.randint(80,250) for i in years]
    rejected=[random.randint(200,450) for i in years]

    df=pd.DataFrame({"Year":years,"Selected":selected,"Rejected":rejected})
    st.plotly_chart(px.bar(df,x="Year",y=["Selected","Rejected"]))

    df2=pd.DataFrame({
        "Category":["Matched Skills","Missing Skills","Experience"],
        "Value":[st.session_state.match,len(st.session_state.missing),st.session_state.exp]
    })
    st.plotly_chart(px.pie(df2,values="Value",names="Category"))

    future_year=[2026,2027]
    future_selected=[random.randint(250,350) for i in future_year]
    future_rejected=[random.randint(350,500) for i in future_year]

    df3=pd.DataFrame({"Year":future_year,"Selected":future_selected,"Rejected":future_rejected})
    st.plotly_chart(px.bar(df3,x="Year",y=["Selected","Rejected"]))

    # ---------------------------
    # NEW FEATURE: SUGGESTIONS
    # ---------------------------

    st.header("Career Improvement Suggestions")

    match=st.session_state.match
    missing=st.session_state.missing
    score=st.session_state.score
    experience=st.session_state.exp

    st.subheader("Strengths")

    if match>=5:
        st.success("Strong skill match")
    else:
        st.warning("Skill match is low")

    if experience>=3:
        st.success("Good experience")
    else:
        st.warning("Low experience")

    st.subheader("Weak Areas")

    if missing:
        for m in missing:
            st.write("•",m)

    st.subheader("Suggestions")

    if score<50:
        st.write("👉 Improve basic skills")
        st.write("👉 Do projects")
    elif score<75:
        st.write("👉 Learn advanced topics")
    else:
        st.write("👉 Start applying for jobs")

    if experience<2:
        st.write("👉 Gain internship experience")

    st.subheader("Recommended Skills to Learn")

    for skill in missing[:5]:
        st.write("📘",skill)

    st.subheader("Final Advice")

    if score>70 and experience>2:
        st.success("You are ready for jobs")
    else:
        st.warning("Improve skills before job applying")

    if st.button("Back"):
        st.session_state.page="domain"
        st.rerun()