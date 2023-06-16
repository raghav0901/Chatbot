from distutils.command.install_egg_info import to_filename
from email.errors import MalformedHeaderDefect
from multiprocessing import active_children
from re import split
from tarfile import RECORDSIZE
from threading import _DummyThread
from tkinter.messagebox import QUESTION
from flask import Flask, render_template, request, flash, send_file
from werkzeug.exceptions import InternalServerError
import os
import pandasql as psql
import pandas as pd
from sqlalchemy import create_engine, text
# models text-davinci-003, text-davinci-002, text-curie-001, text-babbage-001, text-ada-001
# Question list all features for employees lives in Brampton and thier gender is Female and status are Active and hierd between 2017 and 2019
# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"
import openai
from flask_ngrok import run_with_ngrok

# Load the CSV file into a Pandas DataFrame
df = pd.read_excel("employee1.xlsx")
pf=pd.read_excel("beneficiary1.xlsx")
##df['status'] = df['comments']

# Mapping of province abbreviations to full names
ShortFullProvince = {
    "AB": "Alberta",
    "BC": "British Columbia",
    "ON": "Ontario",
    "SK": "Saskatchewan",
    "MB": "Manitoba",
    "QC": "Quebec",
    "NB": "New Brunswick",
    "NL": "Newfoundland and Labrador",
    "YT": "Yukon",
    "PE": "Prince Edward Island",
    "NS": "Nova Scotia"
}

Gender = {
    "Male":"M",
    "Female":"F"
}

statusCode = {
     "TRM":"Terminated",
    "ACT": "Active",
  "TWB": "Staged Servance",
      "MOD" : "Modified LTD",
      "LTD" : "Long Term Disablity",
      "LOP" : "LOA with benifits-payroll",
      "LON" : "LOA with benifits non-personal",
      "LOA" : "comments LOA-opt out",
      "INV" : "Invalid banking",
      "INAC" : "Inactive Associate",
      "INA" : "Inactive",
      "EXC" : "Exception",
      "DTH" : "Deceased",
      "DEL" : "Delinquent",
      "DEC" : "Delinquent",
      "BAN" : "Waiting for banking",
      "APY" : "Active on payroll"
       
}

#df['gender']=df['gender'].map(Gender)




# Add a Province column with full names
#df['province'] = df['province'].map(ShortFullProvince)
#df['employmentProvince']=df['employmentProvince'].map(ShortFullProvince)
df['comment']=df['comment'].map(statusCode)
# Create an in-memory SQLite database engine
db = create_engine('sqlite:///:memory:')

# Define a function to execute SQL statements on the database
def query_db(sql_statement):
    df.to_sql(name="df", con=db)
    pf.to_sql(name="pf",con=db)
    with db.connect() as conn:
        # Use a SQLAlchemy text object to execute the SQL statement
        response = conn.execute(text(sql_statement))
        # Return the result set as a list of tuples
        return response.fetchall()

# Set up the OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define a function to generate SQL statements using OpenAI API
def sql_(df,pf, sql_statement):
    df_name = 'df'
    df_columns = ",".join(df.columns)
    df_name_columns = df_name + '(' + df_columns + ')'
    print(df_name_columns)

    pf_name='pf'
    pf_columns=",".join(pf.columns)
    pf_name_columns= pf_name + '(' + pf_columns + ')'

    modelx=str(request.form.get('model_input'))
    print(modelx)
    print(len(modelx))
    response = openai.Completion.create(
        model=modelx,
        prompt=f"""### SQLite SQL tables, with their properties:\n#\n# {df_name_columns}\n# {pf_name_columns}\n#\n### 
        A query to answer: {sql_statement} \nSELECT""",
        temperature=0,
        max_tokens=150,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["#", ";"]
    )
    return 'SELECT' + response['choices'][0]['text']

app = Flask(__name__)
app.secret_key = "my_secret_key"
run_with_ngrok(app)

@app.route("/")
def index():
    flash(list(df.columns))
    return render_template("index.html")

@app.route("/answer", methods=["POST", "GET"])
def answer():
    Question_ = str(request.form['name_input'])
    Question=Question_ + str(request.form.get('ignore_input'))

    model1 = str(request.form.get('model1'))
    if model1 is None:
            model1 = "text-davinci-003"  # Set a default model name

    # print(model)
    #my_res=transform_sentence(Question)
    #print(my_res)
    response = sql_(df,pf,Question)
    result = query_db(response)

    flash(result)
    result_df = pd.DataFrame(result)
    r=len(result_df)
    dd=result_df.describe()
    result_df.to_csv("query_results.csv", index=False)
    headers = list(result_df.columns)
    return render_template("index.html", results=result, question=Question,Response=response, r=r,dd=dd,headers=headers)

### Objective of the function:
### take the sentence from the user and transoform it to the sentence having appropriate columns to search
###if split_it.index(T)>1 and split_it[(split_it.index(T))-1] in ['as','of','equal to ','equal','equal to']:
###        split_it[split_it.index(T)-2]=x
###    else: 


def transform_sentence(sentence):
 store=[]
 store_it=[]
 string=sentence
 split_it=string.split()
 store_it=[]
 ##
 check_ser=pd.Series({c: df[c].unique() for c in df})
 check_per=pd.Series({c: pf[c].unique() for c in pf})
 ##print(check_ser['comment'])
 for T in split_it:
  for x in list(df.columns):
   if T in check_ser[x] or T in check_per[x]:
     print('I RANN')
     print(T)   
     if split_it.index(T)>1 and split_it[(split_it.index(T))-1] in ['as','of','to','equal','being','be','is','=','like']:
        split_it[split_it.index(T)-2]=x
     else: 
        if T not in store:
         store.append(T)
        store_it.append((T,x))
   elif str(T) in str(x) and ((0 in check_ser[x] and 1 in check_ser[x]) or (0 in check_per[x] and 1 in check_per[x])):
     print('I RANER')
     if split_it.index(T)>1 and split_it[(split_it.index(T))-1] in ['not','non','no','doesnt','dont']:
       split_it[(split_it.index(T))-1]=str(0)  
      

 
 
 if len(store)>0:
  store_it=transform_list(store_it,store)
 print(store_it)
 if len(store)>0:
  for x in store_it:
      split_it.insert(split_it.index(x[0])+1,x[1])
 
 if len(split_it)>0:    
  return " ".join(split_it)
 else:
  return 0  

def transform_list(lis,store):
 returner=[]
 for x in store:
    curr=None
    for y in lis:
       if y[0]==x:
          if curr==None:
            curr=(y[1],(df[str(y[1])] == y[0]).sum())
          elif curr[1]<(df[str(y[1])] == y[0]).sum():
            curr=(y[1],(df[str(y[1])] == y[0]).sum()) 
    returner.append((x,curr[0]))
 return returner             




 
@app.route("/download")
def download():
    file_path = "query_results.csv"
    return send_file(file_path, as_attachment=True)


if __name__ == "__main__":
    #app.debug = True
    app.run()
