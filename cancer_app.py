import pandas as pd
import google.generativeai as genai
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import os
import re
import warnings
import chainlit as cl
import json
import io
import contextlib

warnings.filterwarnings('ignore')

# Set your Gemini API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyBeM3p8WXFVQ6x19aJX252tpWkHm11ckXg"  # Replace with your actual key
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Local cache for conversation history
CONVERSATION_CACHE = "conversation_history.json"

# Function to load conversation history
def load_conversation_history():
    if os.path.exists(CONVERSATION_CACHE):
        with open(CONVERSATION_CACHE, "r") as f:
            return json.load(f)
    return []

# Function to save conversation history
def save_conversation_history(history):
    with open(CONVERSATION_CACHE, "w") as f:
        json.dump(history, f)

# Load the initial conversation history
conversation_history = load_conversation_history()
# Load the CSV data
try:
    df = pd.read_csv(r'C:\Users\debanjali.goswami\Downloads\DEBANJALI\Cancer-Data-Structured\data_Cancer_v2_Merged Data 2.csv')
    df.rename(columns={"YQ (YearQuarter)": "YQ"}, inplace=True)
except FileNotFoundError:
    raise FileNotFoundError(
        "Error: The specified CSV file was not found. Ensure the path is correct."
    )
except pd.errors.ParserError:
    raise ValueError("Error: Could not parse the CSV file. Please check the file format.")

# Initialize the Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
memory = ConversationBufferMemory(memory_key="conversation_history", input_key="query")

prompt_template = PromptTemplate(
    input_variables=["query", "dataframe_head", "conversation_history"],
    template=(
            "You are a Python expert working with Pandas DataFrames.\n"
        "Here is the head of the DataFrame:\n{dataframe_head}\n\n"
        "Conversation so far:\n{conversation_history}\n\n"
         '''
    You are also an expert in English, hence you can read and understand what the given questions asks for.
    Write Python pandas code to answer the question. The code should:
    - Use appropriate aggregations, groupings, and filters.
    - please read the questions carefully and give me the correct answers referring to the column names which have been provided to you.
    - the csv path is:  (r'C:\\Users\debanjali.goswami\Downloads\DEBANJALI\Cancer-Data-Structured\data_Cancer_v2_Merged Data 2.csv')(use this path while reading data into dataframe)
    - Include proper column references based on the dataset provided.
    - Use concise and readable variable names when creating intermediate steps.
    - Provide well-structured pandas expressions.
    - If performing calculations (e.g., averages, counts), name the result columns accordingly.
    - Ensure the code references the following columns where relevant:
      - patientID
      - encounterID
      - diagnosisCodeDescription
      - Cancer Diagnosis Category
      - Specialty
      - Encounter Type
      - encounterAdmitDateTime
      - encounterDischargeDateTime
      - Facility Code(whenever you are asked about cancer provider or such, refer to this column always.)
      - ageAtEncounter
      - ageGroup
      - LOS(hours)
      - readmission
      - YearQuarter
      - gender
      - nationality
      - national
    - Handle date or time columns properly if referenced in the query.
    - Morover incase of date related problems please do not use unique unitl and unless the questions asks for.
    - Please try to generate "all" the discharge dates and admit dates properly when asked in question for the various groups as per question. Please try to follow this point as said.
    - Ensure correctness and completeness of the generated code, specially for date related questions.
    - Ensure to use the exact column names as they appear in the DataFrame, , (e.g., 'YQ' and the column 'YearQuarter' in the dataset are samecolumns)
    - Ensure the code is executable without any explanation text.
    - Please try to differentiate malignant and non-malignant tumor case questions by referring to the dataset information properly. Specially go through these columns carefully:
     - diagnosisCodeDescription
     - Cancer Diagnosis Category and then generate the correct code for it. e.g., if asked for malignant tumor, you should check whether 'Malignant' exists in that column is True then go ahead
     with further functions. similarly for 'Non-malignant' tumor, check if 'Malignant' word is present or not in the data.
    - Please ensure that the resultant code wont produce any syntax error if executed later.
    - Please do not generate any backticks or and extra string in the first part and in the trailing part also.
    - Please adhere to the instructions mentioned above and give me the correct answer at one go.
    '''
    "Please write the complete code starting with `import pandas as pd` and ending with a line that outputs the final result."
        "Task:Hence, write the full Pandas code to answer the following query:\n{query}\n\n"
        "Do not provide explanations, write the full code."

    ),
)

# Create the LLM Chain
llm_chain = LLMChain(llm=llm, prompt=prompt_template, memory=memory)

# Create the Pandas agent
pandas_agent = create_pandas_dataframe_agent(
    llm, df, verbose=False, allow_dangerous_code=True, handle_parsing_errors=True,
    prefix=
    '''YOU ARE A WORLD-CLASS PANDAS EXPERT AGENT, DESIGNED TO EXCEL IN DATA MANIPULATION, ANALYSIS, AND PRESENTATION USING THE PANDAS LIBRARY. YOUR OBJECTIVE IS TO READ AND EXECUTE PYTHON CODE, DISPLAY THE RESULTING DATA IN A WELL-STRUCTURED FORMAT, AND PROVIDE A DETAILED EXPLANATION OR ANSWER IN NATURAL LANGUAGE BASED ON THE DATA OUTPUT.

###INSTRUCTIONS###

1. **CODE EXECUTION AND DATA DISPLAY**:
   - EXECUTE the provided Python code snippet using the Pandas library.
   - RETURN a properly formatted output, displaying DataFrames or series in a structured table-like format.
   - ENSURE that the output is readable, aligned, and consistently formatted for easy understanding. Please print the output of the code.

2. **QUESTION ANSWERING**:
   - ANALYZE the resulting data output.
   - INTERPRET the question provided by the user in relation to the data.
   - GENERATE a NATURAL LANGUAGE RESPONSE that directly answers the question, supported by observations or computations from the data.

3. **DATA INSIGHTS**:
   - If no specific question is asked, SUMMARIZE key insights from the data, such as aggregate statistics, trends, or notable features.
   - SUGGEST additional analyses or visualizations where relevant.

4. **ERROR HANDLING**:
   - IF an error is encountered in the provided code, RETURN a CLEARLY EXPLAINED ERROR MESSAGE indicating the issue and, if possible, SUGGEST a FIX.
   - ENSURE that the explanation is easy to understand, even for users with minimal Python knowledge.
5.You need to display the output of the code after executing it line by line, then explain the question, i.e., analyse the answer you are displaying and print the corresponding analysis explanation.
'''
)
conversation_history = []
#Define the chainlit app
@cl.on_chat_start
async def start():
    await cl.Message(
        content=(
            "Welcome to the Cancer Data Analysis Chatbot! 🎉\n\n"
            "This application allows you to explore and analyze a dataset about cancer cases. "
            "You can ask questions like:\n"
            "- What is the distribution of cancer cases by type?\n"
            "- Which category has the highest number of cases?\n"
            "- Show discharge dates for various groups.\n\n"
            "Feel free to ask your questions!"
        )
    ).send()

@cl.on_message
async def main(message: cl.Message):
    global conversation_history

    user_query = message.content.strip()
    conversation_history.append({"role": "user", "content": user_query})

    # Check for special commands
    if user_query.lower() == "show conversation history":
        # if conversation_history:
        #     history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation_history])
        #     await cl.Message(content=f"Conversation History:\n\n{history}").send()
        # else:
        #     await cl.Message(content="Conversation history is empty.").send()
        # return
        if conversation_history:
            formatted_history = " **Conversation History:**\n\n"
            for i, msg in enumerate(conversation_history, 1):
                formatted_history += f"**Conversation {i}:**\n"
                formatted_history += f" **You asked:** {message.input}\n"
                formatted_history += f" **Response:** {message.output}\n"
                formatted_history += "---\n"
            await cl.Message(content=formatted_history).send()
        else:
            await cl.Message(content="No conversation history yet. Try asking some questions first!").send()
        return

    if user_query.lower() == "delete conversation history":
        conversation_history = []
        save_conversation_history(conversation_history)
        await cl.Message(content="Conversation history has been cleared.").send()
        return

    try:
        # Generate code from LLM
        response = llm_chain.run({
            "query": user_query,
            "dataframe_head": str(df.head()),
            "conversation_history": "\n".join([msg["content"] for msg in conversation_history])
        })

        # Clean up generated code
        generated_code = re.sub(r"```(?:python)?", "", response).strip()
        conversation_history.append({"role": "assistant", "content": generated_code})
        save_conversation_history(conversation_history)

        await cl.Message(content=f"Generated Pandas Code:\n```{generated_code}\n```").send()

        # Redirect output capture
        exec_globals = {"pd": pd, "df": df, "result": None}
        output_capture = io.StringIO()
        with contextlib.redirect_stdout(output_capture):
            exec(generated_code, exec_globals)

        # Retrieve the result and output
        exec_output = output_capture.getvalue()
        result = exec_globals.get("result")

        if result is None and not exec_output.strip():
            result_message = "Execution completed, but no result or printed output was produced."
        elif result is not None:
            result_message = f"Execution Result:\n{result}"
        else:
            result_message = f"Printed Output:\n{exec_output.strip()}"

        # Run additional agent analysis
        # agent_task = f"Analyze the dataframe based on: {user_query}"
        agent_result = pandas_agent.run(generated_code)

        # Save and send results
        conversation_history.append({"role": "assistant", "content": result_message})
        save_conversation_history(conversation_history)

        await cl.Message(content=result_message).send()
        await cl.Message(content=f"Agent Analysis:\n{agent_result}").send()

    except Exception as e:
        error_message = f"Error executing query: {e}"
        conversation_history.append({"role": "assistant", "content": error_message})
        save_conversation_history(conversation_history)
        await cl.Message(content=error_message).send()
 # Define the Chainlit app
# @cl.on_message
# async def main(message: cl.Message):
#     global conversation_history

#     user_query = message.content.strip()
#     conversation_history.append({"role": "user", "content": user_query})

#     # Check for special commands
#     if user_query.lower() == "show conversation history":
#         if conversation_history:
#             history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation_history])
#             await cl.Message(content=f"Conversation History:\n\n{history}").send()
#         else:
#             await cl.Message(content="Conversation history is empty.").send()
#         return

#     if user_query.lower() == "delete conversation history":
#         conversation_history = []
#         save_conversation_history(conversation_history)
#         await cl.Message(content="Conversation history has been cleared.").send()
#         return

#     try:
#         # Generate code from LLM
#         response = llm_chain.run({
#             "query": user_query,
#             "dataframe_head": str(df.head()),
#             "conversation_history": "\n".join([msg["content"] for msg in conversation_history])
#         })

#         # Clean up generated code
#         generated_code = re.sub(r"```(?:python)?", "", response).strip()
#         conversation_history.append({"role": "assistant", "content": generated_code})
#         save_conversation_history(conversation_history)

#         await cl.Message(content=f"Generated Pandas Code:\n```{generated_code}\n```").send()

#         # Execute the generated code
#         exec_globals = {"pd": pd, "df": df, "result": None}
#         try:
#             exec(generated_code, exec_globals)
#         except Exception as exec_error:
#             raise RuntimeError(f"Error in executing the generated code: {exec_error}")

#         result = exec_globals.get("result", "No 'result' variable defined in the code.")
#         #agent_task = f"Analyze the dataframe based on: {result}"
#         agent_result = pandas_agent.run(generated_code)

#         # Save and send results
#         conversation_history.append({"role": "assistant", "content": f"Result:\n{result}"})
#         save_conversation_history(conversation_history)

#         #await cl.Message(content=f"Agent Analysis:\n{agent_result}").send()
#         await cl.Message(content=f"Execution Result:\n{result}").send()
#         await cl.Message(content=f"\n{agent_task}").send()
#         await cl.Message(content=f"Agent Analysis:\n{agent_result}").send()
        

#     except Exception as e:
#         error_message = f"Error executing query: {e}"
#         conversation_history.append({"role": "assistant", "content": error_message})
#         save_conversation_history(conversation_history)
#         await cl.Message(content=error_message).send()

