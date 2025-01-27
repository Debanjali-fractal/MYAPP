

    import pandas as pd
import google.generativeai as genai
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import os
import warnings
import chainlit as cl

warnings.filterwarnings('ignore')

# Set your Gemini API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyBeM3p8WXFVQ6x19aJX252tpWkHm11ckXg"  # Replace with your actual key
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Load the CSV data
try:
    df = pd.read_csv(r'C:\Users\debanjali.goswami\Downloads\DEBANJALI\Cancer-Data-Structured\data_Cancer_v2_Merged Data 2.csv')
    df.rename(columns={"YQ (YearQuarter)": "YQ"}, inplace=True)
except FileNotFoundError:
    print("Error: Dataset not found. Ensure the correct path.")
    exit()
except pd.errors.ParserError:
    print("Error: Could not parse the dataset.")
    exit()

# Initialize the Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)

# Conversation memory for Chainlit
memory = ConversationBufferMemory(memory_key="conversation_history", input_key="query")

# Prompt Template
prompt_template = PromptTemplate(
    input_variables=["query", "dataframe_head", "conversation_history"],
    template=(
        "You are a Python expert working with Pandas DataFrames.\n"
        "Here is the head of the DataFrame:\n{dataframe_head}\n\n"
        "Conversation so far:\n{conversation_history}\n\n"
         """
    You are also an expert in English, hence you can read and understand what the given questions asks for.
    Write Python pandas code to answer the question. The code should:
    - Use appropriate aggregations, groupings, and filters.
    - please read the questions carefully and give me the correct answers referring to the column names which have been provided to you.
    - the csv path is: 'Downloads/DEBANJALI/Cancer-Data-Structured/data_Cancer_v2_Merged Data 2.csv (use this path while reading data into dataframe)
    - Include proper column references based on the dataset provided.
    - Use concise and readable variable names when creating intermediate steps.
    - Provide well-structured pandas expressions.
    - When identifying the highest or lowest value, include both the category/label and its corresponding value in the output.
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
    - You need to be smart enough while dealing with the questions as to when to use 'diagnosisCodeDescription' column and when to use 'Cancer Diagnosis Category' column.
    - Please ensure that the resultant code wont produce any syntax error if executed later.
    - Please do not generate any backticks or and extra string in the first part and in the trailing part also.
    - If the user provides feedback (e.g., 'wrong answer'), please reanalyze the query and try to provide a corrected answer.
      Always write clean and correct Pandas code. 
    - Please adhere to the instructions mentioned above and give me the correct answer at one go.
    
    You are also an expert in data analysis. When generating Python code to answer queries:
    - Always carefully interpret the query and identify relevant columns.
    - Use appropriate filters for categorical columns. For example:
        - To filter for non-malignant tumours, ensure the 'Cancer Diagnosis Category' does NOT contain 'Malignant'.
        - Handle missing values in filtering logic by adding 'na=False' where required.
        - Apply case-insensitive matching for textual data using 'case=False' if necessary.
     - When grouping data:
        - Use the exact column name provided in the dataset (e.g., 'Facility Code').
        - Ensure the grouping column and aggregation column are correctly matched.
     - When sorting results:
        - Sort by count or numerical values in descending order unless otherwise specified.
        - Always test your filtering and logic to ensure accurate results.
        - Ensure all column names match the dataset exactly.
        - Write code that avoids syntax or runtime errors, even when handling edge cases.
        - Handle date and time columns properly if referenced in the query.
        - Generate results only relevant to the query, avoiding redundant data.
    - Do not use abrupt column names please.
    """
        "Task:Hence, write the full Pandas code to answer the following query:\n{query}\n\n"
        "Do not provide explanations, write the full code."
    ),
)

# Create the LLM Chain
llm_chain = LLMChain(llm=llm, prompt=prompt_template, memory=memory)

# Create the Pandas agent
pandas_agent = create_pandas_dataframe_agent(
       llm, df, verbose=False, allow_dangerous_code=True, handle_parsing_errors=True,
       prefix=(
        "You are an expert in Python and Pandas DataFrame operations.\n"
        "Your task is to execute the Python code accurately on the given DataFrame.\n"
        "Ensure correctness and return only the result of the execution. Do not provide explanations or extra output."
    )
)

last_query = None
# Chainlit App
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
async def handle_message(message):
    global last_query
    user_input = message.content.strip().lower()
    if "wrong answer" in user_input or "incorrect" in user_input or "rectify" in user_input:
        if last_query:
            await cl.Message(content="Apologies! Let me recheck and correct the response. Please wait a moment.").send()
            try:
                # Reanalyze the last query
                response = llm_chain.run({
                    "query": last_query,
                    "dataframe_head": str(df.head())
                })
                pandas_code = response.strip('`').replace("```python", "").replace("```", "").strip()

                # Display the corrected code
                await cl.Message(content=f"Corrected Pandas Code:\n```python\n{pandas_code}\n```").send()

                # Execute the corrected code
                result = pandas_agent.run(pandas_code)
                await cl.Message(content=f"Corrected Result:\n{result}").send()
            except Exception as e:
                await cl.Message(content=f"Error during correction: {e}").send()
        else:
            await cl.Message(content="I don't have a previous query to reanalyze. Please provide a new query.").send()
        return

    # Regular query processing
    try:
        last_query = message.content  # Store the query
        response = llm_chain.run({
            "query": last_query,
            "dataframe_head": str(df.head())
        })
        pandas_code = response.strip('`').replace("```python", "").replace("```", "").strip()

        # Display the generated code
        await cl.Message(content=f"Generated Pandas Code:\n```python\n{pandas_code}\n```").send()

        # Execute the code using the Pandas Agent
        result = pandas_agent.run(pandas_code)

        # Display the execution result
        await cl.Message(content=f"Execution Result:\n{result}").send()
    except Exception as e:
        await cl.Message(content=f"Error: {e}").send()




# # Conversation history cache
# conversation_history = []

# @cl.on_message
# async def main(message: cl.Message):
#     global conversation_history
#     user_query = message.content.strip()

#     # Handle the "start" command or first-time message
#     if len(conversation_history) == 0:
#         welcome_message = (
#             "Welcome to the Cancer Data Analysis Assistant! 🤖\n\n"
#             "I can help you analyze and process the Cancer Data using Python and Pandas. "
#             "Feel free to ask questions about the dataset, and I'll generate the code and analyze the data for you.\n\n"
#             "You can also type:\n"
#             "- 'Show conversation history' to see the entire conversation so far.\n"
#             "- 'Delete conversation history' to clear all previous history.\n\n"
#             "Let’s get started! 🚀"
#         )
#         await cl.Message(content=welcome_message).send()

#     # Handle specific commands
#     if user_query.lower() == "show conversation history":
#         if not conversation_history:
#             await cl.Message(content="No conversation history found.").send()
#         else:
#             history = "\n".join(conversation_history)
#             await cl.Message(content=f"Conversation History:\n{history}").send()
#         return

#     elif user_query.lower() == "delete conversation history":
#         conversation_history = []
#         await cl.Message(content="Conversation history has been cleared.").send()
#         return

#     # Add query to conversation history
#     conversation_history.append(f"User: {user_query}")

#     # Process the query (replace this with your logic)
#     response = f"Processing your query: {user_query}"  # Placeholder
#     conversation_history.append(f"Assistant: {response}")
#     await cl.Message(content=response).send()


# @cl.on_message
# async def main(message: cl.Message):
#     global conversation_history
#     user_query = message.content.strip()

#     # Handle conversation history commands
#     if user_query.lower() == "show conversation history":
#         if not conversation_history:
#             await cl.Message(content="No conversation history found.").send()
#         else:
#             history = "\n".join(conversation_history)
#             await cl.Message(content=f"Conversation History:\n{history}").send()
#         return

#     elif user_query.lower() == "delete conversation history":
#         conversation_history = []
#         await cl.Message(content="Conversation history has been cleared.").send()
#         return

#     # Add user query to conversation history
#     conversation_history.append(f"User: {user_query}")

#     # Process the query using the LLM and agent
#     try:
#         # Generate the code using the LLM
#         response_code = llm_chain.run({
#             "query": user_query,
#             "dataframe_head": str(df.head()),
#             "conversation_history": "\n".join(conversation_history)
#         }).strip()

#         # Add the generated code to the conversation history
#         conversation_history.append(f"Generated Code:\n{response_code}")

#         # Execute the code and return the result
#         exec_globals = {"pd": pd, "df": df, "result": None}  # Execution environment
#         exec(response_code, exec_globals)
#         result = exec_globals.get("result", "No 'result' variable defined in the code.")

#         # Use the Pandas agent for an additional explanation
#         agent_result = pandas_agent.run(response_code)

#         # Log results to conversation history
#         conversation_history.append(f"Result:\n{result}")
#         conversation_history.append(f"Agent Explanation:\n{agent_result}")

#         # Send results back to the user
#         await cl.Message(content=f"**Generated Code:**\n```python\n{response_code}\n```").send()
#         await cl.Message(content=f"**Result:**\n{result}").send()
#         await cl.Message(content=f"**Agent Explanation:**\n{agent_result}").send()

#     except Exception as e:
#         error_message = f"An error occurred: {str(e)}"
#         conversation_history.append(f"Error: {error_message}")
#         await cl.Message(content=error_message).send()


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
#         generated_code = response.strip('`').replace("```python", "").replace("```", "").strip()
#         conversation_history.append({"role": "assistant", "content": generated_code})
#         save_conversation_history(conversation_history)

#         await cl.Message(content=f"Generated Pandas Code:\n```{generated_code}\n```").send()

#         # Execute the generated code
#         exec_globals = {"pd": pd, "df": df, "result": None}
#         exec(generated_code, exec_globals)
#         result = exec_globals.get("result", "No 'result' variable defined in the code.")
#         agent_result = pandas_agent.run(generated_code)

#         # Save and send results
#         conversation_history.append({"role": "assistant", "content": f"Result:\n{result}"})
#         save_conversation_history(conversation_history)

#         await cl.Message(content=f"Execution Result:\n{result}").send()
#         await cl.Message(content=f"Agent Analysis:\n{agent_result}").send()

#     except Exception as e:
#         error_message = f"Error executing query: {e}"
#         conversation_history.append({"role": "assistant", "content": error_message})
#         save_conversation_history(conversation_history)
#         await cl.Message(content=error_message).send()