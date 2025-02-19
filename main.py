from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.schema import HumanMessage, AIMessage, SystemMessage
import json

load_dotenv()

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for security in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LanguageModelProcessor:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature=0, google_api_key=os.getenv("GEMINI_API_KEY"))

        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        with open('system_prompt.txt', 'r') as file:  # Make sure system_prompt.txt exists
            system_prompt = file.read().strip()

        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{text}")
        ])

        self.conversation = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory
        )

        self.ai_tutor_prompt_template = ChatPromptTemplate.from_template( # AI Tutor Prompt Template
            """

            ### Inputs:
            - **Chat history:** Tracks previous interactions in the format chat_history to ensure continuity.
            - **Voice command:** The spoken request from the student.
            - **Question:** The specific problem the student is working on.
            - **Whiteboard content:** The student’s written progress, including equations or partial work.

            Chat history:
            {chat_history}

            Voice command:
            {voice_command}

            Question:
            {question}

            Whiteboard content:
            {whiteboard_content}

            ### Instructions:

            Based on the voice command, question, and whiteboard content, provide **step-by-step hints** rather than a full solution.  

            If the student has **made progress**, analyze their work and:
            - Identify **what they did correctly**.
            - **Provide the next step as a hint** while explaining the thought process.
            - If there is a **mistake**, point it out clearly and guide them towards fixing it.

            If **no progress** has been made:
            - Give a **high-level method overview** or suggest a **general approach**.

            For **voice instructions**, provide:
            - A **guiding thought process** leading to the hint (e.g., “How about we try this?”).  
            - **Encouragement**, keeping it concise and engaging in a single natural-sounding sentence.

            **Side Questions:**
            - If the student asks a side question that is related to the main question, provide a short answer.  Prioritize keeping the focus on the primary problem, but address relevant side questions briefly to maintain momentum and understanding.


            ### **Output Format:**
            ```json

                "instructions_to_solve_question": "Provide a paragraph explaining the thought process, next hint, and any mistakes if present. Use Markdown or LaTeX for equations if necessary.",
                "ai_tutor_voice_instructions": "Provide a natural-sounding instruction that combines a guiding thought process with encouragement."


            ---

                ### **Example 1: First Interaction (High-Level Direction)**
                **Chat History:** ``  

                **Voice Command:** *"How should I solve this?"*  

                **Question:** *"Find the derivative of* \\( f(x) = x^2 \\sin(x) \\) *."*  

                **Whiteboard Content:** *(empty)*  

                **Output:**  
                ```json
                
                    "instructions_to_solve_question": "This function involves a product of two functions: **x²** and **sin(x)**. To differentiate, we use the **product rule**, which states: $$ (uv)' = u'v + uv' $$. Identify **u = x²** and **v = sin(x)**, then differentiate each separately before applying the rule.",
                    "ai_tutor_voice_instructions": "Since this is a product of two functions, we can apply the product rule: (uv)' = u'v + uv'. Let's try differentiating each part first!"
                
                ```
            ---

                ### **Example 2: Student Asks a Side Question**
                **Chat History:** 
                ```json
                    
                        "voice_command": "How should I solve this?",
                        "question": "Find the derivative of f(x) = x² sin(x).",
                        "whiteboard_content": "",
                        "response": 
                            "instructions_to_solve_question": "This function involves a product of two functions: **x²** and **sin(x)**. To differentiate, we use the **product rule**, which states: $$ (uv)' = u'v + uv' $$. Identify **u = x²** and **v = sin(x)**, then differentiate each separately before applying the rule.",
                            "ai_tutor_voice_instructions": "Since this is a product of two functions, we can apply the product rule: (uv)' = u'v + uv'. Let's try differentiating each part first!"
                
                ``` 

                **Voice Command:** *"What exactly is a derivative, in simple terms?"*  

                **Question:** *"Find the derivative of* \\( f(x) = x^2 \\sin(x) \\) *."* 

                **Whiteboard Content:**
                    ```

                    ```
                **Output:**  
                    ```json
                    
                        "instructions_to_solve_question": "A derivative represents the instantaneous rate of change of a function. Think of it as the slope of the function at any given point.  Now, back to our problem, remember the product rule...",
                        "ai_tutor_voice_instructions": "A derivative is simply the instantaneous rate of change. Think of it as the slope of the function at any given point. Now, let's get back to our problem using the product rule. You've got this!"

                    ```
            ---

                ### **Example 3: Student Makes a Mistake**
                **Chat History:** 
                ```json
                    
                    "voice_command": "How should I solve this?",
                    "question": "Find the derivative of f(x) = x² sin(x).",
                    "whiteboard_content": "",
                    "response": 
                        "instructions_to_solve_question": "This function involves a product of two functions: **x²** and **sin(x)**. To differentiate, we use the **product rule**, which states: $$ (uv)' = u'v + uv' $$. Identify **u = x²** and **v = sin(x)**, then differentiate each separately before applying the rule.",
                        "ai_tutor_voice_instructions": "Since this is a product of two functions, we can apply the product rule: (uv)' = u'v + uv'. Let's try differentiating each part first!"
                    
                ,
                    "voice_command": "What exactly *is* a derivative, in simple terms?",
                    "question": "Find the derivative of f(x) = x² sin(x).",
                    "whiteboard_content": "",
                    "response":
                        "instructions_to_solve_question": "A derivative represents the instantaneous rate of change of a function. Think of it as the slope of the function at any given point.  Now, back to our problem, remember the product rule...",
                        "ai_tutor_voice_instructions": "A derivative is simply the instantaneous rate of change. Think of it as the slope of the function at any given point. Now, let's get back to our problem using the product rule. You've got this!"
                    
                
                ``` 

                **Voice Command:** *"I used the product rule. Is this correct?"*  

                **Question:** *"Find the derivative of* \\( f(x) = x^2 \\sin(x) \\) *."* 

                **Whiteboard Content:**
                    ```
                    \\[f'(x) = (2x \\cos(x)) + (x^2 \\cos(x))\\]  
                    ```

                    **Output:**  
                    ```json
                    
                        "instructions_to_solve_question": "You're on the right track using the product rule! However, check the first term carefully. The derivative of **x²** is correctly **2x**, but the product rule states that we keep **sin(x)** as is in this term. Similarly, when differentiating **sin(x)**, it should be **cos(x)**, so the second term is correct.",
                        "ai_tutor_voice_instructions": "You're almost there! Just double-check how you applied the product rule—should the first term include the derivative of **sin(x)**? You're doing great!"
                    
                    ```
            ---
                    
                ### **Example 4: Student Gets Correct Answer But Needs Simplification**  
                **Chat History:** 
                ```json
                
                    "voice_command": "How should I solve this?",
                    "question": "Find the derivative of f(x) = x² sin(x).",
                    "whiteboard_content": "",
                    "response": 
                        "instructions_to_solve_question": "This function involves a product of two functions: **x²** and **sin(x)**. To differentiate, we use the **product rule**, which states: $$ (uv)' = u'v + uv' $$. Identify **u = x²** and **v = sin(x)**, then differentiate each separately before applying the rule.",
                        "ai_tutor_voice_instructions": "Since this is a product of two functions, we can apply the product rule: (uv)' = u'v + uv'. Let's try differentiating each part first!"
                ,

                    "voice_command": "What exactly *is* a derivative, in simple terms?",
                    "question": "Find the derivative of f(x) = x² sin(x).",
                    "whiteboard_content": "",
                    "response":
                        "instructions_to_solve_question": "A derivative represents the instantaneous rate of change of a function. Think of it as the slope of the function at any given point.  Now, back to our problem, remember the product rule...",
                        "ai_tutor_voice_instructions": "A derivative is simply the instantaneous rate of change. Think of it as the slope of the function at any given point. Now, let's get back to our problem using the product rule. You've got this!"
                    
                ,
                
                    "voice_command": "I used the product rule. Is this correct?",
                    "question": "Find the derivative of f(x) = x² sin(x).",
                    "whiteboard_content": "f'(x) = (2x cos(x)) + (x² cos(x))",
                    "response": 
                        "instructions_to_solve_question": "You're on the right track using the product rule! However, check the first term carefully. The derivative of **x²** is correctly **2x**, but the product rule states that we keep **sin(x)** as is in this term. Similarly, when differentiating **sin(x)**, it should be **cos(x)**, so the second term is correct.",
                        "ai_tutor_voice_instructions": "You're almost there! Just double-check how you applied the product rule—should the first term include the derivative of **sin(x)**? You're doing great!"
                    
                
                ```  

                **Voice Command:** *"Is there a way to simplify this?"* 

                **Question:** *"Find the derivative of* \\( f(x) = x^2 \\sin(x) \\) *."*  

                **Whiteboard Content:**  
                    ```
                    \\[f'(x) = 2x \\sin(x) + x^2 \\cos(x)\\]  
                    ```

                **Output:**  
                ```json
                
                "instructions_to_solve_question": "Great work! Now, if you need to simplify, notice that both terms contain **x**, so you can factor it out: $$ f'(x) = x (2 \\sin(x) + x \\cos(x)) $$ This might make further calculations easier in related problems.",
                "ai_tutor_voice_instructions": "Nice job getting the derivative! Now, what if we factor out the common term **x**—would that simplify things? You're doing great!"
                
                ```
            ```
            """
            )


    def process(self, voice_command, question, whiteboard_content, chat_history):
        chat_history_str = self.format_chat_history_for_prompt()
        prompt = self.ai_tutor_prompt_template.format(
            chat_history=chat_history_str,
            voice_command=voice_command,
            question=question,
            whiteboard_content=whiteboard_content
        )

        messages = [
            SystemMessage(content="You are an AI tutor specialized in guiding students through questions."),
            HumanMessage(content=prompt)
        ]

        response = self.llm.invoke(messages).content.strip()

        # Remove triple backticks if present:
        if response.startswith("```json") and response.endswith("```"):
            response = response[7:-3].strip()  # Remove ```json and ```
        elif response.startswith("```"):
            response = response[3:-3].strip() # Remove ```

        try:
            json_response = json.loads(response)  # Now try to parse
            instructions = json_response.get("instructions_to_solve_question", "")
            voice_instructions = json_response.get("ai_tutor_voice_instructions", "")
            
            self.memory.chat_memory.add_user_message(f"Voice command: {voice_command}\nQuestion: {question}\nWhiteboard: {whiteboard_content}")
            self.memory.chat_memory.add_ai_message(response) # Store raw response
            
            updated_chat_history = self.format_chat_history_for_prompt()  # Get updated chat history
            
            print(f"Instructions: {instructions}")
            print("\n \n")
            print(f"Voice Instructions: {voice_instructions}")
            return instructions, voice_instructions, updated_chat_history  # Return all parts

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response: {response}") # Print the raw response for debugging
            print(f"JSONDecodeError: {e}") # Print the specific JSON error
            return "Error: Could not parse JSON response.", "Error: Could not parse JSON response.", chat_history

    def format_chat_history_for_prompt(self):
        formatted_history = ""
        for message in self.memory.chat_memory.messages:
            if isinstance(message, HumanMessage):
                formatted_history += f"""
                    "voice_command": "{message.content.split('Voice command: ')[1].split('\\nQuestion:')[0]}",
                    "question": "{message.content.split('Question: ')[1].split('\\nWhiteboard:')[0]}",
                    "whiteboard_content": "{message.content.split('Whiteboard: ')[1]}",
                """
            elif isinstance(message, AIMessage):
                formatted_history += f"""
                    "response": "{message.content}"
                """
        return formatted_history

# Initialize the LanguageModelProcessor
llm_processor = LanguageModelProcessor()

# Request model for whiteboard interaction
class WhiteboardRequest(BaseModel):
    question: str
    whiteboard_content_latex_text: str
    command: str
    chat_history: str

@app.post("/whiteboard")
async def whiteboard_interaction(request: WhiteboardRequest):
    try:
        instructions, voice_instructions, updated_chat_history = llm_processor.process(
            voice_command=request.command,
            question=request.question,
            whiteboard_content=request.whiteboard_content_latex_text,
            chat_history=request.chat_history
        )
        return {"instructions": instructions, "voice_instructions": voice_instructions, "chat_history": updated_chat_history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import requests
import os



DG_API_KEY = os.getenv("DEEPGRAM_API_KEY")
MODEL_NAME = "aura-asteria-en"

@app.get("/speak/")
async def speak(text: str):
    DEEPGRAM_URL = f"https://api.deepgram.com/v1/speak?model={MODEL_NAME}&encoding=linear16&sample_rate=24000"
    headers = {
        "Authorization": f"Token {DG_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"text": text}

    response = requests.post(DEEPGRAM_URL, stream=True, headers=headers, json=payload)
    
    if response.status_code != 200:
        return {"error": f"Deepgram API Error {response.status_code}"}
    
    return StreamingResponse(response.iter_content(chunk_size=1024), media_type="audio/wav")
