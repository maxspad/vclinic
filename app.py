import streamlit as st
from uuid import UUID, uuid1, uuid4
from pydantic import BaseModel, Field
from rich import print
from typing import Literal, Optional
from openai import AzureOpenAI, AsyncAzureOpenAI
from agents import RunContextWrapper, set_default_openai_api, set_default_openai_client, Agent, Runner, set_tracing_disabled, function_tool, handoff, set_tracing_export_api_key
from dotenv import load_dotenv
import os
import asyncio

# Load environment variables
load_dotenv()

# Create OpenAI client using Azure OpenAI
openai_client = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    organization=os.getenv("AZURE_OPENAI_ORG_ID")
)
set_tracing_export_api_key(os.getenv("OPENAI_API_KEY"))
set_default_openai_client(openai_client, use_for_tracing=False)
set_default_openai_api('chat_completions')
set_tracing_disabled(disabled=False)

class Actor(BaseModel):
    pass

class Message(BaseModel):
    time : int
    content : str
    sender : Literal['patient','doctor']
    is_read : bool = False

class Encounter(BaseModel):
    time : int
    subject : str = Field(description='The main subject or title of the encounter, just a few words, like "Refill Request" or "Health Maintenance Visit"')
    short_description : Optional[str] = Field(None, description='A short, one sentence or less summary of the event')
    long_description : Optional[str] = Field(None, description='A longer, more detailed summary of the event, about one paragraph')
    message_history : list[Message] = []
    initiated_by : Literal['user','system']
    currently_open : bool = Field(False, description='Whether this encounter has been fully addressed or not.')

class Conversation(BaseModel):
    id : UUID = Field(default_factory=uuid4)
    participants : list[UUID]
    messages : list[Message] = []

class Event(BaseModel):
    tick : int = Field(description='When the event happened in simulation ticks')
    event_type : Literal['order_placed','patient_visit','message_from_patient','message_to_patient','patient_status_change','study_result']
    subject : str = Field(..., description='The main subject or title of the event, just a few words')
    short_description : Optional[str] = Field(None, description='A short, one sentence or less summary of the event')
    long_description : Optional[str] = Field(None, description='A longer, more detailed summary of the event, about one paragraph')
    content : Optional[str] = Field(None, description='The full text content of the event, such as the full text of a message or order')
    created_by : Literal['user','system'] = Field(description='Whether this event was created by the user (the user choosing to place an order/send a script or send a message) or whether it was created by the simulation manager')
    visible_to_patient : bool = Field(description='Whether or not this event is known to the patient yet')
    visible_to_user : bool = Field(description='Whether or not this event can be shown to the user yet')

class FaceSheet(BaseModel):
    id : str
    name_first : str
    name_last : str
    age : int
    sex : Literal['male','female']
    problems : list[str] = []
    medications : list[str] = []

class Patient(Actor):
    facesheet : FaceSheet
    backstory : str
    events : list[Event] = []

class User(Actor):
    name : str = 'Doctor'

class Context(BaseModel):
    current_tick : int
    user : User = Field(User(name='Doctor'))
    patients : dict[str, Patient]

bob_id = str(uuid4())
bob = Patient(facesheet=FaceSheet(id=bob_id,
                name_first='Bob', 
                name_last='Smith', 
                age=86,
                sex='male',
                problems=['htn','hld'], 
                medications=['lisinopril']),
            backstory='A crochety 86 year old trumper',
            events=[
                Event(tick=0, event_type='message_from_patient', subject='Establishing Care',
                      short_description='Bob was told he needed a PCP and so he messaged acting for an appointment',
                      created_by='system',
                      visible_to_patient=True,
                      visible_to_user=True,
                      content='Uhh, hi. My wife says I need to start seeing a doctor. Can we make an appointment?'),
                Event(tick=0, event_type='patient_visit', subject='Establishing Care',
                      short_description='Bob met with the doctor for the first time. He was started on lisinopril for his hypertension',
                      created_by='system',
                      visible_to_patient=True,
                      visible_to_user=True),
                Event(tick=0, event_type='order_placed', subject='Lisinopril Rx',
                      short_description='The doctor wrote bob a prescription for lisinopril',
                      created_by='system',
                      visible_to_patient=True,
                      visible_to_user=True),
                Event(tick=1, event_type='message_from_patient', subject='Refill Request',
                      short_description='Bob asked for a refill of his lisinopril, about a month late',
                      created_by='system',
                      visible_to_patient=True,
                      visible_to_user=True,
                      content='Hey Doc, can I have a refill of my lisinopril?')
            ])

alice_id = str(uuid4())
alice = Patient(facesheet=FaceSheet(
    id=alice_id,
    name_first='Alice', 
    name_last='Johnson', 
    age=93,
    sex='female',
    problems=['dm2','obesity'], 
    medications=[]),
    backstory='A sweet little 93 year old lady',
    events=[
        Event(tick=0, event_type='patient_visit', subject='Urinary Symptoms',
              short_description='Alice saw the doctor for some dysuria.',
              created_by='system',
              visible_to_patient=True,
              visible_to_user=True),
        Event(tick=0, event_type='order_placed', subject='Urinalysis and CBC',
              short_description='The doctor ordered a urinalysis and CBC',
              created_by='system',
              visible_to_patient=True,
              visible_to_user=True),
        Event(tick=1, event_type='study_result', subject='Urinalysis Result',
              content='Urinalysis: 3+ Leukocyte esterase, 3+ Nitrites. WBC: 100/hpf',
              created_by='system',
              visible_to_patient=False,
              visible_to_user=True)
    ])
            

start_context = Context(current_tick=1, patients={bob_id : bob, alice_id : alice})

# @function_tool
# def get_patient_info(wrapper : RunContextWrapper[Context]) -> list[Patient]:
#     '''Gets the current list of patients and their medical record info'''
#     print(f'get_patient_info')

#     return wrapper.context.patients

@function_tool
def get_patients_summary(wrapper : RunContextWrapper[Context]) -> list[FaceSheet]:
    '''Gets the current list of patients and their basic demographic information, problem lists, and medication lists'''
    print(f'get_patients_summary')
    return [pt.facesheet for pt in wrapper.context.patients.values()]


@function_tool
def get_new_messages_for_patient(wrapper : RunContextWrapper[Context], patient_id : str) -> str:
    '''Gets the list of messages sent from/to the doctor from a given patient on this simulation tick'''
    print('get_messages_for_patient')

    patient = wrapper.context.patients.get(patient_id)
    if not patient:
        toret = f'Error: No patient found with id {patient_id}'
        print(toret)
        return toret
    
    # If we found the patient, filter the patient's events only to "message_from_patient" or "message_to_patient"
    events = [event for event in patient.events if event.event_type in ['message_from_patient', 'message_to_patient'] and event.tick == wrapper.context.current_tick]
    if not events or not len(events):
        toret = f'No messages found for patient {patient.facesheet.name_first} {patient.facesheet.name_last}'
        print(toret)
        return toret

    toret = '\n'.join([f'{event.event_type}: "{event.content}"' for event in events])
    print(toret)
    return toret

@function_tool
def send_patient_a_message(wrapper : RunContextWrapper[Context], patient_id: str, subject : str, message: str) -> str:
    '''Sends a message to a patient from the doctor
    
    Args: 
        patient_id: The UUID of the patient to send the message to
        subject: The short subject of the message. If replying to a previous messsage, this should be the same subject as that message.
        message: The string message to send the patient
    '''
    print('send_patient_a_message')

    patient = wrapper.context.patients.get(patient_id)
    print(f'Got patient: {patient}')
    if not patient:
        toret = f'No patient found with id {patient_id}'
        print(toret)
        return toret
    
    new_message = Event(tick=wrapper.context.current_tick, event_type='message_to_patient', subject=subject, content=message,
                        created_by='user', visible_to_patient=True, visible_to_user=True)
    print(f'Created message {new_message}')
    patient.events.append(new_message)
    toret = f'Message sent to {patient.facesheet.name_first} {patient.facesheet.name_last}: "{message}"'
    print(toret)
    return toret

@function_tool
def get_new_results_for_patient(wrapper : RunContextWrapper[Context], patient_id : str) -> str:
    '''Gets the list of new study results for a given patient on this simulation tick'''
    print('get_new_results_for_patient')

    patient = wrapper.context.patients.get(patient_id)
    if not patient:
        toret = f'Error: No patient found with id {patient_id}'
        print(toret)
        return toret
    
    # If we found the patient, filter the patient's events only to "study_result"
    events = [event for event in patient.events if event.event_type == 'study_result' and event.tick == wrapper.context.current_tick]
    if not events or not len(events):
        toret = f'No new study results found for patient {patient.facesheet.name_first} {patient.facesheet.name_last}'
        print(toret)
        return toret

    toret = '\n'.join([f'{event.subject}: "{event.content}"' for event in events])
    print(toret)
    return toret

@function_tool
def place_order_for_patient(wrapper : RunContextWrapper[Context], patient_id : str, order_subject : str, order_description : str) -> str:
    '''Place a new order for a patient. Orders are medication prescriptions or changes, consults, lab studies, or imaging.
    
    Args:
        order_subject: A short, descriptive subject for the order (e.g., "CBC", "CMP, "MRI Brain", "Lisinopril 20mg").
        order_description: The full order requisition, a few sentences at most for a complex imaging order, even less for a lab order, and the full 
         sig/dose/frequency/etc for a prescription
    '''
    print(f'place_order_for_patient: patient_id={patient_id}, order_subject={order_subject}, order_description={order_description}')

    patient = wrapper.context.patients.get(patient_id)
    if not patient:
        toret = f'Error: No patient found with id {patient_id}'
        print(toret)
        return toret

    new_order = Event(tick=wrapper.context.current_tick, event_type='order_placed', subject=order_subject, content=order_description,
                      created_by='user', visible_to_patient=True, visible_to_user=True)
    patient.events.append(new_order)
    toret = f'Order placed for patient {patient.facesheet.name_first} {patient.facesheet.name_last}: "{order_description}"'
    print(toret)
    return toret


# medical_coder_agent_instructions = (
# '''You are an expert medical coder and biller. You receive handoff from the Clinic Manager RN Agent, who
# asks you to check a conversation between that agent and the physician for any orders placed by the physician.
# If you detect that one or more orders have been placed, you generate a succinct, technical, medical summary of each order in a markdown bullet point list.
# You then check with the user to make sure that is the order(s) they actually want placed.
# You then call a tool to submit the orders, one by one. 
# You then hand control back to the Clinic Manager RN Agent. You do NOT need to announce that you are handing back control.'''
# )
# medical_coder_agent_tool_description = '''Check for any orders placed by the physician in the conversation, and add them to the medical record if they were placed'''
# medical_coder_agent = Agent(
#     'Medical Coder and Biller',
#     tools=[place_order_for_patient],
#     instructions=medical_coder_agent_instructions
# )

# def medical_coder_agent_callback(context : RunContextWrapper[None]):
#     print('medical coder agent: handoff received')

# handoff_obj_medical_coder_agent = handoff(medical_coder_agent,
#                                           tool_description_override=medical_coder_agent_tool_description,
#                                           on_handoff=medical_coder_agent_callback)

summarizer_agent_instructions = (
'''You are a medical chart abstractor.
You are given a conversation between a physician and his/her AI clinic manager. Your job is to extract the clinically relevant action points from their 
conversation. Focus on creating succinct, clear, almost terse summaries. You should use bullet points whenever able. 

Focus ONLY on any patient messages sent, results reviewed, or orders placed. Organize your summary and 
format it using markdown as follows:

**Messages Sent**

- Message 1
- Message 2

**Results Reviewed**

- Result 1
- Result 2

**Orders Placed**

- Order 1
- Order 2
'''
)
summarizer_agent = Agent(
    'Summarizer',
    instructions=summarizer_agent_instructions
)


clinic_manager_agent_instructions = (
'''You are the clinic manager. You are a wise and experienced RN. You are professional always.
Your job is to respond to the user requests as best as possible, using the tools and context available to you. 
You are helpful, but you don\'t suggest any medical information. '''
)
clinic_manager_agent = Agent(
    'Clinic Manager',
    tools=[get_patients_summary, get_new_messages_for_patient, send_patient_a_message, get_new_results_for_patient, place_order_for_patient],  
    instructions=clinic_manager_agent_instructions,
    handoffs=[summarizer_agent]
    # handoffs=[handoff_obj_medical_coder_agent]
)



# If this is the first time this is being run, set up the conversation history
if 'history' not in st.session_state:
    st.session_state['history'] = [('ai', 'Hello Doctor, I am your clinic manager!')]
# Likewise set up the context
if 'context' not in st.session_state:
    st.session_state['context'] = start_context
# And the ChatGPT agent conversation history
if 'agent_history' not in st.session_state:
    st.session_state['agent_history'] = []

# c1, c2 = st.columns([3,1])
summary_response = ''

# Display the conversation history
for role, message in st.session_state['history']:
    with st.chat_message(name=role):
        st.write(message)
user_placeholder_for_current_turn = st.empty()
ai_placeholder_for_current_turn = st.empty()
user_input = st.chat_input()
if user_input: # if the user has typed something in 
    # Echo the user's input
    with user_placeholder_for_current_turn:
        with st.chat_message(name='user'):
            st.write(user_input)
    # Store it in the session output
    st.session_state['history'].append(('user', user_input))

    # Get a response
    response = asyncio.run(Runner.run(clinic_manager_agent, st.session_state['agent_history'] + [{'role': 'user', 'content': user_input}], context=st.session_state['context']))
    with ai_placeholder_for_current_turn:
        with st.chat_message(name='ai'):
            st.write(response.final_output)
        st.session_state['history'].append(('ai', response.final_output))

    # Add it to the agent's conversation history
    st.session_state['agent_history'] += (response.to_input_list())


# with c2:
#      # Summarize the actions taken to this point
#     summary_response = asyncio.run(Runner.run(summarizer_agent, st.session_state['agent_history'], context=st.session_state['context'])).final_output
#     st.markdown('## Clinic Day Summary')
#     st.write(summary_response)

with st.sidebar:
    st.write(st.session_state['context'].model_dump())