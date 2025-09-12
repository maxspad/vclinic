import streamlit as st
from uuid import UUID, uuid1, uuid4
from pydantic import BaseModel, Field
from rich import print
from typing import Literal, Optional
from openai import AzureOpenAI, AsyncAzureOpenAI
from openai.types.shared import Reasoning
from agents import ModelSettings, RunContextWrapper, set_default_openai_api, set_default_openai_client, Agent, Runner, set_tracing_disabled, function_tool, handoff, set_tracing_export_api_key
from dotenv import load_dotenv
import os
import asyncio

# Load environment variables
# load_dotenv()
print("AZURE_OPENAI_API_KEY:", os.getenv("AZURE_OPENAI_API_KEY"))
print("AZURE_OPENAI_API_VERSION:", os.getenv("AZURE_OPENAI_API_VERSION"))
print("AZURE_OPENAI_ENDPOINT:", os.getenv("AZURE_OPENAI_ENDPOINT"))
print("AZURE_OPENAI_DEPLOYMENT:", os.getenv("AZURE_OPENAI_DEPLOYMENT"))
print("AZURE_OPENAI_ORG_ID:", os.getenv("AZURE_OPENAI_ORG_ID"))
print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
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

class Event(BaseModel):
    simulation_week : int = Field(description='When the event happened in simulated weeks')
    event_type : Literal['order_placed','patient_visit','message_from_patient','message_to_patient','patient_status_change','study_result']
    subject : str = Field(..., description='The main subject or title of the event, just a few words')
    short_description : Optional[str] = Field(None, description='A short, one sentence or less summary of the event')
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
    current_week : int
    user : User = Field(User(name='Doctor'))
    patients : dict[str, Patient]

bob_id = 'bob_smith'
bob = Patient(facesheet=FaceSheet(id=bob_id,
                name_first='Bob', 
                name_last='Smith', 
                age=86,
                sex='male',
                problems=['htn','hld'], 
                medications=['lisinopril 20mg BID']),
            backstory='A crochety 86 year old',
            events=[
                Event(simulation_week=0, event_type='message_from_patient', subject='Establishing Care',
                      short_description='Bob was told he needed a PCP and so he messaged acting for an appointment',
                      created_by='system',
                      visible_to_patient=True,
                      visible_to_user=True,
                      content='Uhh, hi. My wife says I need to start seeing a doctor. Can we make an appointment?'),
                Event(simulation_week=0, event_type='patient_visit', subject='Establishing Care',
                      short_description='Bob met with the doctor for the first time. He was started on lisinopril for his hypertension',
                      created_by='system',
                      visible_to_patient=True,
                      visible_to_user=True),
                Event(simulation_week=0, event_type='order_placed', subject='Lisinopril Rx',
                      short_description='The doctor wrote bob a prescription for lisinopril',
                      created_by='system',
                      visible_to_patient=True,
                      visible_to_user=True),
                Event(simulation_week=1, event_type='message_from_patient', subject='Refill Request',
                      short_description='Bob asked for a refill of his lisinopril, about a month late',
                      created_by='system',
                      visible_to_patient=True,
                      visible_to_user=True,
                      content='Hey Doc, can I have a refill of my lisinopril?')
            ])

alice_id = 'alice_johnson'
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
        Event(simulation_week=0, event_type='patient_visit', subject='Urinary Symptoms',
              short_description='Alice saw the doctor for some dysuria.',
              created_by='system',
              visible_to_patient=True,
              visible_to_user=True),
        Event(simulation_week=0, event_type='order_placed', subject='Urinalysis and CBC',
              short_description='The doctor ordered a urinalysis and CBC',
              created_by='system',
              visible_to_patient=True,
              visible_to_user=True),
        Event(simulation_week=1, event_type='study_result', subject='Urinalysis Result',
              content='Urinalysis: 3+ Leukocyte esterase, 3+ Nitrites. WBC: 100/hpf',
              created_by='system',
              visible_to_patient=False,
              visible_to_user=True)
    ])
            
# --- New Patient 1: 34-year-old female trying to conceive ---
julia_id = 'julia_nguyen'
julia = Patient(
    facesheet=FaceSheet(
        id=julia_id,
        name_first='Julia',
        name_last='Nguyen',
        age=34,
        sex='female',
        problems=['trying_to_conceive'],
        medications=[]
    ),
    backstory='34-year-old software designer actively trying to get pregnant over the last 3 months.',
    events=[
        Event(
            simulation_week=0,
            event_type='patient_visit',
            subject='Preconception Counseling',
            short_description='Discussed cycle tracking and lifestyle factors; plan for prenatal vitamins.',
            created_by='system',
            visible_to_patient=True,
            visible_to_user=True
        ),
        Event(
            simulation_week=0,
            event_type='order_placed',
            subject='Prenatal Vitamin Rx',
            short_description='Started daily prenatal vitamin with folic acid.',
            content='Prenatal vitamin PO daily with 0.4–0.8 mg folic acid.',
            created_by='system',
            visible_to_patient=True,
            visible_to_user=True
        ),
        Event(
            simulation_week=1,
            event_type='study_result',
            subject='Serum β-hCG Result',
            short_description='Negative pregnancy test after a late period.',
            content='Serum β-hCG: <5 mIU/mL (negative). Testing done because of period 4 days late.',
            created_by='system',
            visible_to_patient=False,
            visible_to_user=True
        )
    ]
)

# --- New Patient 2: 23-year-old male with illness anxiety disorder ---
liam_id = 'liam_roberts'
liam = Patient(
    facesheet=FaceSheet(
        id=liam_id,
        name_first='Liam',
        name_last='Roberts',
        age=23,
        sex='male',
        problems=['illness_anxiety_disorder'],
        medications=[]
    ),
    backstory='23-year-old graduate student with a history of frequent online health searches and reassurance seeking.',
    events=[
        Event(
            simulation_week=0,
            event_type='message_from_patient',
            subject='Worried About Chest Sensations',
            short_description='Reports intermittent chest “twinges” after reading about heart disease.',
            content='Hi—should I be worried about quick chest twinges I felt after studying late? It lasted a second.',
            created_by='system',
            visible_to_patient=True,
            visible_to_user=True
        ),
        Event(
            simulation_week=0,
            event_type='patient_visit',
            subject='Evaluation of Chest Discomfort',
            short_description='Clinic visit for reassurance and assessment; benign exam.',
            created_by='system',
            visible_to_patient=True,
            visible_to_user=True
        ),
        Event(
            simulation_week=0,
            event_type='order_placed',
            subject='ECG',
            short_description='Baseline ECG ordered during visit.',
            content='12-lead ECG ordered as baseline during visit for chest sensations.',
            created_by='system',
            visible_to_patient=True,
            visible_to_user=True
        ),
        Event(
            simulation_week=1,
            event_type='message_from_patient',
            subject='New Symptom Concern',
            short_description='Asks about a mole after reading about melanoma.',
            content='Hello again—should I be concerned about a small mole on my shoulder I just noticed?',
            created_by='system',
            visible_to_patient=True,
            visible_to_user=True
        )
    ]
)

# --- Add to simulation context ---
start_context = Context(
    current_week=1,
    patients={
        bob_id: bob,
        alice_id: alice,
        julia_id: julia,
        liam_id: liam,
    }
)


@function_tool
def get_patients_summary(wrapper : RunContextWrapper[Context]) -> list[FaceSheet]:
    '''Gets the current list of patients and their basic demographic information, problem lists, and medication lists'''
    print(f'get_patients_summary')
    return [pt.facesheet for pt in wrapper.context.patients.values()]


@function_tool
def get_messages_for_patient(wrapper : RunContextWrapper[Context], patient_id : str, new_only : bool = True) -> str:
    '''Gets the list of messages sent from/to the doctor from a given patient on this simulation week

    Args:
        patient_id: The string id of the patient to get messages for
        new_only: If True, only returns messages from the current simulation week. If False, returns all messages.

    Returns:
        A string containing the list of messages.
    '''
    print('get_messages_for_patient')

    patient = wrapper.context.patients.get(patient_id)
    if not patient:
        toret = f'Error: No patient found with id {patient_id}'
        print(toret)
        return toret
    
    # If we found the patient, filter the patient's events only to "message_from_patient" or "message_to_patient"
    events = [event for event in patient.events if event.event_type in ['message_from_patient', 'message_to_patient']]
    if new_only:
        events = [event for event in events if event.simulation_week == wrapper.context.current_week]
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
        patient_id: The string id of the patient to send the message to
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
    
    new_message = Event(simulation_week=wrapper.context.current_week, event_type='message_to_patient', subject=subject, content=message,
                        created_by='user', visible_to_patient=True, visible_to_user=True)
    print(f'Created message {new_message}')
    patient.events.append(new_message)
    toret = f'Message sent to {patient.facesheet.name_first} {patient.facesheet.name_last}: "{message}"'
    print(toret)
    st.session_state['event_log'].append(toret)
    return toret

@function_tool
def get_results_for_patient(wrapper : RunContextWrapper[Context], patient_id : str, new_only : bool = True) -> str:
    '''Gets the list of new study results for a given patient on this simulation week. 
    
    Args:
        new_only: If True, only returns results from the current simulation week. If False, returns all results.

    Returns:
        A string containing the list of study results.
    '''
    print('get_new_results_for_patient')

    patient = wrapper.context.patients.get(patient_id)
    if not patient:
        toret = f'Error: No patient found with id {patient_id}'
        print(toret)
        return toret
    
    # If we found the patient, filter the patient's events only to "study_result"
    events = [event for event in patient.events if event.event_type == 'study_result']
    if new_only:
        events = [event for event in events if event.simulation_week == wrapper.context.current_week]
    if not events or not len(events):
        toret = f'No new study results found for patient {patient.facesheet.name_first} {patient.facesheet.name_last}'
        print(toret)
        return toret

    toret = '\n'.join([f'{event.subject}: "{event.content}"' for event in events])
    print(toret)
    # st.session_state['event_log'].append(toret)
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

    new_order = Event(simulation_week=wrapper.context.current_week, event_type='order_placed', subject=order_subject, content=order_description,
                      created_by='user', visible_to_patient=True, visible_to_user=True)
    patient.events.append(new_order)
    toret = f'Order placed for patient {patient.facesheet.name_first} {patient.facesheet.name_last}: "{order_description}"'
    st.session_state['event_log'].append(toret)
    print(toret)
    return toret


@function_tool
def patient_progression_add_event(wrapper : RunContextWrapper[Context], patient_id : str, event : Event):
    '''Add a new event to a patient's record in order to progress the patient in the simulation. 
    For the event "content" attribute, include a short, 1 sentence justification for you adding this event'''

    print(f'patient_progression_add_event: patient_id={patient_id}, event={event}')
    patient = wrapper.context.patients.get(patient_id)
    if not patient:
        toret = f'Error: No patient found with id {patient_id}'
        print(toret)
        return toret

    patient.events.append(event)
    toret = f'{patient.facesheet.name_first} {patient.facesheet.name_last}: "{event.content}"'
    st.session_state['event_log'].append(toret)
    print(toret)
    return toret

patient_progression_agent_instructions = (
'''You are an expert physician and medical educator who is managing an active simulation to teach medical students longitudinal patient care.

You receive two items as input:
- A current_week integer: This is the current time of the simulation. 
- A patient record: This is a nested data structure containing patient demographics, problem lists, medication lists, and, most importantly, *events*.
Your job is to progress the patient forward in time, based on their CURRENT problems, medications, and previous events. You do NOT provide any of your 
own care. If a patient hasn't been adequately treated for something, it is NOT your job to correct the mistake. You PREDICT what happends *as a result of* any mistake (or correct treatment). 

To do so, you think step-by-step:
- First, you analyze the patient's current status and history, including their previous events
- Second, you use your medical and educational expertise to formulate an advancement plan. 
This plan is a list of events are events that *you* create. These events can be laboratory or imaging results, 
patient portal messages sent from the patient to their doctor, consultant recommendations, or misecellaneous patient status updates, like the patient improving, or getting sicker, or needing to go to the ER. 
You write these planned events down as a list. Remember - do NOT actually care for the patient. Just use your expertise to forecast what the patient would happen to the patient or what the patient would do in response given the current state. 
- Finally, for each planned event in the list, you use tools to add them to the patient's record. If you are submitting a status update, you should decide if you need to
send a message to the doctor from the patient letting the doctor know about the status update. 

You must keep keep the simulation engaging and educational, but not overwhelming. Not every patient will need new events generated every week of the simulation. 
'''
)
patient_progression_agent = Agent(
    name='Patient Simulation Progression Agent',
    instructions=patient_progression_agent_instructions,
    tools=[patient_progression_add_event],
    # model_settings=ModelSettings(reasoning=Reasoning(effort='high'), verbosity='medium')
)

@function_tool
async def end_clinic_day(wrapper : RunContextWrapper[Context]) -> str: 
    '''Ends the clinic day and prepares to progress to the next time step in the simulation'''

    print(f'end_clinic_day')
    current_week = wrapper.context.current_week
    next_week = current_week + 1
    wrapper.context.current_week = next_week

    progression_result = await Runner.run(patient_progression_agent, input=wrapper.context.model_dump_json(), context=wrapper.context)

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
'''You are the clinic manager. You are a wise and experienced RN. You are professional always, but not above the very occasional banter (though you don't talk about your lighter side)
You are the main user interface for the physician managing a longitudinal continuity clinic.
You try to use bullet points when appropriate to keep things clear and concise. Use bolding, italicizing, underlining, etc in order to make the information clear and easy to read.

Your job is to respond to the user requests as best as possible, using the tools and context available to you. 
Make sure to let the user know that when they are finished with the clinic day, they can let you know in order to progress forward in time. 
You are helpful, but you don\'t suggest any specific medical information.

If the user requests to end the clinic day,
first generate a summary of their actions to this point. ALWAYS confirm that the user does not want to go back and make any more changes, and warn 
them that if they end the day, they won't be able to go back. If they confirm, end the day. 

If the user requests to see messages or results, offer them the newest ones first, but remind them they can ask for all messages or results.
'''
)
clinic_manager_agent = Agent(
    'Clinic Manager',
    tools=[get_patients_summary, get_messages_for_patient, send_patient_a_message, 
           get_results_for_patient, place_order_for_patient, end_clinic_day],  
    instructions=clinic_manager_agent_instructions,
    handoffs=[summarizer_agent],
    # model_settings=ModelSettings(reasoning=Reasoning(effort='low'), verbosity='medium')
    # handoffs=[handoff_obj_medical_coder_agent]
)

# If this is the first time this is being run, set up the conversation history
if 'history' not in st.session_state:
    st.session_state['history'] = [('ai',
        asyncio.run(Runner.run(clinic_manager_agent, input='Introduce yourself and write a brief overview of your cababilities. No need to say "Sure!" or "Okay!" in response to this message.', context=start_context)).final_output
    )]
# Likewise set up the context
if 'context' not in st.session_state:
    st.session_state['context'] = start_context
# And the ChatGPT agent conversation history
if 'agent_history' not in st.session_state:
    st.session_state['agent_history'] = []
if 'event_log' not in st.session_state:
    st.session_state['event_log'] = []

# c1, c2 = st.columns(2)

# with c2:
# st.header(f'Week {st.session_state["context"].current_week}')
# to_write = '\n'.join([f'- {ev}' for ev in st.session_state['event_log']])
# print(to_write)
# st.markdown(to_write)
# # st.write(st.session_state['event_log'])

# with c1:
#     with st.container(height=600):
        # Display the conversation history
st.header('🥼 Virtual Longitudinal Clinic')
st.caption('Using AI agents to drive a long-term, dynamic simulation of patient care.')
chat_tab, event_log_tab, help_tab = st.tabs(['💬 Chat', '📋 Event Log', '❔ Help'])
with chat_tab:
    with st.container(height=500):
        for role, message in st.session_state['history']:
            with st.chat_message(name=role):
                st.caption(f'{"Clinic Manager" if role == "ai" else "User"} - Week {st.session_state["context"].current_week}')
                st.write(message)
        user_placeholder_for_current_turn = st.empty()
        ai_placeholder_for_current_turn = st.empty()
        end_of_day_placeholder = st.empty()
        user_input = st.chat_input()
        current_week = st.session_state['context'].current_week

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

            # Check if the user ended the day
            if st.session_state['context'].current_week != current_week:
                del st.session_state['agent_history']
                del st.session_state['history']
                st.balloons()
                end_of_day_placeholder.success(f'Clinic day ended. It is now week {st.session_state["context"].current_week} of the simulation.')

with event_log_tab:
    st.caption('This helps you understand what the simulation is doing, so you can peek behind the scenes. It keeps track of all of the events initiated by the system and the user. ')
    
    for patient in st.session_state['context'].patients.values():
        st.subheader(f'Patient: {patient.facesheet.name_first} {patient.facesheet.name_last}')
        for event in patient.events:
                badge_col, text_col = st.columns([1,15])
                with badge_col:
                    if event.created_by == 'user':
                        st.badge('User', color='green')
                    else:
                        st.badge('AI', color='blue')
                with text_col:
                    st.write(f'Week {event.simulation_week}: {event.subject}')
                    if event.content:
                        st.write(f'- {event.content}')


about_this_simulation = (
'''This is a simulated virtual clinic which employs AI agents to simulate patient evolution over time. You play as a physician who has a panel
of patients. These patients can grow and change over time in response to your actions. Like in real life, you can look up your patients,
examine their histories, correspond with them via portal messages, and prescribe them medications or consults. At the end of your "clinic day"
an AI agent will use your actions and the patient's histories and progress the simulation, creating new encounters, problems, and emergent behavior over time

In this relatively simple example, there are only a few patients. However, it serves as a proof of concept to show how dynamic agent-based simulations can run semi-autonomously. 
**Make sure to check the Event Log to see how the simulation unfolds!**

If you have questions, reach out to maxspad@med.umich.edu. Thanks!'''
)
with help_tab:
    st.header('About This Simulation')
    st.write(about_this_simulation)