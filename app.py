import streamlit as st
import networkx as nx
import os
import google.generativeai as genai
from collections import Counter
import pandas as pd
import numpy as np
import ast 

st.set_page_config(
    page_title="🎓 AI Academic Advisor",
    page_icon="🧠",
    layout="wide"
)

GRAPH_FILE = "university_kg.graphml"
EMBEDDINGS_FILE = "embeddings.csv"

try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception:
    st.error("Google API Key not found. Please configure it in your Streamlit secrets.")
    st.stop()

# --- Data Loading ---

@st.cache_data
def load_embeddings_and_graph(graph_path, embeddings_path):
    """
    Loads both the knowledge graph and the pre-computed embeddings DataFrame.
    This function is cached to run only once.
    """
    try:
        g = nx.read_graphml(graph_path)
        if not nx.is_directed(g):
            g = g.to_directed()
    except FileNotFoundError:
        st.error(f"Knowledge Graph file not found at '{graph_path}'.")
        return None, None

    try:
        df = pd.read_csv(embeddings_path)
        df['embedding'] = df['embedding'].apply(ast.literal_eval)
    except FileNotFoundError:
        st.error(f"Embeddings file not found at '{embeddings_path}'. Please run the generation script first.")
        return None, None
    
    return g, df

G, embeddings_df = load_embeddings_and_graph(GRAPH_FILE, EMBEDDINGS_FILE)

if G is None or embeddings_df is None:
    st.stop()


def find_closest_node_by_embedding(query: str, node_type: str = None, top_k: int = 3, score_cutoff: float = 0.8):
    """
    Finds the best node match using a hybrid approach:
    1. First, check for a perfect, literal match to prevent loops.
    2. If no perfect match, use semantic search with embeddings.
    """
    if embeddings_df.empty:
        return None, []

    if node_type:
        filtered_df = embeddings_df[embeddings_df['type'] == node_type].copy()
    else:
        filtered_df = embeddings_df.copy()

    if filtered_df.empty:
        return None, []
        
    perfect_match = filtered_df[filtered_df['name'] == query]
    if not perfect_match.empty:
        node_name = perfect_match.iloc[0]['name']
        return node_name, [node_name]

    try:
        query_embedding = genai.embed_content(
            model='models/text-embedding-004',
            content=query,
            task_type="RETRIEVAL_QUERY"
        )['embedding']
    except Exception as e:
        st.error(f"Failed to create query embedding: {e}")
        return None, []

    filtered_df['similarity'] = filtered_df['embedding'].apply(
        lambda emb: np.dot(emb, query_embedding) / (np.linalg.norm(emb) * np.linalg.norm(query_embedding))
    )

    top_results = filtered_df.sort_values(by='similarity', ascending=False).head(top_k)

    if top_results.empty:
        return None, []
        
    top_match = top_results.iloc[0]
    suggestions = top_results['name'].tolist()

    verified_node = top_match['name'] if top_match['similarity'] >= score_cutoff else None
    
    return verified_node, suggestions

find_closest_node = find_closest_node_by_embedding

def find_courses_for_skill(skill_name: str) -> str:
    """Finds all courses that teach a specific skill."""
    verified_skill, suggestions = find_closest_node(skill_name, "Skill", score_cutoff=0.75)
    if not verified_skill:
        return f"I couldn't find an exact match for the skill '{skill_name}'. Did you mean one of these? {', '.join(suggestions)}" if suggestions else f"Sorry, I couldn't find any skill matching '{skill_name}'."
    
    courses = sorted([p for p, _ in G.in_edges(verified_skill) if G.nodes[p].get('type') == 'Course'])
    if not courses:
        return f"No courses were found that teach the skill: **{verified_skill}**."
    return f"The following courses teach **{verified_skill}**:\n" + "\n".join(f"- {c}" for c in courses)

def find_programs_for_career(career_name: str) -> str:
    """Finds academic programs that prepare students for a specific career."""
    verified_career, suggestions = find_closest_node(career_name, "Career", score_cutoff=0.75)
    if not verified_career:
        return f"I couldn't find an exact match for the career '{career_name}'. Did you mean one of these? {', '.join(suggestions)}" if suggestions else f"Sorry, I couldn't find any career matching '{career_name}'."
        
    program_scores = Counter()
    relevant_courses = {p for p, _ in G.in_edges(verified_career) if G.nodes[p].get('type') == 'Course'}
    for course in relevant_courses:
        programs_for_course = {p for p, _ in G.in_edges(course) if G.nodes[p].get('type') == 'Program'}
        for prog in programs_for_course:
            program_scores[prog] += 1
    if not program_scores:
        return f"No programs were found containing courses that directly lead to a career as a **{verified_career}**."
    result = f"The best programs to prepare for a **{verified_career}** career are:\n"
    for prog, count in program_scores.most_common(5):
        result += f"- **{prog}** (offers {count} relevant course(s))\n"
    return result

def compare_skills_between_programs(program_1_name: str, program_2_name: str) -> str:
    """Compares the skills taught in two different academic programs."""
    prog1, prog1_sugg = find_closest_node(program_1_name, "Program")
    if not prog1:
        return f"Could not find a program matching '{program_1_name}'. Did you mean: {', '.join(prog1_sugg)}" if prog1_sugg else f"Could not find a program matching '{program_1_name}'."

    prog2, prog2_sugg = find_closest_node(program_2_name, "Program")
    if not prog2:
        return f"Could not find a program matching '{program_2_name}'. Did you mean: {', '.join(prog2_sugg)}" if prog2_sugg else f"Could not find a program matching '{program_2_name}'."

    def get_program_skills(prog_node):
        skills = set()
        courses_in_program = {s for _, s in G.out_edges(prog_node) if G.nodes[s].get('type') == 'Course'}
        for course in courses_in_program:
            skills_in_course = {s for _, s in G.out_edges(course) if G.nodes[s].get('type') == 'Skill'}
            skills.update(skills_in_course)
        return skills

    skills1, skills2 = get_program_skills(prog1), get_program_skills(prog2)
    common = sorted(list(skills1.intersection(skills2)))
    unique1 = sorted(list(skills1 - skills2))
    unique2 = sorted(list(skills2 - skills1))
    
    result = f"**Comparison between {prog1} and {prog2}**\n\n"
    result += f"**Shared Skills ({len(common)}):**\n" + ("\n".join(f"- {s}" for s in common) if common else "_None_") + "\n\n"
    result += f"**Skills Unique to {prog1} ({len(unique1)}):**\n" + ("\n".join(f"- {s}" for s in unique1) if unique1 else "_None_") + "\n\n"
    result += f"**Skills Unique to {prog2} ({len(unique2)}):**\n" + ("\n".join(f"- {s}" for s in unique2) if unique2 else "_None_")
    return result

def find_similar_courses(course_name: str) -> str:
    """Finds courses that are similar to a given course based on shared skills."""
    verified_course, suggestions = find_closest_node(course_name, "Course")
    if not verified_course:
        return f"I couldn't find a course matching '{course_name}'. Did you mean one of these? {', '.join(suggestions)}" if suggestions else f"Sorry, I couldn't find any course matching '{course_name}'."
        
    target_skills = {s for _, s in G.out_edges(verified_course) if G.nodes[s].get('type') == 'Skill'}
    if not target_skills:
        return f"Could not find any skills for '{verified_course}' to compare."
        
    similarities = {}
    course_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'Course' and n != verified_course]

    for other_node in course_nodes:
        other_skills = {s for _, s in G.out_edges(other_node) if G.nodes[s].get('type') == 'Skill'}
        overlap = len(target_skills.intersection(other_skills))
        if overlap > 0:
            similarities[other_node] = overlap
            
    if not similarities:
        return f"No other courses with shared skills were found for '{verified_course}'."
        
    sorted_similar = sorted(similarities.items(), key=lambda item: item[1], reverse=True)[:5]
    result = f"Based on shared skills, courses similar to **{verified_course}** are:\n"
    for name, count in sorted_similar:
        result += f"- **{name}** ({count} shared skills)\n"
    return result

def find_courses_in_program_with_skill(program_name: str, skill_name: str) -> str:
    """Finds courses within a specific program that teach a particular skill."""
    program_node, prog_sugg = find_closest_node(program_name, "Program")
    if not program_node:
        return f"Program '{program_name}' not found. Did you mean: {', '.join(prog_sugg)}" if prog_sugg else f"Program '{program_name}' not found."

    skill_node, skill_sugg = find_closest_node(skill_name, "Skill")
    if not skill_node:
        return f"Skill '{skill_name}' not found. Did you mean: {', '.join(skill_sugg)}" if skill_sugg else f"Skill '{skill_name}' not found."

    program_courses = {s for _, s in G.out_edges(program_node) if G.nodes[s].get('type') == 'Course'}
    skill_courses = {p for p, _ in G.in_edges(skill_node) if G.nodes[p].get('type') == 'Course'}
    matching_courses = sorted(list(program_courses.intersection(skill_courses)))
    
    if not matching_courses:
        return f"No courses teaching **{skill_node}** were found in the **{program_node}** program."
        
    result = f"Courses in **{program_node}** that teach **{skill_node}**:\n"
    result += "\n".join(f"- {course}" for course in matching_courses)
    return result

def show_path_from_program_to_career(program_name: str, career_name: str) -> str:
    """Shows a potential educational and professional path from a program to a career."""
    program_node, prog_sugg = find_closest_node(program_name, "Program")
    if not program_node:
        return f"Program '{program_name}' not found. Did you mean: {', '.join(prog_sugg)}" if prog_sugg else f"Program '{program_name}' not found."

    career_node, career_sugg = find_closest_node(career_name, "Career")
    if not career_node:
        return f"Career '{career_name}' not found. Did you mean: {', '.join(career_sugg)}" if career_sugg else f"Career '{career_name}' not found."

    try:
        path = nx.shortest_path(G, source=program_node, target=career_node)
        path_str = " → ".join([f"**{n}** ({G.nodes[n].get('type', 'Node')})" for n in path])
        return f"A potential path from **{program_node}** to **{career_node}** is:\n{path_str}"
    except nx.NetworkXNoPath:
        return f"Sorry, no direct path could be found in the knowledge graph from **{program_node}** to **{career_node}**."

def find_programs_covering_topic(topic_name: str) -> str:
    """Finds which programs cover a specific topic, like 'Artificial Intelligence'."""
    topic_node, topic_sugg = find_closest_node(topic_name, "Topic", top_k=5)
    nodes_to_process = []
    if topic_node:
        nodes_to_process.append(topic_node)
    if topic_sugg:
        for sugg in topic_sugg:
            if topic_name.lower() in sugg.lower() and sugg not in nodes_to_process:
                nodes_to_process.append(sugg)
    
    if not nodes_to_process:
        return f"Sorry, I couldn't find any topic areas related to '{topic_name}' in the knowledge base."
    
    programs = set()
    for node in nodes_to_process:
        courses_for_topic = {p for p, _ in G.in_edges(topic_node) if G.nodes[p].get('type') == 'Course'}
        for course in courses_for_topic:
            programs_for_course = {p for p, _ in G.in_edges(course) if G.nodes[p].get('type') == 'Program'}
            programs.update(programs_for_course)
        
    if not programs:
        return f"While I found topic areas related to '{topic_name}' (like '{', '.join(nodes_to_process)}'), no specific programs are currently linked to them."
    result = f"Based on your query for **{topic_name}**, I found the following programs:\n"
    result += "\n".join(f"- {prog}" for prog in sorted(list(programs)))
    return result

def find_course_leading_to_most_careers() -> str:
    """Calculates which single course in the knowledge base leads to the most career options."""
    course_careers = {}
    course_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'Course']
    
    for course in course_nodes:
        descendants = nx.descendants(G, course)
        careers_count = sum(1 for d in descendants if G.nodes[d].get('type') == 'Career')
        if careers_count > 0:
            course_careers[course] = careers_count
            
    if not course_careers:
        return "Could not determine career options from any course."
        
    sorted_courses = sorted(course_careers.items(), key=lambda item: item[1], reverse=True)[:5]
    result = "The top 5 courses that lead to the most career options are:\n"
    for name, count in sorted_courses:
        result += f"- **{name}** ({count} career options)\n"
    return result

def find_courses_combining_two_skills(skill_1_name: str, skill_2_name: str) -> str:
    """Finds courses that teach two different specified skills together."""
    skill1_node, skill1_sugg = find_closest_node(skill_1_name, "Skill")
    if not skill1_node:
        return f"Skill '{skill_1_name}' not found. Did you mean: {', '.join(skill1_sugg)}" if skill1_sugg else f"Skill '{skill_1_name}' not found."

    skill2_node, skill2_sugg = find_closest_node(skill_2_name, "Skill")
    if not skill2_node:
        return f"Skill '{skill_2_name}' not found. Did you mean: {', '.join(skill2_sugg)}" if skill2_sugg else f"Skill '{skill_2_name}' not found."

    courses1 = {p for p, _ in G.in_edges(skill1_node) if G.nodes[p].get('type') == 'Course'}
    courses2 = {p for p, _ in G.in_edges(skill2_node) if G.nodes[p].get('type') == 'Course'}
    intersection = sorted(list(courses1.intersection(courses2)))
    
    if not intersection:
        return f"No single course was found that teaches both **{skill1_node}** and **{skill2_node}**."
        
    result = f"Courses teaching both **{skill1_node}** and **{skill2_node}**:\n"
    result += "\n".join(f"- {course}" for course in intersection)
    return result


SYSTEM_PROMPT = """
You are a helpful and clever AI Academic Advisor. Your primary goal is to provide accurate, relevant information by using tools that connect to a university's knowledge graph.

**IMPORTANT: Your most critical task is to expand abbreviations, informal terms, or partial names into their full, formal versions before calling any tool.** This is essential for matching with the formal names in the knowledge graph.
- For instance, if a user asks about 'CS', you must expand it to 'Computer Science' in the tool call.

Apply this expansion logic broadly using the following examples as a guide:
-   CS → Computer Science
-   AI → Artificial Intelligence
-   ML → Machine Learning
-   Data Sci → Data Science
-   Web Dev → Web Development
-   Cyber Sec → Cybersecurity
-   DB → Database
-   Bus → Business
-   Econ → Economics
-   Stats → Statistics

**Conversation and Error Handling Strategy:**
Here is how you MUST handle conversations:

1.  If the user's query is ambiguous and a tool provides a list of suggestions, you must present these suggestions clearly to the user and ask them to clarify.
2.  When the user selects an option you've provided, you MUST treat that selection as the correct and verified entity for the next tool call.
3.  **Crucially, you must avoid asking the user the same question repeatedly.** If you have provided a list of options, the user picks one, and the tool still fails or returns the same suggestions, do not repeat the question. Instead, assume there is a temporary system error. Apologize to the user for the technical difficulty and ask them to try a different, related query.
"""

# --- Streamlit UI and Chat Logic ---

st.title("🎓 AI Academic Advisor (Semantic Search)")
st.caption("I am an AI assistant powered by your university's Knowledge Graph. Ask me anything!")

st.sidebar.title("Example Questions")
st.sidebar.markdown("""
- What courses teach Python?
- Compare the CS and Econ programs.
- Which program is best for a Software Engineer career?
- What courses are similar to Information in Organisations?
- Show the path from the Marketing to a Data Analyst career.
- What programs cover the topic Artificial Intelligence?
- I'm in CS, what courses teach Data Analysis?
- Which single course opens up the most career options?
- What courses teach both SQL and Database Design?
""")

if "model" not in st.session_state:
    st.session_state.model = genai.GenerativeModel(
        model_name='gemini-2.0-flash', 
        system_instruction=SYSTEM_PROMPT,
        tools=[
            find_courses_for_skill,
            find_programs_for_career,
            compare_skills_between_programs,
            find_similar_courses,
            find_courses_in_program_with_skill,
            show_path_from_program_to_career,
            find_programs_covering_topic,
            find_course_leading_to_most_careers,
            find_courses_combining_two_skills
        ]
    )

if "chat" not in st.session_state:
    st.session_state.chat = st.session_state.model.start_chat(enable_automatic_function_calling=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if prompt := st.chat_input("Ask me about programs, skills, or careers..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        response = st.session_state.chat.send_message(prompt)
    
    st.session_state.messages.append({"role": "assistant", "content": response.text})
    with st.chat_message("assistant"):
        st.markdown(response.text)