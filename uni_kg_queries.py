import streamlit as st
import networkx as nx
import os
from collections import Counter

st.set_page_config(
    page_title="University KG Explorer",
    page_icon="🎓",
    layout="centered",
)

@st.cache_resource
def load_graph():
    graph_file_path = "university_kg.graphml"
    if not os.path.exists(graph_file_path):
        st.error(f"Error: '{graph_file_path}' not found. Please make sure the file is in the same folder as this script.")
        return None
    try:
        g = nx.read_graphml(graph_file_path)
        if not nx.is_directed(g):
            g = g.to_directed()
        print("Knowledge Graph loaded successfully.")
        return g
    except Exception as e:
        st.error(f"Could not load or parse the graph file: {e}")
        return None

G = load_graph()

def find_node(name):
    if not G: return None
    for n in G.nodes():
        if n.lower() == name.lower():
            return n
    return None

st.title("🎓 University Knowledge Graph Explorer")
st.markdown("Use the sidebar to select a question and explore the university's offerings.")

st.sidebar.title("Select a Question")
question = st.sidebar.selectbox(
    "Choose one of the following questions:",
    [
        "1. Which courses teach a skill?",
        "2. What programs prepare for a career?",
        "3. Find courses similar to another.",
        "4. Find courses in a program.",
        "5. Show the path from a program to a career.",
        "6. Which programs cover a topic?",
        "7. Compare skills between two programs.",
        "8. What courses can a student take?",
        "9. Which course leads to most careers?",
        "10. What courses combine two skills?",
    ]
)

if not G:
    st.warning("Graph data could not be loaded. Please check the console for errors.")
    st.stop()

if "1." in question:
    st.subheader("1. Find Courses Teaching a Skill")
    skill_input = st.text_input("Enter Skill Name", placeholder="e.g., Python, Data Analysis", key="q1_input")
    if st.button("Find Courses", key="q1_btn"):
        with st.spinner("Searching..."):
            skill_node = find_node(skill_input)
            if not skill_node or G.nodes[skill_node].get('type') != 'Skill':
                st.warning(f"Skill '{skill_input}' not found.")
            else:
                courses = [p for p in G.predecessors(skill_node) if G.nodes[p].get('type') == 'Course']
                if not courses:
                    st.info("No courses found teaching this skill.")
                else:
                    st.markdown("#### Courses teaching this skill:")
                    for course in sorted(courses):
                        st.markdown(f"- {course}")

elif "2." in question:
    st.subheader("2. Find Programs Leading to a Career")
    career_input = st.text_input("Enter Career Title", placeholder="e.g., Software Engineer", key="q2")
    if st.button("Find Programs", key="q2_btn"):
        career_node = find_node(career_input)
        if not career_node or G.nodes[career_node].get('type') != 'Career':
            st.warning(f"Career '{career_input}' not found.")
        else:
            with st.spinner("Analyzing career paths..."):
                program_scores = Counter()
                relevant_courses = {p for p in G.predecessors(career_node) if G.nodes[p].get('type') == 'Course'}
                
                if not relevant_courses:
                    st.info(f"No specific courses found for the career '{career_input}'.")
                else:
                    for course in relevant_courses:
                        programs_for_course = {p for p in G.predecessors(course) if G.nodes[p].get('type') == 'Program'}
                        for prog in programs_for_course:
                            program_scores[prog] += 1
                    
                    if not program_scores:
                        st.info(f"No programs found containing the courses for '{career_input}'.")
                    else:
                        st.markdown("#### Top programs that prepare for this career:")
                        for prog, count in program_scores.most_common(5):
                            st.markdown(f"- **{prog}** (Offers {count} relevant course(s))")

elif "3." in question:
    st.subheader("3. Find Courses Similar to Another")
    course_input = st.text_input("Enter Course Name", placeholder="e.g., Introduction to Programming", key="q3_input")
    if st.button("Find Similar Courses", key="q3_btn"):
        with st.spinner("Comparing..."):
            course_node = find_node(course_input)
            if not course_node or G.nodes[course_node].get('type') != 'Course':
                st.warning(f"Course '{course_input}' not found.")
            else:
                target_skills = {s for s in G.successors(course_node) if G.nodes[s].get('type') == 'Skill'}
                if not target_skills:
                    st.info("Could not find skills for this course to compare.")
                else:
                    similarities = {}
                    for other_node, data in G.nodes(data=True):
                        if data.get('type') == 'Course' and other_node != course_node:
                            other_skills = {s for s in G.successors(other_node) if G.nodes[s].get('type') == 'Skill'}
                            overlap = len(target_skills.intersection(other_skills))
                            if overlap > 0:
                                similarities[other_node] = overlap
                    if not similarities:
                        st.info("No other courses with shared skills found.")
                    else:
                        sorted_similar = sorted(similarities.items(), key=lambda item: item[1], reverse=True)[:5]
                        st.markdown("#### Top similar courses (by shared skills):")
                        for name, count in sorted_similar:
                            st.markdown(f"- **{name}** ({count} shared skills)")

elif "4." in question:
    st.subheader("4. Find Skill Courses Within a Program")
    program_input_q4 = st.text_input("Enter Program Name", placeholder="e.g., BSc (Hons) Computer Science", key="q4_prog_input")
    skill_input_q4 = st.text_input("Enter Skill Name", placeholder="e.g., Python", key="q4_skill_input")
    if st.button("Find Courses in Program", key="q4_btn"):
        with st.spinner("Searching..."):
            program_node = find_node(program_input_q4)
            skill_node = find_node(skill_input_q4)
            if not program_node or G.nodes[program_node].get('type') != 'Program':
                st.warning(f"Program '{program_input_q4}' not found.")
            elif not skill_node or G.nodes[skill_node].get('type') != 'Skill':
                st.warning(f"Skill '{skill_input_q4}' not found.")
            else:
                program_courses = {s for s in G.successors(program_node) if G.nodes[s].get('type') == 'Course'}
                skill_courses = {p for p in G.predecessors(skill_node) if G.nodes[p].get('type') == 'Course'}
                matching_courses = program_courses.intersection(skill_courses)
                if not matching_courses:
                    st.info(f"No '{skill_input_q4}' courses found in the '{program_input_q4}' program.")
                else:
                    st.markdown(f"#### '{skill_input_q4}' courses in '{program_input_q4}':")
                    for course in sorted(matching_courses):
                        st.markdown(f"- {course}")

elif "5." in question:
    st.subheader("5. Show Path from Program to Career")
    program_input_q5 = st.text_input("Enter Starting Program Name", key="q5_prog_input")
    career_input_q5 = st.text_input("Enter Target Career Name", key="q5_career_input")
    if st.button("Find Path", key="q5_btn"):
        with st.spinner("Searching for path..."):
            program_node = find_node(program_input_q5)
            career_node = find_node(career_input_q5)
            if not program_node or G.nodes[program_node].get('type') != 'Program':
                st.warning(f"Program '{program_input_q5}' not found.")
            elif not career_node or G.nodes[career_node].get('type') != 'Career':
                st.warning(f"Career '{career_input_q5}' not found.")
            else:
                try:
                    path = nx.shortest_path(G, source=program_node, target=career_node)
                    st.success("Path found!")
                    st.markdown(" -> ".join([f"**({G.nodes[n].get('type', 'Node')})** {n}" for n in path]))
                except nx.NetworkXNoPath:
                    st.error(f"No path found in the graph from '{program_input_q5}' to '{career_input_q5}'.")

elif "6." in question:
    st.subheader("6. Find Programs Covering a Topic")
    topic_input = st.text_input("Enter Topic Name", placeholder="e.g., Artificial Intelligence, Internet of Things, Immersive Technologies", key="q6")
    if st.button("Find Programs", key="q6_btn"):
        topic_node = find_node(topic_input)
        if not topic_node or G.nodes[topic_node].get('type') != 'Topic':
            st.warning(f"Topic '{topic_input}' not found.")
        else:
            programs = set()
            courses_for_topic = {p for p in G.predecessors(topic_node) if G.nodes[p].get('type') == 'Course'}
            for course in courses_for_topic:
                programs_for_course = {p for p in G.predecessors(course) if G.nodes[p].get('type') == 'Program'}
                programs.update(programs_for_course)
            
            if not programs:
                st.info("No programs found that cover this topic.")
            else:
                st.markdown("#### Programs that cover this topic:")
                st.markdown("\n".join(f"- {prog}" for prog in sorted(programs)))

elif "7." in question:
    st.subheader("7. Compare Skills Between Two Programs")
    col1, col2 = st.columns(2)
    with col1:
        prog1_input = st.text_input("Enter First Program Name", key="q7_prog1")
    with col2:
        prog2_input = st.text_input("Enter Second Program Name", key="q7_prog2")

    if st.button("Compare Programs", key="q7_btn"):
        def get_program_skills(prog_name):
            prog_node = find_node(prog_name)
            if not prog_node or G.nodes[prog_node].get('type') != 'Program':
                return None
            
            program_skills = set()
            courses_in_program = {s for s in G.successors(prog_node) if G.nodes[s].get('type') == 'Course'}
            for course in courses_in_program:
                skills_in_course = {s for s in G.successors(course) if G.nodes[s].get('type') == 'Skill'}
                program_skills.update(skills_in_course)
            return program_skills

        with st.spinner("Comparing..."):
            skills1 = get_program_skills(prog1_input)
            skills2 = get_program_skills(prog2_input)

            if skills1 is None:
                st.warning(f"Program '{prog1_input}' not found.")
            elif skills2 is None:
                st.warning(f"Program '{prog2_input}' not found.")
            else:
                common = sorted(list(skills1.intersection(skills2)))
                unique1 = sorted(list(skills1 - skills2))
                unique2 = sorted(list(skills2 - skills1))
                
                st.markdown(f"#### Comparison between **{prog1_input}** and **{prog2_input}**")
                st.markdown(f"**Shared Skills ({len(common)}):**")
                st.markdown("- " + "\n- ".join(common) if common else "None")
                st.markdown(f"**Unique to {prog1_input} ({len(unique1)}):**")
                st.markdown("- " + "\n- ".join(unique1) if unique1 else "None")
                st.markdown(f"**Unique to {prog2_input} ({len(unique2)}):**")
                st.markdown("- " + "\n- ".join(unique2) if unique2 else "None")

elif "8." in question:
    st.subheader("8. Find Courses for a Student")
    prog_input_q8 = st.text_input("Enter Your Program Name", placeholder="e.g., BSc (Hons) Computer Science", key="q8_prog_input")
    skill_input_q8 = st.text_input("Enter Skill to Learn", placeholder="e.g., Data Analysis", key="q8_skill_input")
    if st.button("Find Available Courses", key="q8_btn"):
        if not find_node(prog_input_q8):
            st.warning(f"Program '{prog_input_q8}' not found, but searching for courses anyway...")
        
        with st.spinner("Searching..."):
            skill_node = find_node(skill_input_q8)
            if not skill_node or G.nodes[skill_node].get('type') != 'Skill':
                st.warning(f"Skill '{skill_input_q8}' not found.")
            else:
                courses = [p for p in G.predecessors(skill_node) if G.nodes[p].get('type') == 'Course']
                if not courses:
                    st.info(f"No courses found teaching '{skill_input_q8}'.")
                else:
                    st.markdown(f"#### Courses teaching **{skill_input_q8}**:")
                    for course in sorted(courses):
                        st.markdown(f"- {course}")

elif "9." in question:
    st.subheader("9. Which Course Leads to the Most Career Options?")
    if st.button("Calculate Now", key="q9_btn"):
        with st.spinner("Analyzing graph... This may take a moment."):
            course_careers = {}
            for node, data in G.nodes(data=True):
                if data.get('type') == 'Course':
                    descendants = nx.descendants(G, node)
                    careers_count = sum(1 for d in descendants if G.nodes[d].get('type') == 'Career')
                    if careers_count > 0:
                        course_careers[node] = careers_count
            
            if not course_careers:
                st.info("Could not determine career options from any course.")
            else:
                sorted_courses = sorted(course_careers.items(), key=lambda item: item[1], reverse=True)[:5]
                st.markdown("#### Top 5 courses by number of career options:")
                for name, count in sorted_courses:
                    st.markdown(f"- **{name}** ({count} options)")

elif "10." in question:
    st.subheader("10. Find Courses Combining Two Skills")
    col1, col2 = st.columns(2)
    with col1:
        skill1_input = st.text_input("Enter First Skill", key="q10_skill1")
    with col2:
        skill2_input = st.text_input("Enter Second Skill", key="q10_skill2")
    
    if st.button("Find Combining Courses", key="q10_btn"):
        with st.spinner("Searching..."):
            skill1_node = find_node(skill1_input)
            skill2_node = find_node(skill2_input)
            if not skill1_node or G.nodes[skill1_node].get('type') != 'Skill':
                st.warning(f"Skill '{skill1_input}' not found.")
            elif not skill2_node or G.nodes[skill2_node].get('type') != 'Skill':
                st.warning(f"Skill '{skill2_input}' not found.")
            else:
                courses1 = {p for p in G.predecessors(skill1_node) if G.nodes[p].get('type') == 'Course'}
                courses2 = {p for p in G.predecessors(skill2_node) if G.nodes[p].get('type') == 'Course'}
                intersection = courses1.intersection(courses2)
                if not intersection:
                    st.info(f"No single course found teaching both '{skill1_input}' and '{skill2_input}'.")
                else:
                    st.markdown("#### Courses teaching both skills:")
                    for course in sorted(intersection):
                        st.markdown(f"- {course}")