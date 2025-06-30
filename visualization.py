# visualizations.py
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from typing import Dict, List, Set
from datetime import datetime, timedelta
import pandas as pd
import logging

# Configure logging
logger = logging.getLogger(__name__)

def plot_conflict_graph(graph: Dict[int, Set[int]], courses: Dict, filename: str = "conflict_graph.png"):
    """
    Visualizes the conflict graph where nodes are courses and edges represent conflicts.

    Args:
        graph: Adjacency list representation {course_id: {conflicting_course_ids}}
        courses: Dictionary of course details {course_id: course_data}
        filename: Name of the file to save the plot
    """
    logger.info(f"Generating conflict graph visualization: {filename}")
    G = nx.Graph()

    for node, neighbors in graph.items():
        G.add_node(node, label=courses.get(node, {}).get('course_name', f'Course {node}'))
        for neighbor in neighbors:
            G.add_edge(node, neighbor)

    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=0.15, iterations=50) # Fruchterman-Reingold force-directed algorithm
    
    node_labels = nx.get_node_attributes(G, 'label')
    
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1000)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.6)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
    
    plt.title("Exam Conflict Graph")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    logger.info("Conflict graph visualization generated.")

def plot_gantt_chart(exam_slots: List, timeslots: Dict, courses: Dict, instructors: Dict, classrooms: Dict, filename: str = "gantt_chart.png"):
    """
    Visualizes the scheduled exams as a Gantt chart.

    Args:
        exam_slots: List of ExamSlot objects
        timeslots: Dictionary of timeslot details
        courses: Dictionary of course details
        instructors: Dictionary of instructor details
        classrooms: Dictionary of classroom details
        filename: Name of the file to save the plot
    """
    logger.info(f"Generating Gantt chart visualization: {filename}")
    if not exam_slots:
        logger.warning("No exam slots to plot for Gantt chart.")
        return

    # Prepare data for Gantt chart
    gantt_data = []
    for slot in exam_slots:
        timeslot_info = timeslots.get(slot.timeslot_id)
        course_info = courses.get(slot.course_id)
        instructor_info = instructors.get(slot.instructor_id)
        classroom_info = classrooms.get(slot.classroom_id)

        if not all([timeslot_info, course_info, instructor_info, classroom_info]):
            logger.warning(f"Missing data for exam slot {slot.course_id}, skipping.")
            continue

        start_time_str = timeslot_info['start_time']
        end_time_str = timeslot_info['end_time']
        day = timeslot_info['day']

        # Convert to datetime objects for plotting
        # Assuming all exams are on the same 'day' conceptually for time plotting,
        # or we need a way to differentiate days on the X-axis.
        # For simplicity, let's treat time within a day.
        # If multiple days, we'd need to create distinct time ranges for each day.
        start_dt = datetime.strptime(start_time_str, '%H:%M')
        end_dt = datetime.strptime(end_time_str, '%H:%M')
        duration_minutes = (end_dt - start_dt).total_seconds() / 60

        gantt_data.append({
            'course_name': course_info['course_name'],
            'classroom_name': f"{classroom_info['building_name']}-{classroom_info['room_number']}",
            'instructor_name': f"{instructor_info['first_name']} {instructor_info['last_name']}",
            'day': day,
            'start_time': start_dt,
            'duration': duration_minutes
        })
    
    if not gantt_data:
        logger.warning("No valid data points for Gantt chart after filtering.")
        return

    df = pd.DataFrame(gantt_data)
    df['end_time'] = df['start_time'] + pd.to_timedelta(df['duration'], unit='m')

    # Sort by day and then by start time for better visualization
    df['day_order'] = df['day'].astype('category').cat.set_categories(
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], ordered=True
    )
    df = df.sort_values(by=['day_order', 'start_time'])

    # Create distinct "resource" categories (e.g., Classroom-Day)
    df['resource_id'] = df['classroom_name'] + " (" + df['day'] + ")"
    unique_resources = df['resource_id'].unique()
    resource_map = {res: i for i, res in enumerate(unique_resources)}
    df['resource_num'] = df['resource_id'].map(resource_map)

    plt.figure(figsize=(15, 0.5 * len(unique_resources) + 2)) # Dynamic height
    
    # Plotting horizontal bars
    for i, row in df.iterrows():
        # Convert start_time to minutes from midnight for X-axis
        start_minutes = row['start_time'].hour * 60 + row['start_time'].minute
        plt.barh(y=row['resource_num'], width=row['duration'], left=start_minutes, 
                 height=0.6, align='center', color='skyblue', edgecolor='black')
        
        # Add course name label
        plt.text(start_minutes + row['duration'] / 2, row['resource_num'], 
                 row['course_name'], ha='center', va='center', color='black', fontsize=8)

    # Set Y-axis ticks and labels
    plt.yticks(list(resource_map.values()), list(resource_map.keys()), fontsize=9)
    plt.xlabel("Time of Day (Minutes from Midnight)", fontsize=12)
    plt.title("Exam Schedule Gantt Chart (by Classroom and Day)", fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Set X-axis to represent time in HH:MM format
    # Max minutes can be 24*60 = 1440.
    # We can set major ticks at every 60 minutes (1 hour) and format them.
    max_minutes = max(df['end_time'].apply(lambda x: x.hour * 60 + x.minute)) if not df.empty else 1440
    min_minutes = min(df['start_time'].apply(lambda x: x.hour * 60 + x.minute)) if not df.empty else 0
    
    # Extend limits a bit for better visibility
    plt.xlim(max(0, min_minutes - 30), min(1440, max_minutes + 30))

    tick_interval = 60 # Every hour
    start_tick = (min_minutes // tick_interval) * tick_interval
    end_tick = ((max_minutes + tick_interval - 1) // tick_interval) * tick_interval

    x_ticks = range(start_tick, end_tick + tick_interval, tick_interval)
    x_labels = [f"{(m // 60):02d}:{(m % 60):02d}" for m in x_ticks]
    
    plt.xticks(x_ticks, x_labels, rotation=45, ha='right', fontsize=10)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    logger.info("Gantt chart visualization generated.")


def plot_student_load_heatmap(exam_slots: List, timeslots: Dict, filename: str = "student_load_heatmap.png"):
    """
    Visualizes student load across timeslots as a heatmap.

    Args:
        exam_slots: List of ExamSlot objects
        timeslots: Dictionary of timeslot details
        filename: Name of the file to save the plot
    """
    logger.info(f"Generating student load heatmap visualization: {filename}")
    if not exam_slots:
        logger.warning("No exam slots to plot for student load heatmap.")
        return

    # Aggregate student count per timeslot
    timeslot_student_counts = {}
    for slot in exam_slots:
        timeslot_id = slot.timeslot_id
        if timeslot_id not in timeslot_student_counts:
            timeslot_student_counts[timeslot_id] = 0
        timeslot_student_counts[timeslot_id] += len(slot.students)
    
    # Prepare data for heatmap
    heatmap_data = []
    for ts_id, student_count in timeslot_student_counts.items():
        timeslot_info = timeslots.get(ts_id)
        if timeslot_info:
            heatmap_data.append({
                'day': timeslot_info['day'],
                'start_time': timeslot_info['start_time'],
                'student_count': student_count
            })
    
    if not heatmap_data:
        logger.warning("No valid data points for student load heatmap after filtering.")
        return

    df = pd.DataFrame(heatmap_data)

    # Create time categories or convert to sortable format if needed
    df['time_label'] = df['start_time'].apply(lambda x: datetime.strptime(x, '%H:%M').strftime('%H:%M'))
    
    # Pivot table for heatmap (Day vs. Time)
    # Ensure all days and timeslots are represented for a consistent grid
    all_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    all_times = sorted(list(set(df['time_label']))) # Get all unique start times

    # Create a full grid dataframe
    full_index = pd.MultiIndex.from_product([all_days, all_times], names=['day', 'time_label'])
    full_df = pd.DataFrame(index=full_index).reset_index()
    full_df = full_df.merge(df, on=['day', 'time_label'], how='left').fillna(0)
    
    # Sort days for consistent heatmap order
    full_df['day'] = pd.Categorical(full_df['day'], categories=all_days, ordered=True)
    full_df = full_df.sort_values(['day', 'time_label'])

    pivot_table = full_df.pivot_table(index='day', columns='time_label', values='student_count')
    
    # Ensure columns (time_labels) are sorted chronologically
    time_columns_sorted = sorted(pivot_table.columns, key=lambda x: datetime.strptime(x, '%H:%M'))
    pivot_table = pivot_table[time_columns_sorted]

    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap="YlGnBu", linewidths=.5, linecolor='gray',
                cbar_kws={'label': 'Total Students Examined'})
    
    plt.title("Student Load Heatmap by Day and Timeslot", fontsize=16)
    plt.xlabel("Timeslot Start Time", fontsize=12)
    plt.ylabel("Day of Week", fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    logger.info("Student load heatmap visualization generated.")