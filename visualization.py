"""Compact Visualisation Module"""

import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import pandas as pd
from typing import Dict, List, Set
from datetime import datetime

def plot_conflict_graph(graph: Dict[int, Set[int]], courses: Dict, filename: str = "conflict_graph.png"):
    """ Visualise conflict graph"""
    G = nx.Graph()
    for node, neighbors in graph.items():
        G.add_node(node, label=courses.get(node, {}).get('course_name', f'Course {node}'))
        G.add_edges_from((node, neighbor) for neighbor in neighbors)
    
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=0.15, iterations=50)
    labels = nx.get_node_attributes(G, 'label')
    
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1000)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.6)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    
    plt.title("Exam Conflict Graph")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_gantt_chart(exam_slots: List, timeslots: Dict, courses: Dict, 
                    instructors: Dict, classrooms: Dict, filename: str = "gantt_chart.png"):
    """Visualize exam schedule as Gantt chart"""
    if not exam_slots:
        return
    
    # Prepare data
    data = []
    for slot in exam_slots:
        ts_info = timeslots.get(slot.timeslot_id)
        course_info = courses.get(slot.course_id)
        classroom_info = classrooms.get(slot.classroom_id)
        
        if all([ts_info, course_info, classroom_info]):
            start_dt = datetime.strptime(ts_info['start_time'], '%H:%M')
            end_dt = datetime.strptime(ts_info['end_time'], '%H:%M')
            duration = (end_dt - start_dt).total_seconds() / 60
            
            data.append({
                'course_name': course_info['course_name'],
                'classroom': f"{classroom_info['building_name']}-{classroom_info['room_number']}",
                'day': ts_info['day'],
                'start_minutes': start_dt.hour * 60 + start_dt.minute,
                'duration': duration
            })
    
    if not data:
        return
    
    df = pd.DataFrame(data)
    df['resource'] = df['classroom'] + " (" + df['day'] + ")"
    unique_resources = df['resource'].unique()
    resource_map = {res: i for i, res in enumerate(unique_resources)}
    df['y_pos'] = df['resource'].map(resource_map)
    
    plt.figure(figsize=(15, len(unique_resources) * 0.5 + 2))
    
    for _, row in df.iterrows():
        plt.barh(y=row['y_pos'], width=row['duration'], left=row['start_minutes'], 
                height=0.6, color='skyblue', edgecolor='black')
        plt.text(row['start_minutes'] + row['duration']/2, row['y_pos'], 
                row['course_name'], ha='center', va='center', fontsize=8)
    
    plt.yticks(list(resource_map.values()), list(resource_map.keys()))
    plt.xlabel("Time (Minutes from Midnight)")
    plt.title("Exam Schedule Gantt Chart")
    plt.grid(axis='x', alpha=0.3)
    
    # Format x-axis as time
    start_min = df['start_minutes'].min()
    end_min = (df['start_minutes'] + df['duration']).max()
    x_ticks = range(int(start_min//60)*60, int(end_min//60)*60 + 120, 60)
    x_labels = [f"{m//60:02d}:{m%60:02d}" for m in x_ticks]
    plt.xticks(x_ticks, x_labels, rotation=45)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_student_load_heatmap(exam_slots: List, timeslots: Dict, filename: str = "student_load_heatmap.png"):
    """Visualize student load as heatmap"""
    if not exam_slots:
        return
    
    # Aggregate data
    data = []
    for slot in exam_slots:
        ts_info = timeslots.get(slot.timeslot_id)
        if ts_info:
            data.append({
                'day': ts_info['day'],
                'start_time': ts_info['start_time'],
                'student_count': len(slot.students)
            })
    
    if not data:
        return
    
    df = pd.DataFrame(data)
    
    # Create pivot table
    all_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    all_times = sorted(df['start_time'].unique(), key=lambda x: datetime.strptime(x, '%H:%M'))
    
    # Ensure all combinations exist
    full_index = pd.MultiIndex.from_product([all_days, all_times], names=['day', 'start_time'])
    full_df = pd.DataFrame(index=full_index).reset_index()
    full_df = full_df.merge(df, on=['day', 'start_time'], how='left').fillna(0)
    
    pivot = full_df.pivot_table(index='day', columns='start_time', values='student_count')
    pivot = pivot.reindex(all_days)[all_times]  # Ensure proper ordering
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu", linewidths=0.5,
                cbar_kws={'label': 'Total Students'})
    
    plt.title("Student Load Heatmap")
    plt.xlabel("Start Time")
    plt.ylabel("Day")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
