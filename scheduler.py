"""Compact Exam Scheduler"""

import csv
from typing import Dict, List, Set, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ExamSlot:
    course_id: int
    instructor_id: str
    classroom_id: int
    timeslot_id: int
    students: Set[str]

@dataclass
class SchedulingConstraints:
    min_exam_duration: int = 90
    max_exam_duration: int = 180
    buffer_time: int = 30

class ExamScheduler:
    def __init__(self, constraints: Optional[SchedulingConstraints] = None):
        self.constraints = constraints or SchedulingConstraints()
        self.students = {}
        self.courses = {}
        self.instructors = {}
        self.classrooms = {}
        self.timeslots = {}
        self.enrollments = []
        self.exam_slots = []
        self.conflict_graph = {}
        self.color_assignment = {}
        self.valid_timeslots = []
    
    def load_data(self, data_dir: str = ".") -> bool:
        try:
            loaders = [
                ("students.csv", self._load_csv, self.students, 'student_id'),
                ("courses.csv", self._load_csv, self.courses, 'course_id'),
                ("instructors.csv", self._load_csv, self.instructors, 'instructor_id'),
                ("classrooms.csv", self._load_csv, self.classrooms, 'classroom_id'),
                ("timeslots.csv", self._load_timeslots, None, None),
                ("schedule.csv", self._load_enrollments, None, None)
            ]
            
            for filename, loader, storage, key_field in loaders:
                if not loader(f"{data_dir}/{filename}", storage, key_field):
                    return False
            return True
        except Exception:
            return False
    
    def _load_csv(self, filepath: str, storage: dict, key_field: str) -> bool:
        with open(filepath, 'r', encoding='utf-8') as file:
            for row in csv.DictReader(file):
                key = int(row[key_field]) if key_field.endswith('_id') and row[key_field].isdigit() else row[key_field]
                storage[key] = row
        return True
    
    def _load_timeslots(self, filepath: str, *args) -> bool:
        with open(filepath, 'r', encoding='utf-8') as file:
            for row in csv.DictReader(file):
                timeslot_id = int(row['timeslot_id'])
                if self._is_valid_timeslot(row['start_time'], row['end_time']):
                    self.timeslots[timeslot_id] = row
                    self.valid_timeslots.append(timeslot_id)
        return True
    
    def _load_enrollments(self, filepath: str, *args) -> bool:
        with open(filepath, 'r', encoding='utf-8') as file:
            self.enrollments = [{**row, 'course_id': int(row['course_id']), 
                               'classroom_id': int(row['classroom_id']), 
                               'timeslot_id': int(row['timeslot_id'])} for row in csv.DictReader(file)]
        return True
    
    def _is_valid_timeslot(self, start_time: str, end_time: str) -> bool:
        try:
            start = datetime.strptime(start_time, '%H:%M')
            end = datetime.strptime(end_time, '%H:%M')
            return (end - start).total_seconds() / 60 >= self.constraints.min_exam_duration
        except ValueError:
            return False
    
    def build_conflict_graph(self) -> Dict[int, Set[int]]:
        self.conflict_graph = {int(course_id): set() for course_id in self.courses.keys()}
        
        # Group by student
        student_courses = {}
        for enrollment in self.enrollments:
            student_id = enrollment['student_id']
            course_id = enrollment['course_id']
            student_courses.setdefault(student_id, set()).add(course_id)
        
        # Add conflicts
        for courses in student_courses.values():
            course_list = list(courses)
            for i in range(len(course_list)):
                for j in range(i + 1, len(course_list)):
                    self.conflict_graph[course_list[i]].add(course_list[j])
                    self.conflict_graph[course_list[j]].add(course_list[i])
        
        return self.conflict_graph
    
    def schedule_exams(self) -> bool:
        self.build_conflict_graph()
        return self._apply_graph_coloring() and self._assign_resources()
    
    def _apply_graph_coloring(self) -> bool:
        # Greedy coloring with largest-first
        courses_by_degree = sorted(self.courses.keys(), 
                                 key=lambda c: len(self.conflict_graph[c]), reverse=True)
        
        for course_id in courses_by_degree:
            used_colors = {self.color_assignment[neighbor] for neighbor in self.conflict_graph[course_id] 
                          if neighbor in self.color_assignment}
            color = 0
            while color in used_colors:
                color += 1
            self.color_assignment[course_id] = color
        
        return len(set(self.color_assignment.values())) <= len(self.valid_timeslots)
    
    def _assign_resources(self) -> bool:
        # Group by color
        color_groups = {}
        for course_id, color in self.color_assignment.items():
            color_groups.setdefault(color, []).append(course_id)
        
        self.exam_slots = []
        for color, courses in color_groups.items():
            if color >= len(self.valid_timeslots):
                return False
            
            timeslot_id = self.valid_timeslots[color]
            for course_id in courses:
                slot = self._create_exam_slot(course_id, timeslot_id)
                if slot:
                    self.exam_slots.append(slot)
                else:
                    return False
        return True
    
    def _create_exam_slot(self, course_id: int, timeslot_id: int) -> Optional[ExamSlot]:
        # Find course data
        enrolled_students = set()
        instructors = {}
        classrooms = {}
        
        for enrollment in self.enrollments:
            if enrollment['course_id'] == course_id:
                enrolled_students.add(enrollment['student_id'])
                instructors[enrollment['instructor_id']] = instructors.get(enrollment['instructor_id'], 0) + 1
                classrooms[enrollment['classroom_id']] = classrooms.get(enrollment['classroom_id'], 0) + 1
        
        if not enrolled_students:
            return None
        
        # Select most common instructor and suitable classroom
        selected_instructor = max(instructors.keys(), key=instructors.get)
        selected_classroom = min([cid for cid in classrooms.keys() 
                                if self.classrooms[str(cid)]['capacity'] >= str(len(enrolled_students))],
                               key=lambda cid: int(self.classrooms[str(cid)]['capacity']), default=max(classrooms.keys()))
        
        return ExamSlot(course_id, selected_instructor, selected_classroom, timeslot_id, enrolled_students)
    
    def export_schedule(self, filepath: str = "exam_schedule.csv") -> bool:
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['course_id', 'course_name', 'instructor_name', 'classroom', 
                               'day', 'start_time', 'end_time', 'enrolled_students'])
                
                for slot in sorted(self.exam_slots, key=lambda s: (s.timeslot_id, s.course_id)):
                    course = self.courses[slot.course_id]
                    instructor = self.instructors[slot.instructor_id]
                    classroom = self.classrooms[slot.classroom_id]
                    timeslot = self.timeslots[slot.timeslot_id]
                    
                    writer.writerow([
                        slot.course_id, course['course_name'],
                        f"{instructor['first_name']} {instructor['last_name']}",
                        f"{classroom['building_name']}-{classroom['room_number']}",
                        timeslot['day'], timeslot['start_time'], timeslot['end_time'],
                        len(slot.students)
                    ])
            return True
        except Exception:
            return False
    
    def get_schedule_summary(self) -> Dict:
        if not self.exam_slots:
            return {"error": "No exams scheduled"}
        
        return {
            "total_exams": len(self.exam_slots),
            "total_students": len(self.students),
            "total_time_periods": len(set(slot.timeslot_id for slot in self.exam_slots))
        }
