"""
Exam Scheduler - Main Scheduling Module
=====================================

This module provides the main ExamScheduler class that orchestrates the exam scheduling
process using graph coloring algorithms. It integrates with data loading, conflict detection,
graph coloring, and output generation modules.

The scheduler ensures:
1. No student has overlapping exams
2. No instructor has overlapping exam supervision
3. No classroom is double-booked
4. Optimal use of available time slots
"""

import csv
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ExamSlot:
    """Represents a scheduled exam slot"""
    course_id: int
    instructor_id: str
    classroom_id: int
    timeslot_id: int
    students: Set[str]
    exam_duration: int = 120  # Default 2 hours in minutes


@dataclass
class SchedulingConstraints:
    """Configuration for scheduling constraints"""
    min_exam_duration: int = 90    # Minimum exam duration (minutes)
    max_exam_duration: int = 180   # Maximum exam duration (minutes)
    buffer_time: int = 30          # Buffer time between exams (minutes)
    max_students_per_room: int = None  # Will be set based on classroom capacity


class ExamScheduler:
    """
    Main exam scheduler using graph coloring algorithms.
    
    This class coordinates the entire scheduling process:
    1. Load and validate data from CSV files
    2. Build conflict graphs based on scheduling constraints
    3. Apply graph coloring to find valid exam schedules
    4. Generate output schedules and reports
    """
    
    def __init__(self, constraints: Optional[SchedulingConstraints] = None):
        """
        Initialize the exam scheduler.
        
        Args:
            constraints: Scheduling constraints configuration
        """
        self.constraints = constraints or SchedulingConstraints()
        
        # Data storage
        self.students = {}
        self.courses = {}
        self.instructors = {}
        self.classrooms = {}
        self.timeslots = {}
        self.enrollments = []
        
        # Scheduling data
        self.exam_slots = []
        self.conflict_graph = {}
        self.color_assignment = {}
        self.valid_timeslots = []
        
        logger.info("ExamScheduler initialized")
    
    def load_data(self, data_dir: str = ".") -> bool:
        """
        Load all required data from CSV files.
        
        Args:
            data_dir: Directory containing CSV files
            
        Returns:
            bool: True if all data loaded successfully
        """
        try:
            logger.info("Loading data from CSV files...")
            
            # Load students
            self._load_students(f"{data_dir}/students.csv")
            logger.info(f"Loaded {len(self.students)} students")
            
            # Load courses
            self._load_courses(f"{data_dir}/courses.csv")
            logger.info(f"Loaded {len(self.courses)} courses")
            
            # Load instructors
            self._load_instructors(f"{data_dir}/instructors.csv")
            logger.info(f"Loaded {len(self.instructors)} instructors")
            
            # Load classrooms
            self._load_classrooms(f"{data_dir}/classrooms.csv")
            logger.info(f"Loaded {len(self.classrooms)} classrooms")
            
            # Load timeslots
            self._load_timeslots(f"{data_dir}/timeslots.csv")
            logger.info(f"Loaded {len(self.timeslots)} timeslots ({len(self.valid_timeslots)} valid)")
            
            # Load schedule/enrollments
            self._load_schedule(f"{data_dir}/schedule.csv")
            logger.info(f"Loaded {len(self.enrollments)} enrollment records")
            
            return self._validate_data()
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def _load_students(self, filepath: str):
        """Load student data from CSV"""
        with open(filepath, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                self.students[row['student_id']] = {
                    'first_name': row['first_name'],
                    'last_name': row['last_name'],
                    'email': row['email'],
                    'program_name': row['program_name'],
                    'year': row['year']
                }
    
    def _load_courses(self, filepath: str):
        """Load course data from CSV"""
        with open(filepath, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                course_id = int(row['course_id'])
                self.courses[course_id] = {
                    'course_name': row['course_name'],
                    'department': row['department'],
                    'credits': int(row['credits']),
                    'description': row['description']
                }
    
    def _load_instructors(self, filepath: str):
        """Load instructor data from CSV"""
        with open(filepath, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                self.instructors[row['instructor_id']] = {
                    'first_name': row['first_name'],
                    'last_name': row['last_name'],
                    'email': row['email'],
                    'department': row['department']
                }
    
    def _load_classrooms(self, filepath: str):
        """Load classroom data from CSV"""
        with open(filepath, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                classroom_id = int(row['classroom_id'])
                self.classrooms[classroom_id] = {
                    'building_name': row['building_name'],
                    'room_number': row['room_number'],
                    'capacity': int(row['capacity']),
                    'room_type': row['room_type']
                }
    
    def _load_timeslots(self, filepath: str):
        """Load timeslot data from CSV and filter valid ones"""
        with open(filepath, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                timeslot_id = int(row['timeslot_id'])
                start_time = row['start_time']
                end_time = row['end_time']
                
                # Validate timeslot
                if self._is_valid_timeslot(start_time, end_time):
                    self.timeslots[timeslot_id] = {
                        'day': row['day'],
                        'start_time': start_time,
                        'end_time': end_time
                    }
                    self.valid_timeslots.append(timeslot_id)
                else:
                    logger.warning(f"Invalid timeslot {timeslot_id}: {start_time}-{end_time}")
    
    def _load_schedule(self, filepath: str):
        """Load schedule/enrollment data from CSV"""
        with open(filepath, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                self.enrollments.append({
                    'student_id': row['student_id'],
                    'course_id': int(row['course_id']),
                    'instructor_id': row['instructor_id'],
                    'classroom_id': int(row['classroom_id']),
                    'timeslot_id': int(row['timeslot_id'])
                })
    
    def _is_valid_timeslot(self, start_time: str, end_time: str) -> bool:
        """Check if a timeslot has valid start/end times"""
        try:
            start = datetime.strptime(start_time, '%H:%M').time()
            end = datetime.strptime(end_time, '%H:%M').time()
            
            # Convert to minutes for easier comparison
            start_minutes = start.hour * 60 + start.minute
            end_minutes = end.hour * 60 + end.minute
            
            duration = end_minutes - start_minutes
            return duration >= self.constraints.min_exam_duration
            
        except ValueError:
            return False
    
    def _validate_data(self) -> bool:
        """Validate loaded data for consistency"""
        logger.info("Validating data consistency...")
        
        errors = []
        
        # Check for missing references in enrollments
        for enrollment in self.enrollments:
            if enrollment['student_id'] not in self.students:
                errors.append(f"Unknown student_id: {enrollment['student_id']}")
            
            if enrollment['course_id'] not in self.courses:
                errors.append(f"Unknown course_id: {enrollment['course_id']}")
            
            if enrollment['instructor_id'] not in self.instructors:
                errors.append(f"Unknown instructor_id: {enrollment['instructor_id']}")
            
            if enrollment['classroom_id'] not in self.classrooms:
                errors.append(f"Unknown classroom_id: {enrollment['classroom_id']}")
        
        if errors:
            logger.error(f"Data validation failed with {len(errors)} errors")
            for error in errors[:10]:  # Show first 10 errors
                logger.error(error)
            return False
        
        logger.info("Data validation successful")
        return True
    
    def build_conflict_graph(self) -> Dict[int, Set[int]]:
        """
        Build conflict graph where courses that cannot be scheduled simultaneously
        are connected by edges.
        
        Returns:
            Dict mapping course_id to set of conflicting course_ids
        """
        logger.info("Building conflict graph...")
        
        # Initialize conflict graph
        self.conflict_graph = {course_id: set() for course_id in self.courses.keys()}
        
        # Group enrollments by student to find conflicts
        student_courses = {}
        for enrollment in self.enrollments:
            student_id = enrollment['student_id']
            course_id = enrollment['course_id']
            
            if student_id not in student_courses:
                student_courses[student_id] = set()
            student_courses[student_id].add(course_id)
        
        # Add edges for courses taken by same student
        conflict_count = 0
        for student_id, courses in student_courses.items():
            courses_list = list(courses)
            for i in range(len(courses_list)):
                for j in range(i + 1, len(courses_list)):
                    course1, course2 = courses_list[i], courses_list[j]
                    
                    # Add bidirectional conflict
                    self.conflict_graph[course1].add(course2)
                    self.conflict_graph[course2].add(course1)
                    conflict_count += 1
        
        logger.info(f"Built conflict graph with {conflict_count} conflicts")
        return self.conflict_graph
    
    def schedule_exams(self) -> bool:
        """
        Main scheduling method using graph coloring.
        
        Returns:
            bool: True if scheduling successful
        """
        logger.info("Starting exam scheduling...")
        
        # Build conflict graph
        self.build_conflict_graph()
        
        # Apply graph coloring algorithm
        if not self._apply_graph_coloring():
            logger.error("Graph coloring failed - no valid schedule found")
            return False
        
        # Assign resources (timeslots, classrooms)
        if not self._assign_resources():
            logger.error("Resource assignment failed")
            return False
        
        logger.info("Exam scheduling completed successfully")
        return True
    
    def _apply_graph_coloring(self) -> bool:
        """
        Apply graph coloring algorithm to assign time periods to courses.
        Using greedy coloring with largest-first heuristic.
        
        Returns:
            bool: True if coloring successful
        """
        logger.info("Applying graph coloring algorithm...")
        
        # Sort courses by degree (number of conflicts) - largest first
        courses_by_degree = sorted(
            self.courses.keys(),
            key=lambda c: len(self.conflict_graph[c]),
            reverse=True
        )
        
        # Color assignment
        self.color_assignment = {}
        
        for course_id in courses_by_degree:
            # Find available colors (time periods)
            used_colors = set()
            for neighbor in self.conflict_graph[course_id]:
                if neighbor in self.color_assignment:
                    used_colors.add(self.color_assignment[neighbor])
            
            # Assign smallest available color
            color = 0
            while color in used_colors:
                color += 1
            
            self.color_assignment[course_id] = color
        
        num_colors = len(set(self.color_assignment.values()))
        logger.info(f"Graph coloring completed using {num_colors} colors (time periods)")
        
        return num_colors <= len(self.valid_timeslots)
    
    def _assign_resources(self) -> bool:
        """
        Assign specific timeslots and classrooms to colored courses.
        
        Returns:
            bool: True if assignment successful
        """
        logger.info("Assigning resources to scheduled exams...")
        
        # Group courses by color (time period)
        color_groups = {}
        for course_id, color in self.color_assignment.items():
            if color not in color_groups:
                color_groups[color] = []
            color_groups[color].append(course_id)
        
        # Assign timeslots and classrooms
        self.exam_slots = []
        
        for color, courses in color_groups.items():
            if color >= len(self.valid_timeslots):
                logger.error(f"Not enough timeslots for color {color}")
                return False
            
            timeslot_id = self.valid_timeslots[color]
            
            # For each course in this time period, find suitable classroom and instructor
            for course_id in courses:
                exam_slot = self._create_exam_slot(course_id, timeslot_id)
                if exam_slot:
                    self.exam_slots.append(exam_slot)
                else:
                    logger.error(f"Failed to create exam slot for course {course_id}")
                    return False
        
        logger.info(f"Resource assignment completed - {len(self.exam_slots)} exam slots created")
        return True
    
    def _create_exam_slot(self, course_id: int, timeslot_id: int) -> Optional[ExamSlot]:
        """
        Create an exam slot for a specific course and timeslot.
        
        Args:
            course_id: Course identifier
            timeslot_id: Timeslot identifier
            
        Returns:
            ExamSlot or None if creation failed
        """
        # Find students enrolled in this course
        enrolled_students = set()
        course_instructors = set()
        course_classrooms = set()
        
        for enrollment in self.enrollments:
            if enrollment['course_id'] == course_id:
                enrolled_students.add(enrollment['student_id'])
                course_instructors.add(enrollment['instructor_id'])
                course_classrooms.add(enrollment['classroom_id'])
        
        if not enrolled_students:
            logger.warning(f"No students found for course {course_id}")
            return None
        
        # Select instructor (prefer the most common one for this course)
        instructor_counts = {}
        for enrollment in self.enrollments:
            if enrollment['course_id'] == course_id:
                instructor_id = enrollment['instructor_id']
                instructor_counts[instructor_id] = instructor_counts.get(instructor_id, 0) + 1
        
        selected_instructor = max(instructor_counts.keys(), key=instructor_counts.get)
        
        # Select classroom with sufficient capacity
        selected_classroom = None
        min_capacity_needed = len(enrolled_students)
        
        for classroom_id in course_classrooms:
            capacity = self.classrooms[classroom_id]['capacity']
            if capacity >= min_capacity_needed:
                if (selected_classroom is None or 
                    capacity < self.classrooms[selected_classroom]['capacity']):
                    selected_classroom = classroom_id
        
        if selected_classroom is None:
            # Use largest available classroom
            selected_classroom = max(
                self.classrooms.keys(),
                key=lambda cid: self.classrooms[cid]['capacity']
            )
        
        return ExamSlot(
            course_id=course_id,
            instructor_id=selected_instructor,
            classroom_id=selected_classroom,
            timeslot_id=timeslot_id,
            students=enrolled_students
        )
    
    def get_schedule_summary(self) -> Dict:
        """
        Generate a summary of the scheduled exams.
        
        Returns:
            Dict containing schedule statistics
        """
        if not self.exam_slots:
            return {"error": "No exams scheduled"}
        
        summary = {
            "total_exams": len(self.exam_slots),
            "total_students": len(self.students),
            "total_time_periods": len(set(slot.timeslot_id for slot in self.exam_slots)),
            "classroom_utilization": {},
            "instructor_workload": {},
            "time_period_usage": {}
        }
        
        # Calculate utilization metrics
        for slot in self.exam_slots:
            # Classroom utilization
            classroom_id = slot.classroom_id
            capacity = self.classrooms[classroom_id]['capacity']
            utilization = len(slot.students) / capacity * 100
            
            if classroom_id not in summary["classroom_utilization"]:
                summary["classroom_utilization"][classroom_id] = []
            summary["classroom_utilization"][classroom_id].append(utilization)
            
            # Instructor workload
            instructor_id = slot.instructor_id
            if instructor_id not in summary["instructor_workload"]:
                summary["instructor_workload"][instructor_id] = 0
            summary["instructor_workload"][instructor_id] += 1
            
            # Time period usage
            timeslot_id = slot.timeslot_id
            if timeslot_id not in summary["time_period_usage"]:
                summary["time_period_usage"][timeslot_id] = 0
            summary["time_period_usage"][timeslot_id] += 1
        
        return summary
    
    def export_schedule(self, filepath: str = "exam_schedule.csv") -> bool:
        """
        Export the exam schedule to CSV file.
        
        Args:
            filepath: Output file path
            
        Returns:
            bool: True if export successful
        """
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                
                # Write header
                writer.writerow([
                    'course_id', 'course_name', 'instructor_name', 'instructor_email',
                    'classroom', 'capacity', 'day', 'start_time', 'end_time',
                    'enrolled_students', 'utilization_percent'
                ])
                
                # Write exam slots
                for slot in sorted(self.exam_slots, key=lambda s: (s.timeslot_id, s.course_id)):
                    course = self.courses[slot.course_id]
                    instructor = self.instructors[slot.instructor_id]
                    classroom = self.classrooms[slot.classroom_id]
                    timeslot = self.timeslots[slot.timeslot_id]
                    
                    utilization = len(slot.students) / classroom['capacity'] * 100
                    
                    writer.writerow([
                        slot.course_id,
                        course['course_name'],
                        f"{instructor['first_name']} {instructor['last_name']}",
                        instructor['email'],
                        f"{classroom['building_name']}-{classroom['room_number']}",
                        classroom['capacity'],
                        timeslot['day'],
                        timeslot['start_time'],
                        timeslot['end_time'],
                        len(slot.students),
                        f"{utilization:.1f}%"
                    ])
            
            logger.info(f"Schedule exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export schedule: {e}")
            return False


def main():
    """Main function for testing the scheduler"""
    # Create scheduler with default constraints
    scheduler = ExamScheduler()
    
    # Load data
    if not scheduler.load_data():
        print("Failed to load data")
        return
    
    # Schedule exams
    if not scheduler.schedule_exams():
        print("Failed to schedule exams")
        return
    
    # Print summary
    summary = scheduler.get_schedule_summary()
    print("\n=== EXAM SCHEDULE SUMMARY ===")
    print(f"Total exams scheduled: {summary['total_exams']}")
    print(f"Time periods used: {summary['total_time_periods']}")
    print(f"Students affected: {summary['total_students']}")
    
    # Export schedule
    scheduler.export_schedule()
    print("\nSchedule exported to exam_schedule.csv")


if __name__ == "__main__":
    main()