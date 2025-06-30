#!/usr/bin/env python3
"""
Exam Scheduling System - Main Entry Point
========================================

This is the main entry point for the exam scheduling system that uses graph coloring
algorithms to optimize exam scheduling while avoiding conflicts.

Features:
- Command-line interface for different operations
- Integration with graph coloring algorithms
- Data validation and error handling
- Multiple output formats
- Performance benchmarking
- Interactive and batch modes

Usage:
    python main.py schedule [options]
    python main.py validate [options]
    python main.py benchmark [options]
    python main.py interactive
"""

import argparse
import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Import our modules
from scheduler import ExamScheduler, SchedulingConstraints, ExamSlot
from graph_coloring import GraphColoring, ColoringStrategy, ColoringParameters, compare_algorithms
from visualization import plot_conflict_graph, plot_gantt_chart, plot_student_load_heatmap # NEW: Added import

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('exam_scheduler.log')
    ]
)
logger = logging.getLogger(__name__)


class ExamSchedulingSystem:
    """
    Main system class that orchestrates the exam scheduling process.
    
    This class provides a high-level interface for:
    - Loading and validating data
    - Running scheduling algorithms
    - Generating reports and outputs
    - Performance analysis
    """
    
    def __init__(self, data_dir: str = ".", output_dir: str = "output"):
        """
        Initialize the scheduling system.
        
        Args:
            data_dir: Directory containing input CSV files
            output_dir: Directory for output files
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.scheduler = None
        self.results = {}
        
        logger.info(f"Initialized ExamSchedulingSystem")
        logger.info(f"Data directory: {self.data_dir.absolute()}")
        logger.info(f"Output directory: {self.output_dir.absolute()}")
    
    def validate_data_files(self) -> bool:
        """
        Validate that all required CSV files exist and are readable.
        
        Returns:
            bool: True if all files are valid
        """
        required_files = [
            'students.csv',
            'courses.csv', 
            'instructors.csv',
            'classrooms.csv',
            'timeslots.csv',
            'schedule.csv'
        ]
        
        logger.info("Validating data files...")
        
        missing_files = []
        for filename in required_files:
            filepath = self.data_dir / filename
            if not filepath.exists():
                missing_files.append(filename)
            elif not filepath.is_file():
                missing_files.append(f"{filename} (not a file)")
        
        if missing_files:
            logger.error(f"Missing required files: {missing_files}")
            return False
        
        logger.info("All required data files found")
        return True
    
    def create_scheduler(self, constraints: Optional[SchedulingConstraints] = None) -> bool:
        """
        Create and initialize the exam scheduler.
        
        Args:
            constraints: Custom scheduling constraints
            
        Returns:
            bool: True if scheduler created successfully
        """
        try:
            self.scheduler = ExamScheduler(constraints)
            
            if not self.scheduler.load_data(str(self.data_dir)):
                logger.error("Failed to load data into scheduler")
                return False
            
            logger.info("Scheduler created and data loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create scheduler: {e}")
            return False
    
    def run_scheduling(self, strategy: Optional[ColoringStrategy] = None) -> bool:
        """
        Run the exam scheduling process.
        
        Args:
            strategy: Specific graph coloring strategy to use
            
        Returns:
            bool: True if scheduling successful
        """
        if not self.scheduler:
            logger.error("Scheduler not initialized")
            return False
        
        logger.info("Starting exam scheduling process...")
        start_time = time.time()
        
        try:
            # If specific strategy requested, integrate it with the scheduler
            if strategy:
                logger.info(f"Using graph coloring strategy: {strategy.value}")
                success = self._run_scheduling_with_strategy(strategy)
            else:
                # Use default scheduler approach
                success = self.scheduler.schedule_exams()
            
            execution_time = time.time() - start_time
            
            if success:
                logger.info(f"Scheduling completed successfully in {execution_time:.2f}s")
                self.results['execution_time'] = execution_time
                self.results['success'] = True
                return True
            else:
                logger.error(f"Scheduling failed after {execution_time:.2f}s")
                self.results['success'] = False
                return False
                
        except Exception as e:
            logger.error(f"Scheduling process failed: {e}")
            self.results['success'] = False
            return False
    
    def _run_scheduling_with_strategy(self, strategy: ColoringStrategy) -> bool:
        """
        Run scheduling with a specific graph coloring strategy.
        
        Args:
            strategy: Graph coloring strategy to use
            
        Returns:
            bool: True if successful
        """
        # Build conflict graph
        conflict_graph = self.scheduler.build_conflict_graph()
        
        # Apply graph coloring with specified strategy
        coloring_params = ColoringParameters(
            max_iterations=1000,
            random_seed=42,
            early_termination=True
        )
        
        graph_coloring = GraphColoring(conflict_graph, coloring_params)
        coloring_result = graph_coloring.color_graph(strategy)
        
        if not coloring_result.is_valid:
            logger.error("Graph coloring failed to find valid solution")
            return False
        
        # Override scheduler's color assignment
        self.scheduler.color_assignment = coloring_result.coloring
        
        # Continue with resource assignment
        return self.scheduler._assign_resources()
    
    def generate_reports(self) -> bool:
        """
        Generate comprehensive reports about the scheduling results.
        
        Returns:
            bool: True if reports generated successfully
        """
        if not self.scheduler or not self.scheduler.exam_slots:
            logger.error("No scheduling results to report")
            return False
        
        try:
            logger.info("Generating reports...")
            
            # Export main schedule
            schedule_file = self.output_dir / "exam_schedule.csv"
            self.scheduler.export_schedule(str(schedule_file))
            
            # Generate summary report
            self._generate_summary_report()
            
            # Generate conflict analysis
            self._generate_conflict_analysis()
            
            # Generate resource utilization report
            self._generate_utilization_report()

            # --- NEW: Generate Visualizations ---
            logger.info("Generating visualizations...")
            
            # Conflict Graph
            if self.scheduler.conflict_graph and self.scheduler.courses:
                plot_conflict_graph(self.scheduler.conflict_graph, self.scheduler.courses, 
                                    filename=str(self.output_dir / "conflict_graph.png"))
            else:
                logger.warning("Cannot generate conflict graph: missing conflict data or courses.")

            # Gantt Chart
            if self.scheduler.exam_slots and self.scheduler.timeslots and \
               self.scheduler.courses and self.scheduler.instructors and self.scheduler.classrooms:
                plot_gantt_chart(self.scheduler.exam_slots, self.scheduler.timeslots, 
                                 self.scheduler.courses, self.scheduler.instructors, 
                                 self.scheduler.classrooms, 
                                 filename=str(self.output_dir / "gantt_chart.png"))
            else:
                logger.warning("Cannot generate Gantt chart: missing exam slots or related data.")

            # Student Load Heatmap
            if self.scheduler.exam_slots and self.scheduler.timeslots:
                plot_student_load_heatmap(self.scheduler.exam_slots, self.scheduler.timeslots, 
                                          filename=str(self.output_dir / "student_load_heatmap.png"))
            else:
                logger.warning("Cannot generate student load heatmap: missing exam slots or timeslots.")
            # --- END NEW ---
            
            logger.info("All reports and visualizations generated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate reports: {e}")
            return False
    
    def _generate_summary_report(self):
        """Generate a summary report in JSON format"""
        summary = self.scheduler.get_schedule_summary()
        summary['system_info'] = {
            'data_directory': str(self.data_dir),
            'output_directory': str(self.output_dir),
            'execution_time': self.results.get('execution_time', 0),
            'success': self.results.get('success', False)
        }
        
        summary_file = self.output_dir / "schedule_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Summary report saved to {summary_file}")
    
    def _generate_conflict_analysis(self):
        """Generate conflict analysis report"""
        if not self.scheduler.conflict_graph:
            return
        
        analysis = {
            'total_courses': len(self.scheduler.courses),
            'total_conflicts': sum(len(conflicts) for conflicts in self.scheduler.conflict_graph.values()) // 2,
            'conflict_distribution': {},
            'most_conflicted_courses': []
        }
        
        # Analyze conflict distribution
        conflict_counts = {}
        for course_id, conflicts in self.scheduler.conflict_graph.items():
            count = len(conflicts)
            conflict_counts[count] = conflict_counts.get(count, 0) + 1
        
        analysis['conflict_distribution'] = conflict_counts
        
        # Find most conflicted courses
        course_conflicts = [(course_id, len(conflicts)) for course_id, conflicts in self.scheduler.conflict_graph.items()]
        course_conflicts.sort(key=lambda x: x[1], reverse=True)
        
        for course_id, conflict_count in course_conflicts[:10]:
            course_name = self.scheduler.courses[course_id]['course_name']
            analysis['most_conflicted_courses'].append({
                'course_id': course_id,
                'course_name': course_name,
                'conflicts': conflict_count
            })
        
        conflict_file = self.output_dir / "conflict_analysis.json"
        with open(conflict_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Conflict analysis saved to {conflict_file}")
    
    def _generate_utilization_report(self):
        """Generate resource utilization report"""
        utilization = {
            'timeslots': {},
            'classrooms': {},
            'instructors': {}
        }
        
        # Analyze timeslot usage
        for slot in self.scheduler.exam_slots:
            ts_id = slot.timeslot_id
            if ts_id not in utilization['timeslots']:
                timeslot = self.scheduler.timeslots[ts_id]
                utilization['timeslots'][ts_id] = {
                    'day': timeslot['day'],
                    'time': f"{timeslot['start_time']}-{timeslot['end_time']}",
                    'exams_scheduled': 0,
                    'total_students': 0
                }
            
            utilization['timeslots'][ts_id]['exams_scheduled'] += 1
            utilization['timeslots'][ts_id]['total_students'] += len(slot.students)
        
        # Analyze classroom usage
        for slot in self.scheduler.exam_slots:
            room_id = slot.classroom_id
            if room_id not in utilization['classrooms']:
                classroom = self.scheduler.classrooms[room_id]
                utilization['classrooms'][room_id] = {
                    'location': f"{classroom['building_name']}-{classroom['room_number']}",
                    'capacity': classroom['capacity'],
                    'usage_sessions': 0,
                    'total_occupancy': 0,
                    'utilization_rates': []
                }
            
            room_util = utilization['classrooms'][room_id]
            room_util['usage_sessions'] += 1
            room_util['total_occupancy'] += len(slot.students)
            
            capacity = self.scheduler.classrooms[room_id]['capacity']
            rate = len(slot.students) / capacity * 100
            room_util['utilization_rates'].append(rate)
        
        # Calculate average utilization for classrooms
        for room_data in utilization['classrooms'].values():
            if room_data['utilization_rates']:
                room_data['avg_utilization'] = sum(room_data['utilization_rates']) / len(room_data['utilization_rates'])
        
        # Analyze instructor workload
        for slot in self.scheduler.exam_slots:
            inst_id = slot.instructor_id
            if inst_id not in utilization['instructors']:
                instructor = self.scheduler.instructors[inst_id]
                utilization['instructors'][inst_id] = {
                    'name': f"{instructor['first_name']} {instructor['last_name']}",
                    'department': instructor['department'],
                    'exams_supervised': 0,
                    'students_supervised': 0
                }
            
            utilization['instructors'][inst_id]['exams_supervised'] += 1
            utilization['instructors'][inst_id]['students_supervised'] += len(slot.students)
        
        util_file = self.output_dir / "utilization_report.json"
        with open(util_file, 'w') as f:
            json.dump(utilization, f, indent=2, default=str)
        
        logger.info(f"Utilization report saved to {util_file}")
    
    def benchmark_algorithms(self, strategies: Optional[List[ColoringStrategy]] = None) -> Dict:
        """
        Benchmark different graph coloring algorithms.
        
        Args:
            strategies: List of strategies to benchmark
            
        Returns:
            Dict: Benchmark results
        """
        if not self.scheduler:
            logger.error("Scheduler not initialized")
            return {}
        
        if strategies is None:
            strategies = [
                ColoringStrategy.GREEDY_LARGEST_FIRST,
                ColoringStrategy.WELSH_POWELL,
                ColoringStrategy.DSATUR,
                ColoringStrategy.SIMULATED_ANNEALING
            ]
        
        logger.info(f"Benchmarking {len(strategies)} algorithms...")
        
        # Build conflict graph
        conflict_graph = self.scheduler.build_conflict_graph()
        
        # Run benchmark
        results = compare_algorithms(conflict_graph, strategies)
        
        # Format results for output
        benchmark_results = {}
        for strategy, result in results.items():
            benchmark_results[strategy.value] = {
                'num_colors': result.num_colors,
                'execution_time': result.execution_time,
                'is_valid': result.is_valid,
                'conflicts': result.conflicts,
                'iterations': getattr(result, 'iterations', 0)
            }
        
        # Save benchmark results
        benchmark_file = self.output_dir / "algorithm_benchmark.json"
        with open(benchmark_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        logger.info(f"Benchmark results saved to {benchmark_file}")
        return benchmark_results


def create_sample_constraints() -> SchedulingConstraints:
    """Create sample scheduling constraints"""
    return SchedulingConstraints(
        min_exam_duration=90,
        max_exam_duration=180,
        buffer_time=15,
        max_students_per_room=None
    )


def interactive_mode():
    """Run the system in interactive mode"""
    print("=== Exam Scheduling System - Interactive Mode ===")
    print()
    
    # Get data directory
    data_dir = input("Enter data directory path [current directory]: ").strip()
    if not data_dir:
        data_dir = "."
    
    # Initialize system
    system = ExamSchedulingSystem(data_dir)
    
    if not system.validate_data_files():
        print("ERROR: Data validation failed. Please check your CSV files.")
        return
    
    # Get scheduling options
    print("\nAvailable graph coloring strategies:")
    strategies = list(ColoringStrategy)
    for i, strategy in enumerate(strategies, 1):
        print(f"  {i}. {strategy.value}")
    
    print("  0. Use default scheduler approach")
    
    try:
        choice = int(input("\nSelect strategy [0]: ") or "0")
        selected_strategy = None if choice == 0 else strategies[choice - 1]
    except (ValueError, IndexError):
        print("Invalid choice, using default approach")
        selected_strategy = None
    
    # Create scheduler
    constraints = create_sample_constraints()
    if not system.create_scheduler(constraints):
        print("ERROR: Failed to create scheduler")
        return
    
    # Run scheduling
    print("\nRunning scheduling algorithm...")
    if system.run_scheduling(selected_strategy):
        print("SUCCESS: Scheduling completed!")
        
        # Generate reports
        print("Generating reports...")
        system.generate_reports()
        
        # Show summary
        summary = system.scheduler.get_schedule_summary()
        print(f"\n=== SCHEDULE SUMMARY ===")
        print(f"Total exams scheduled: {summary['total_exams']}")
        print(f"Time periods used: {summary['total_time_periods']}")
        print(f"Students affected: {summary['total_students']}")
        print(f"Output files saved to: {system.output_dir}")
        
    else:
        print("ERROR: Scheduling failed")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Exam Scheduling System using Graph Coloring Algorithms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s schedule --data-dir ./data --strategy dsatur
  %(prog)s validate --data-dir ./data
  %(prog)s benchmark --data-dir ./data --output-dir ./results
  %(prog)s interactive
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Schedule command
    schedule_parser = subparsers.add_parser('schedule', help='Run exam scheduling')
    schedule_parser.add_argument('--data-dir', default='datasets', help='Data directory path')
    schedule_parser.add_argument('--output-dir', default='output', help='Output directory path')
    schedule_parser.add_argument('--strategy', choices=[s.value for s in ColoringStrategy],
                                help='Graph coloring strategy to use')
    schedule_parser.add_argument('--min-duration', type=int, default=90,
                                help='Minimum exam duration (minutes)')
    schedule_parser.add_argument('--max-duration', type=int, default=180,
                                help='Maximum exam duration (minutes)')
    schedule_parser.add_argument('--buffer-time', type=int, default=15,
                                help='Buffer time between exams (minutes)')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate data files')
    validate_parser.add_argument('--data-dir', default='.', help='Data directory path')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark algorithms')
    benchmark_parser.add_argument('--data-dir', default='.', help='Data directory path')
    benchmark_parser.add_argument('--output-dir', default='output', help='Output directory path')
    benchmark_parser.add_argument('--strategies', nargs='+', 
                                 choices=[s.value for s in ColoringStrategy],
                                 help='Strategies to benchmark')
    
    # Interactive command
    subparsers.add_parser('interactive', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'interactive':
        interactive_mode()
        
    elif args.command == 'validate':
        system = ExamSchedulingSystem(args.data_dir)
        if system.validate_data_files():
            print("✓ All data files are valid")
        else:
            print("✗ Data validation failed")
            sys.exit(1)
    
    elif args.command == 'schedule':
        # Create constraints
        constraints = SchedulingConstraints(
            min_exam_duration=args.min_duration,
            max_exam_duration=args.max_duration,
            buffer_time=args.buffer_time
        )
        
        # Initialize system
        system = ExamSchedulingSystem(args.data_dir, args.output_dir)
        
        if not system.validate_data_files():
            print("ERROR: Data validation failed")
            sys.exit(1)
        
        if not system.create_scheduler(constraints):
            print("ERROR: Failed to create scheduler")
            sys.exit(1)
        
        # Determine strategy
        strategy = None
        if args.strategy:
            strategy = ColoringStrategy(args.strategy)
        
        # Run scheduling
        if system.run_scheduling(strategy):
            print("SUCCESS: Scheduling completed")
            system.generate_reports()
            print(f"Results saved to: {system.output_dir}")
        else:
            print("ERROR: Scheduling failed")
            sys.exit(1)
    
    elif args.command == 'benchmark':
        system = ExamSchedulingSystem(args.data_dir, args.output_dir)
        
        if not system.validate_data_files():
            print("ERROR: Data validation failed")
            sys.exit(1)
        
        if not system.create_scheduler():
            print("ERROR: Failed to create scheduler")
            sys.exit(1)
        
        # Determine strategies
        strategies = None
        if args.strategies:
            strategies = [ColoringStrategy(s) for s in args.strategies]
        
        results = system.benchmark_algorithms(strategies)
        
        if results:
            print("\n=== BENCHMARK RESULTS ===")
            for strategy, metrics in results.items():
                print(f"{strategy:25} | Colors: {metrics['num_colors']:2d} | "
                      f"Time: {metrics['execution_time']:6.3f}s | "
                      f"Valid: {metrics['is_valid']}")
            print(f"\nDetailed results saved to: {system.output_dir}")
        else:
            print("ERROR: Benchmark failed")
            sys.exit(1)


if __name__ == "__main__":
    main()