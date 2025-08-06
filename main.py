#!/usr/bin/env python3
"""Compact Exam Scheduling System Main Entry Point"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

from scheduler import ExamScheduler, SchedulingConstraints
from graph_coloring import GraphColoring, ColoringStrategy, compare_algorithms
from visualization import plot_conflict_graph, plot_gantt_chart, plot_student_load_heatmap

class ExamSchedulingSystem:
    def __init__(self, data_dir: str = ".", output_dir: str = "output"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.scheduler = None
        self.results = {}
    
    def validate_data_files(self) -> bool:
        required = ['students.csv', 'courses.csv', 'instructors.csv', 
                   'classrooms.csv', 'timeslots.csv', 'schedule.csv']
        return all((self.data_dir / f).exists() for f in required)
    
    def create_scheduler(self, constraints=None) -> bool:
        try:
            self.scheduler = ExamScheduler(constraints)
            return self.scheduler.load_data(str(self.data_dir))
        except Exception:
            return False
    
    def run_scheduling(self, strategy=None) -> bool:
        if not self.scheduler:
            return False
        
        start_time = time.time()
        
        if strategy:
            success = self._run_with_strategy(strategy)
        else:
            success = self.scheduler.schedule_exams()
        
        self.results['execution_time'] = time.time() - start_time
        self.results['success'] = success
        return success
    
    def _run_with_strategy(self, strategy: ColoringStrategy) -> bool:
        conflict_graph = self.scheduler.build_conflict_graph()
        gc = GraphColoring(conflict_graph)
        result = gc.color_graph(strategy)
        
        if not result.is_valid:
            return False
        
        self.scheduler.color_assignment = result.coloring
        return self.scheduler._assign_resources()
    
    def generate_reports(self) -> bool:
        if not self.scheduler or not self.scheduler.exam_slots:
            return False
        
        try:
            # Export schedule
            self.scheduler.export_schedule(str(self.output_dir / "exam_schedule.csv"))
            
            # Summary report
            summary = self.scheduler.get_schedule_summary()
            summary['system_info'] = {
                'execution_time': self.results.get('execution_time', 0),
                'success': self.results.get('success', False)
            }
            
            with open(self.output_dir / "schedule_summary.json", 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Generate visualizations
            if hasattr(self.scheduler, 'conflict_graph') and self.scheduler.conflict_graph:
                plot_conflict_graph(self.scheduler.conflict_graph, self.scheduler.courses,
                                   str(self.output_dir / "conflict_graph.png"))
            
            if self.scheduler.exam_slots:
                plot_gantt_chart(self.scheduler.exam_slots, self.scheduler.timeslots,
                                self.scheduler.courses, self.scheduler.instructors,
                                self.scheduler.classrooms, str(self.output_dir / "gantt_chart.png"))
                
                plot_student_load_heatmap(self.scheduler.exam_slots, self.scheduler.timeslots,
                                         str(self.output_dir / "student_load_heatmap.png"))
            
            return True
        except Exception:
            return False
    
    def benchmark_algorithms(self, strategies=None) -> Dict:
        if not self.scheduler:
            return {}
        
        if strategies is None:
            strategies = list(ColoringStrategy)
        
        conflict_graph = self.scheduler.build_conflict_graph()
        results = {}
        
        for strategy in strategies:
            gc = GraphColoring(conflict_graph)
            result = gc.color_graph(strategy)
            results[strategy.value] = {
                'num_colors': result.num_colors,
                'execution_time': result.execution_time,
                'is_valid': result.is_valid
            }
        
        # Save results
        with open(self.output_dir / "algorithm_benchmark.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        return results

def interactive_mode():
    print("=== Exam Scheduling System - Interactive Mode ===")
    
    data_dir = input("Enter data directory path [current directory]: ").strip() or "."
    system = ExamSchedulingSystem(data_dir)
    
    if not system.validate_data_files():
        print("ERROR: Data validation failed")
        return
    
    # Strategy selection
    strategies = list(ColoringStrategy)
    print("\nAvailable strategies:")
    for i, strategy in enumerate(strategies, 1):
        print(f"  {i}. {strategy.value}")
    print("  0. Use default")
    
    try:
        choice = int(input("\nSelect strategy [0]: ") or "0")
        selected_strategy = None if choice == 0 else strategies[choice - 1]
    except (ValueError, IndexError):
        selected_strategy = None
    
    # Run scheduling
    if not system.create_scheduler():
        print("ERROR: Failed to create scheduler")
        return
    
    print("\nRunning scheduling...")
    if system.run_scheduling(selected_strategy):
        print("SUCCESS: Scheduling completed!")
        system.generate_reports()
        summary = system.scheduler.get_schedule_summary()
        print(f"\nTotal exams: {summary['total_exams']}")
        print(f"Time periods: {summary['total_time_periods']}")
        print(f"Output saved to: {system.output_dir}")
    else:
        print("ERROR: Scheduling failed")

def main():
    parser = argparse.ArgumentParser(description="Exam Scheduling System")
    subparsers = parser.add_subparsers(dest='command')
    
    # Schedule command
    schedule_parser = subparsers.add_parser('schedule')
    schedule_parser.add_argument('--data-dir', default='datasets')
    schedule_parser.add_argument('--output-dir', default='output')
    schedule_parser.add_argument('--strategy', choices=[s.value for s in ColoringStrategy])
    
    # Validate command
    validate_parser = subparsers.add_parser('validate')
    validate_parser.add_argument('--data-dir', default='.')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark')
    benchmark_parser.add_argument('--data-dir', default='.')
    benchmark_parser.add_argument('--output-dir', default='output')
    
    # Interactive command
    subparsers.add_parser('interactive')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'interactive':
        interactive_mode()
    
    elif args.command == 'validate':
        system = ExamSchedulingSystem(args.data_dir)
        print("✓ Valid" if system.validate_data_files() else "✗ Invalid")
    
    elif args.command == 'schedule':
        constraints = SchedulingConstraints()
        system = ExamSchedulingSystem(args.data_dir, args.output_dir)
        
        if not system.validate_data_files() or not system.create_scheduler(constraints):
            print("ERROR: Setup failed")
            return
        
        strategy = ColoringStrategy(args.strategy) if args.strategy else None
        
        if system.run_scheduling(strategy):
            system.generate_reports()
            print(f"SUCCESS: Results saved to {system.output_dir}")
        else:
            print("ERROR: Scheduling failed")
