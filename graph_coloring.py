"""Compact Graph Coloring Module for Exam Scheduling"""

import random
import math
import time
from typing import Dict, List, Set, Optional
from dataclasses import dataclass
from enum import Enum

class ColoringStrategy(Enum):
    GREEDY_LARGEST_FIRST = "greedy_largest_first"
    WELSH_POWELL = "welsh_powell"
    DSATUR = "dsatur"
    SIMULATED_ANNEALING = "simulated_annealing"

@dataclass
class ColoringResult:
    coloring: Dict[int, int]
    num_colors: int
    is_valid: bool
    execution_time: float
    strategy_used: ColoringStrategy

class GraphColoring:
    def __init__(self, graph: Dict[int, Set[int]]):
        self.graph = graph
        self.nodes = list(graph.keys())
    
    def color_graph(self, strategy: ColoringStrategy = ColoringStrategy.DSATUR) -> ColoringResult:
        start_time = time.time()
        
        algorithms = {
            ColoringStrategy.GREEDY_LARGEST_FIRST: self._greedy_largest_first,
            ColoringStrategy.WELSH_POWELL: self._welsh_powell,
            ColoringStrategy.DSATUR: self._dsatur,
            ColoringStrategy.SIMULATED_ANNEALING: self._simulated_annealing
        }
        
        coloring = algorithms[strategy]()
        
        return ColoringResult(
            coloring=coloring,
            num_colors=len(set(coloring.values())),
            is_valid=self._validate_coloring(coloring),
            execution_time=time.time() - start_time,
            strategy_used=strategy
        )
    
    def _greedy_largest_first(self) -> Dict[int, int]:
        # Sort by degree (descending)
        sorted_nodes = sorted(self.nodes, key=lambda n: len(self.graph[n]), reverse=True)
        return self._greedy_color(sorted_nodes)
    
    def _greedy_color(self, node_order: List[int]) -> Dict[int, int]:
        coloring = {}
        for node in node_order:
            used_colors = {coloring[neighbor] for neighbor in self.graph[node] if neighbor in coloring}
            color = 0
            while color in used_colors:
                color += 1
            coloring[node] = color
        return coloring
    
    def _welsh_powell(self) -> Dict[int, int]:
        sorted_nodes = sorted(self.nodes, key=lambda n: len(self.graph[n]), reverse=True)
        coloring = {}
        color = 0
        
        while len(coloring) < len(self.nodes):
            color_nodes = []
            for node in sorted_nodes:
                if node in coloring:
                    continue
                if all(neighbor not in coloring or coloring[neighbor] != color 
                       for neighbor in self.graph[node]):
                    if all(node not in self.graph[colored_node] for colored_node in color_nodes):
                        color_nodes.append(node)
            
            for node in color_nodes:
                coloring[node] = color
            color += 1
        return coloring
    
    def _dsatur(self) -> Dict[int, int]:
        coloring = {}
        # Start with highest degree node
        first_node = max(self.nodes, key=lambda n: len(self.graph[n]))
        coloring[first_node] = 0
        uncolored = set(self.nodes) - {first_node}
        
        while uncolored:
            best_node = max(uncolored, key=lambda n: (
                len({coloring[neighbor] for neighbor in self.graph[n] if neighbor in coloring}),
                len(self.graph[n] & uncolored)
            ))
            
            used_colors = {coloring[neighbor] for neighbor in self.graph[best_node] if neighbor in coloring}
            color = 0
            while color in used_colors:
                color += 1
            
            coloring[best_node] = color
            uncolored.remove(best_node)
        
        return coloring
    
    def _simulated_annealing(self) -> Dict[int, int]:
        current = self._greedy_largest_first()
        best = current.copy()
        temperature = 100.0
        
        for _ in range(1000):
            if temperature < 0.1:
                break
                
            new_solution = current.copy()
            node = random.choice(self.nodes)
            
            # Try different color
            used_colors = {new_solution[neighbor] for neighbor in self.graph[node] 
                          if neighbor in new_solution}
            available = [c for c in range(len(set(current.values())) + 1) if c not in used_colors]
            
            if available:
                new_solution[node] = random.choice(available)
                
                current_colors = len(set(current.values()))
                new_colors = len(set(new_solution.values()))
                
                if new_colors < current_colors or random.random() < math.exp(-(new_colors - current_colors) / temperature):
                    current = new_solution
                    if new_colors < len(set(best.values())):
                        best = new_solution.copy()
            
            temperature *= 0.95
        
        return best
    
    def _validate_coloring(self, coloring: Dict[int, int]) -> bool:
        return all(coloring[node] != coloring[neighbor] 
                  for node in coloring for neighbor in self.graph[node] if neighbor in coloring)

def compare_algorithms(graph: Dict[int, Set[int]]) -> Dict[ColoringStrategy, ColoringResult]:
    gc = GraphColoring(graph)
    return {strategy: gc.color_graph(strategy) for strategy in ColoringStrategy}
