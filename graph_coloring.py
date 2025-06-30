"""
Graph Coloring Module for Exam Scheduling
=========================================

This module implements various graph coloring algorithms specifically designed for
exam scheduling problems. It provides multiple coloring strategies and optimization
techniques to minimize the number of time slots required while ensuring no conflicts.

Algorithms implemented:
1. Greedy Coloring (Largest First, Smallest Last, Random)
2. Welsh-Powell Algorithm
3. DSATUR (Degree of Saturation) Algorithm
4. Backtracking with Constraint Propagation
5. Simulated Annealing for optimization
"""

import random
import math
import time
from typing import Dict, List, Set, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ColoringStrategy(Enum):
    """Available graph coloring strategies"""
    GREEDY_LARGEST_FIRST = "greedy_largest_first"
    GREEDY_SMALLEST_LAST = "greedy_smallest_last"
    GREEDY_RANDOM = "greedy_random"
    WELSH_POWELL = "welsh_powell"
    DSATUR = "dsatur"
    BACKTRACKING = "backtracking"
    SIMULATED_ANNEALING = "simulated_annealing"


@dataclass
class ColoringResult:
    """Result of a graph coloring operation"""
    coloring: Dict[int, int]  # node_id -> color
    num_colors: int
    is_valid: bool
    execution_time: float
    strategy_used: ColoringStrategy
    iterations: int = 0
    conflicts: int = 0


@dataclass
class ColoringParameters:
    """Parameters for coloring algorithms"""
    max_colors: Optional[int] = None
    max_iterations: int = 1000
    random_seed: Optional[int] = None
    temperature_initial: float = 100.0
    temperature_final: float = 0.1
    cooling_rate: float = 0.95
    early_termination: bool = True
    timeout_seconds: float = 300.0


class GraphColoring:
    """
    Main graph coloring class implementing multiple algorithms for exam scheduling.
    
    This class provides various graph coloring strategies optimized for exam scheduling
    constraints, including conflict minimization and resource optimization.
    """
    
    def __init__(self, graph: Dict[int, Set[int]], parameters: Optional[ColoringParameters] = None):
        """
        Initialize the graph coloring solver.
        
        Args:
            graph: Adjacency list representation {node_id: {neighbor_ids}}
            parameters: Algorithm parameters and constraints
        """
        self.graph = graph
        self.nodes = list(graph.keys())
        self.parameters = parameters or ColoringParameters()
        
        if self.parameters.random_seed is not None:
            random.seed(self.parameters.random_seed)
        
        logger.info(f"Initialized graph with {len(self.nodes)} nodes and "
                   f"{sum(len(neighbors) for neighbors in graph.values()) // 2} edges")
    
    def color_graph(self, strategy: ColoringStrategy = ColoringStrategy.DSATUR) -> ColoringResult:
        """
        Color the graph using the specified strategy.
        
        Args:
            strategy: Coloring algorithm to use
            
        Returns:
            ColoringResult with coloring assignment and metadata
        """
        start_time = time.time()
        
        try:
            if strategy == ColoringStrategy.GREEDY_LARGEST_FIRST:
                result = self._greedy_coloring(self._largest_first_ordering)
            elif strategy == ColoringStrategy.GREEDY_SMALLEST_LAST:
                result = self._greedy_coloring(self._smallest_last_ordering)
            elif strategy == ColoringStrategy.GREEDY_RANDOM:
                result = self._greedy_coloring(self._random_ordering)
            elif strategy == ColoringStrategy.WELSH_POWELL:
                result = self._welsh_powell_coloring()
            elif strategy == ColoringStrategy.DSATUR:
                result = self._dsatur_coloring()
            elif strategy == ColoringStrategy.BACKTRACKING:
                result = self._backtracking_coloring()
            elif strategy == ColoringStrategy.SIMULATED_ANNEALING:
                result = self._simulated_annealing_coloring()
            else:
                raise ValueError(f"Unknown coloring strategy: {strategy}")
            
            result.execution_time = time.time() - start_time
            result.strategy_used = strategy
            result.is_valid = self._validate_coloring(result.coloring)
            result.conflicts = self._count_conflicts(result.coloring)
            
            logger.info(f"Coloring completed: {result.num_colors} colors, "
                       f"{result.execution_time:.3f}s, valid={result.is_valid}")
            
            return result
            
        except Exception as e:
            logger.error(f"Coloring failed with strategy {strategy}: {e}")
            return ColoringResult(
                coloring={},
                num_colors=0,
                is_valid=False,
                execution_time=time.time() - start_time,
                strategy_used=strategy,
                conflicts=float('inf')
            )
    
    def find_optimal_coloring(self, strategies: Optional[List[ColoringStrategy]] = None) -> ColoringResult:
        """
        Try multiple strategies and return the best result.
        
        Args:
            strategies: List of strategies to try (default: all strategies)
            
        Returns:
            Best ColoringResult found
        """
        if strategies is None:
            strategies = [
                ColoringStrategy.DSATUR,
                ColoringStrategy.WELSH_POWELL,
                ColoringStrategy.GREEDY_LARGEST_FIRST,
                ColoringStrategy.SIMULATED_ANNEALING,
                ColoringStrategy.BACKTRACKING
            ]
        
        best_result = None
        
        for strategy in strategies:
            logger.info(f"Trying strategy: {strategy.value}")
            result = self.color_graph(strategy)
            
            if result.is_valid and (best_result is None or 
                                   result.num_colors < best_result.num_colors):
                best_result = result
                logger.info(f"New best result: {result.num_colors} colors")
                
                # Early termination if we find a very good solution
                if self.parameters.early_termination and result.num_colors <= 3:
                    break
        
        return best_result or ColoringResult(
            coloring={}, num_colors=0, is_valid=False, 
            execution_time=0, strategy_used=strategies[0]
        )
    
    def _greedy_coloring(self, ordering_func: Callable) -> ColoringResult:
        """
        Generic greedy coloring with custom node ordering.
        
        Args:
            ordering_func: Function to determine node processing order
            
        Returns:
            ColoringResult
        """
        node_order = ordering_func()
        coloring = {}
        
        for node in node_order:
            # Find colors used by neighbors
            used_colors = set()
            for neighbor in self.graph[node]:
                if neighbor in coloring:
                    used_colors.add(coloring[neighbor])
            
            # Assign smallest available color
            color = 0
            while color in used_colors:
                color += 1
                
                # Check max colors constraint
                if self.parameters.max_colors and color >= self.parameters.max_colors:
                    break
            
            coloring[node] = color
        
        return ColoringResult(
            coloring=coloring,
            num_colors=len(set(coloring.values())),
            is_valid=True,
            execution_time=0,
            strategy_used=ColoringStrategy.GREEDY_LARGEST_FIRST  # Will be overridden
        )
    
    def _largest_first_ordering(self) -> List[int]:
        """Order nodes by degree (descending)"""
        return sorted(self.nodes, key=lambda n: len(self.graph[n]), reverse=True)
    
    def _smallest_last_ordering(self) -> List[int]:
        """Order nodes by smallest-last (reverse elimination)"""
        remaining_nodes = set(self.nodes)
        graph_copy = {n: self.graph[n].copy() for n in self.nodes}
        order = []
        
        while remaining_nodes:
            # Find node with minimum degree in remaining graph
            min_node = min(remaining_nodes, 
                          key=lambda n: len(graph_copy[n] & remaining_nodes))
            
            order.append(min_node)
            remaining_nodes.remove(min_node)
            
            # Remove from neighbors' adjacency lists
            for neighbor in graph_copy[min_node]:
                if neighbor in graph_copy:
                    graph_copy[neighbor].discard(min_node)
        
        return list(reversed(order))
    
    def _random_ordering(self) -> List[int]:
        """Random node ordering"""
        order = self.nodes.copy()
        random.shuffle(order)
        return order
    
    def _welsh_powell_coloring(self) -> ColoringResult:
        """
        Welsh-Powell algorithm: sort by degree then greedy color.
        
        Returns:
            ColoringResult
        """
        # Sort nodes by degree (descending)
        sorted_nodes = sorted(self.nodes, key=lambda n: len(self.graph[n]), reverse=True)
        
        coloring = {}
        color = 0
        
        while len(coloring) < len(self.nodes):
            # Find all uncolored nodes that can use current color
            color_nodes = []
            
            for node in sorted_nodes:
                if node in coloring:
                    continue
                
                # Check if node can use current color
                can_use_color = True
                for neighbor in self.graph[node]:
                    if neighbor in coloring and coloring[neighbor] == color:
                        can_use_color = False
                        break
                
                if can_use_color:
                    # Additional check: ensure no conflicts with nodes already assigned this color
                    conflict_free = True
                    for colored_node in color_nodes:
                        if node in self.graph[colored_node]:
                            conflict_free = False
                            break
                    
                    if conflict_free:
                        color_nodes.append(node)
            
            # Assign current color to selected nodes
            for node in color_nodes:
                coloring[node] = color
            
            color += 1
            
            # Safety check
            if color > len(self.nodes):
                break
        
        return ColoringResult(
            coloring=coloring,
            num_colors=len(set(coloring.values())),
            is_valid=True,
            execution_time=0,
            strategy_used=ColoringStrategy.WELSH_POWELL
        )
    
    def _dsatur_coloring(self) -> ColoringResult:
        """
        DSATUR algorithm: prioritize nodes with highest saturation degree.
        
        Returns:
            ColoringResult
        """
        coloring = {}
        uncolored = set(self.nodes)
        
        # Start with highest degree node
        first_node = max(self.nodes, key=lambda n: len(self.graph[n]))
        coloring[first_node] = 0
        uncolored.remove(first_node)
        
        while uncolored:
            # Calculate saturation degree for each uncolored node
            best_node = None
            best_saturation = -1
            best_degree = -1
            
            for node in uncolored:
                # Saturation degree: number of different colors in neighborhood
                neighbor_colors = set()
                for neighbor in self.graph[node]:
                    if neighbor in coloring:
                        neighbor_colors.add(coloring[neighbor])
                
                saturation = len(neighbor_colors)
                degree = len(self.graph[node] & uncolored)  # Degree in uncolored subgraph
                
                # Choose node with highest saturation, break ties by degree
                if (saturation > best_saturation or 
                    (saturation == best_saturation and degree > best_degree)):
                    best_node = node
                    best_saturation = saturation
                    best_degree = degree
            
            # Color the selected node
            if best_node is not None:
                used_colors = set()
                for neighbor in self.graph[best_node]:
                    if neighbor in coloring:
                        used_colors.add(coloring[neighbor])
                
                color = 0
                while color in used_colors:
                    color += 1
                
                coloring[best_node] = color
                uncolored.remove(best_node)
        
        return ColoringResult(
            coloring=coloring,
            num_colors=len(set(coloring.values())),
            is_valid=True,
            execution_time=0,
            strategy_used=ColoringStrategy.DSATUR
        )
    
    def _backtracking_coloring(self) -> ColoringResult:
        """
        Backtracking algorithm with constraint propagation.
        
        Returns:
            ColoringResult
        """
        if self.parameters.max_colors is None:
            # Estimate upper bound using greedy coloring
            greedy_result = self._greedy_coloring(self._largest_first_ordering)
            max_colors = greedy_result.num_colors
        else:
            max_colors = self.parameters.max_colors
        
        # Try different numbers of colors, starting from a lower bound
        lower_bound = max(1, self._calculate_clique_lower_bound())
        
        for num_colors in range(lower_bound, max_colors + 1):
            coloring = {}
            if self._backtrack_recursive(coloring, 0, num_colors):
                return ColoringResult(
                    coloring=coloring,
                    num_colors=num_colors,
                    is_valid=True,
                    execution_time=0,
                    strategy_used=ColoringStrategy.BACKTRACKING
                )
        
        # Fallback to greedy if backtracking fails
        return self._greedy_coloring(self._largest_first_ordering)
    
    def _backtrack_recursive(self, coloring: Dict[int, int], node_idx: int, max_colors: int) -> bool:
        """
        Recursive backtracking helper.
        
        Args:
            coloring: Current partial coloring
            node_idx: Index of current node to color
            max_colors: Maximum number of colors allowed
            
        Returns:
            bool: True if valid coloring found
        """
        if node_idx >= len(self.nodes):
            return True  # All nodes colored successfully
        
        node = self.nodes[node_idx]
        
        # Try each color
        for color in range(max_colors):
            if self._is_color_valid(node, color, coloring):
                coloring[node] = color
                
                if self._backtrack_recursive(coloring, node_idx + 1, max_colors):
                    return True
                
                del coloring[node]  # Backtrack
        
        return False
    
    def _is_color_valid(self, node: int, color: int, coloring: Dict[int, int]) -> bool:
        """Check if assigning a color to a node is valid"""
        for neighbor in self.graph[node]:
            if neighbor in coloring and coloring[neighbor] == color:
                return False
        return True
    
    def _calculate_clique_lower_bound(self) -> int:
        """Estimate lower bound based on maximum clique size"""
        # Simple greedy approach to find a large clique
        max_clique_size = 1
        
        for node in self.nodes:
            clique = {node}
            candidates = self.graph[node].copy()
            
            while candidates:
                # Find node in candidates with most connections to current clique
                best_candidate = max(candidates, 
                                   key=lambda c: len(self.graph[c] & clique))
                
                # Check if this candidate is connected to all nodes in clique
                if self.graph[best_candidate] >= clique:
                    clique.add(best_candidate)
                    candidates &= self.graph[best_candidate]
                else:
                    candidates.remove(best_candidate)
            
            max_clique_size = max(max_clique_size, len(clique))
        
        return max_clique_size
    
    def _simulated_annealing_coloring(self) -> ColoringResult:
        """
        Simulated annealing optimization for graph coloring.
        
        Returns:
            ColoringResult
        """
        # Start with greedy solution
        current_solution = self._greedy_coloring(self._largest_first_ordering)
        current_colors = current_solution.num_colors
        best_solution = current_solution.coloring.copy()
        best_colors = current_colors
        
        temperature = self.parameters.temperature_initial
        iterations = 0
        
        while (temperature > self.parameters.temperature_final and 
               iterations < self.parameters.max_iterations):
            
            # Generate neighbor solution by recoloring a random node
            new_coloring = current_solution.coloring.copy()
            node = random.choice(self.nodes)
            
            # Try a different color
            available_colors = list(range(current_colors + 1))
            used_colors = {new_coloring[neighbor] for neighbor in self.graph[node] 
                          if neighbor in new_coloring}
            available_colors = [c for c in available_colors if c not in used_colors]
            
            if available_colors:
                new_color = random.choice(available_colors)
                new_coloring[node] = new_color
                
                # Calculate number of colors in new solution
                new_colors = len(set(new_coloring.values()))
                
                # Accept or reject based on simulated annealing criteria
                delta = new_colors - current_colors
                
                if delta < 0 or random.random() < math.exp(-delta / temperature):
                    current_solution.coloring = new_coloring
                    current_colors = new_colors
                    
                    if new_colors < best_colors:
                        best_solution = new_coloring.copy()
                        best_colors = new_colors
            
            temperature *= self.parameters.cooling_rate
            iterations += 1
        
        return ColoringResult(
            coloring=best_solution,
            num_colors=best_colors,
            is_valid=True,
            execution_time=0,
            strategy_used=ColoringStrategy.SIMULATED_ANNEALING,
            iterations=iterations
        )
    
    def _validate_coloring(self, coloring: Dict[int, int]) -> bool:
        """
        Validate that a coloring is conflict-free.
        
        Args:
            coloring: Node to color mapping
            
        Returns:
            bool: True if coloring is valid
        """
        for node, color in coloring.items():
            for neighbor in self.graph[node]:
                if neighbor in coloring and coloring[neighbor] == color:
                    return False
        return True
    
    def _count_conflicts(self, coloring: Dict[int, int]) -> int:
        """
        Count the number of conflicts in a coloring.
        
        Args:
            coloring: Node to color mapping
            
        Returns:
            int: Number of conflicts (adjacent nodes with same color)
        """
        conflicts = 0
        for node, color in coloring.items():
            for neighbor in self.graph[node]:
                if neighbor in coloring and coloring[neighbor] == color and node < neighbor:
                    conflicts += 1
        return conflicts
    
    def get_coloring_statistics(self, coloring: Dict[int, int]) -> Dict:
        """
        Generate detailed statistics about a coloring.
        
        Args:
            coloring: Node to color mapping
            
        Returns:
            Dict: Statistics about the coloring
        """
        if not coloring:
            return {"error": "Empty coloring"}
        
        color_counts = {}
        for color in coloring.values():
            color_counts[color] = color_counts.get(color, 0) + 1
        
        return {
            "num_colors": len(set(coloring.values())),
            "num_nodes": len(coloring),
            "is_valid": self._validate_coloring(coloring),
            "conflicts": self._count_conflicts(coloring),
            "color_distribution": color_counts,
            "max_color_size": max(color_counts.values()),
            "min_color_size": min(color_counts.values()),
            "avg_color_size": sum(color_counts.values()) / len(color_counts),
            "balance_ratio": min(color_counts.values()) / max(color_counts.values())
        }
    
    def visualize_coloring_text(self, coloring: Dict[int, int]) -> str:
        """
        Generate a text-based visualization of the coloring.
        
        Args:
            coloring: Node to color mapping
            
        Returns:
            str: Text representation of the coloring
        """
        if not coloring:
            return "No coloring available"
        
        # Group nodes by color
        color_groups = {}
        for node, color in coloring.items():
            if color not in color_groups:
                color_groups[color] = []
            color_groups[color].append(node)
        
        # Generate visualization
        lines = [f"Graph Coloring Result ({len(color_groups)} colors):"]
        lines.append("=" * 50)
        
        for color in sorted(color_groups.keys()):
            nodes = sorted(color_groups[color])
            lines.append(f"Color {color}: {nodes} ({len(nodes)} nodes)")
        
        lines.append("=" * 50)
        stats = self.get_coloring_statistics(coloring)
        lines.append(f"Valid: {stats['is_valid']}, Conflicts: {stats['conflicts']}")
        lines.append(f"Balance ratio: {stats['balance_ratio']:.3f}")
        
        return "\n".join(lines)


def compare_algorithms(graph: Dict[int, Set[int]], 
                      strategies: Optional[List[ColoringStrategy]] = None) -> Dict[ColoringStrategy, ColoringResult]:
    """
    Compare multiple coloring algorithms on the same graph.
    
    Args:
        graph: Graph to color
        strategies: List of strategies to compare
        
    Returns:
        Dict mapping strategies to their results
    """
    if strategies is None:
        strategies = list(ColoringStrategy)
    
    coloring_engine = GraphColoring(graph)
    results = {}
    
    print(f"Comparing {len(strategies)} algorithms on graph with {len(graph)} nodes...")
    print("-" * 80)
    
    for strategy in strategies:
        print(f"Running {strategy.value}...", end=" ")
        result = coloring_engine.color_graph(strategy)
        results[strategy] = result
        
        print(f"{result.num_colors} colors, {result.execution_time:.3f}s, "
              f"valid={result.is_valid}")
    
    print("-" * 80)
    
    # Find best result
    valid_results = [(s, r) for s, r in results.items() if r.is_valid]
    if valid_results:
        best_strategy, best_result = min(valid_results, key=lambda x: x[1].num_colors)
        print(f"Best result: {best_strategy.value} with {best_result.num_colors} colors")
    
    return results


def main():
    """Test the graph coloring algorithms"""
    # Create a test graph (Petersen graph - known to need 3 colors)
    test_graph = {
        0: {1, 4, 5},
        1: {0, 2, 6},
        2: {1, 3, 7},
        3: {2, 4, 8},
        4: {0, 3, 9},
        5: {0, 7, 8},
        6: {1, 8, 9},
        7: {2, 5, 9},
        8: {3, 5, 6},
        9: {4, 6, 7}
    }
    
    print("Testing Graph Coloring Algorithms")
    print("=" * 40)
    
    # Test individual algorithm
    coloring_engine = GraphColoring(test_graph)
    result = coloring_engine.color_graph(ColoringStrategy.DSATUR)
    
    print("DSATUR Result:")
    print(coloring_engine.visualize_coloring_text(result.coloring))
    print()
    
    # Compare algorithms
    strategies = [
        ColoringStrategy.GREEDY_LARGEST_FIRST,
        ColoringStrategy.WELSH_POWELL,
        ColoringStrategy.DSATUR
    ]
    
    results = compare_algorithms(test_graph, strategies)


if __name__ == "__main__":
    main()