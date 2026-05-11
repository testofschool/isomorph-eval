"""
Isomorph-Eval: Core Data Structures
====================================
Defines the mathematical objects from Phase 1:
  - ReasoningNode, ReasoningGraph: the G in (S, G, V)
  - EntityBinding, MutationSpec: what the mutator swaps
  - IsomorphicItem: the full (S, G, V) tuple with provenance
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple, Union
)
import hashlib
import json


# ============================================================================
# Operation Taxonomy T
# ============================================================================
# This is the finite set of operation types τ(n) from Definition 2.
# Granularity is critical: too coarse → false isomorphs pass;
# too fine → no mutations possible.

class OpType(Enum):
    """
    Operation taxonomy T for mathematical reasoning.
    Each member defines an atomic cognitive operation.
    Domain-specific taxonomies inherit from this pattern.
    """
    # Arithmetic primitives
    ASSIGN       = "assign"         # x = literal
    ADD          = "add"            # x = a + b
    SUBTRACT     = "subtract"       # x = a - b
    MULTIPLY     = "multiply"       # x = a * b
    DIVIDE       = "divide"         # x = a / b
    MODULO       = "modulo"         # x = a % b
    EXPONENTIATE = "exponentiate"   # x = a ^ b

    # Comparison / branching
    COMPARE_GT   = "compare_gt"     # x = (a > b)
    COMPARE_LT   = "compare_lt"
    COMPARE_EQ   = "compare_eq"

    # Aggregation
    SUM_OVER     = "sum_over"       # x = Σ a_i
    PRODUCT_OVER = "product_over"   # x = Π a_i
    COUNT        = "count"          # x = |S|
    AVERAGE      = "average"        # x = mean(a_i)

    # Logical (for logic-domain extension)
    MODUS_PONENS      = "modus_ponens"
    MODUS_TOLLENS     = "modus_tollens"
    DISJUNCTIVE_SYLL  = "disjunctive_syllogism"
    UNIVERSAL_INST    = "universal_instantiation"
    NEGATION          = "negation"
    CONJUNCTION       = "conjunction"
    DISJUNCTION       = "disjunction"


# ============================================================================
# Complexity Weight φ
# ============================================================================

@dataclass(frozen=True)
class ComplexityWeight:
    """
    The φ(n) from Definition 2: quantitative load at each node.

    Two nodes must have matching ComplexityWeight for τ-isomorphism.
    This prevents mutations like: "16 eggs" → "1,000,000 eggs"
    which changes digit count and thus arithmetic difficulty.
    """
    n_operands: int              # how many inputs feed this node
    digit_class: int             # max digits of any operand (1-digit, 2-digit, etc.)
    nesting_depth: int = 0       # depth in nested sub-expressions
    involves_fraction: bool = False
    involves_negative: bool = False
    loop_bound_class: int = 0    # 0 = no loop, 1 = small (≤10), 2 = medium, etc.

    def is_compatible(self, other: ComplexityWeight) -> bool:
        """Check if two weights are within tolerance for isomorphism."""
        return (
            self.n_operands == other.n_operands
            and self.digit_class == other.digit_class
            and self.nesting_depth == other.nesting_depth
            and self.involves_fraction == other.involves_fraction
            and self.involves_negative == other.involves_negative
            and self.loop_bound_class == other.loop_bound_class
        )


# ============================================================================
# Reasoning Graph Components
# ============================================================================

@dataclass
class ReasoningNode:
    """
    A single computation node in the reasoning graph G.

    Each node represents one cognitive step: an operation τ(n)
    applied to its operands, producing a result.
    """
    node_id: str                          # unique identifier within graph
    op_type: OpType                       # τ(n): operation taxonomy label
    complexity: ComplexityWeight           # φ(n): quantitative load
    operand_refs: List[str] = field(default_factory=list)
        # references to other node_ids or leaf_ids
    result_symbol: str = ""               # symbolic name: "remaining_eggs"
    concrete_value: Optional[float] = None  # the numeric answer at this node
    metadata: Dict[str, Any] = field(default_factory=dict)
        # domain-specific annotations (units, entity role, etc.)

    @property
    def is_leaf(self) -> bool:
        """Leaf nodes are ASSIGN operations with no upstream dependencies."""
        return self.op_type == OpType.ASSIGN and len(self.operand_refs) == 0


@dataclass
class EntityBinding:
    """
    Maps a semantic role to a concrete surface entity.

    Example: role="agent" → entity="Janet"
             role="produced_item" → entity="eggs"
             role="venue" → entity="farmers' market"

    During mutation, we swap the entity but preserve the role.
    """
    role: str                   # abstract semantic role
    entity: str                 # concrete surface string
    category: str = ""          # ontological category: "person", "animal", etc.
    constraints: Dict[str, Any] = field(default_factory=dict)
        # e.g., {"gender": "female", "plural": True}


@dataclass
class UnitBinding:
    """
    Maps a quantity role to a concrete unit.

    Example: role="price" → unit="dollars", symbol="$", position="prefix"
    """
    role: str
    unit: str
    symbol: str = ""
    position: str = "suffix"    # "prefix" for $, "suffix" for kg


@dataclass
class ReasoningGraph:
    """
    The complete reasoning graph G = (N, E, τ, φ) from Definition 1.

    This is the core invariant of τ-isomorphism:
    two items are isomorphic iff their ReasoningGraphs are isomorphic
    under the constraints of Definition 2.
    """
    nodes: Dict[str, ReasoningNode]           # N: node_id → node
    edges: Set[Tuple[str, str]]               # E: directed (source, target)
    entity_bindings: List[EntityBinding] = field(default_factory=list)
    unit_bindings: List[UnitBinding] = field(default_factory=list)
    leaf_values: Dict[str, float] = field(default_factory=dict)
        # leaf_id → concrete numeric value (for ASSIGN nodes)
    answer: Optional[float] = None            # final answer
    answer_node_id: str = ""                  # which node produces the answer

    # ---- Structural accessors ----

    @property
    def topology(self) -> Tuple[int, FrozenSet[Tuple[int, int]]]:
        """Abstract topology: (n_nodes, edge_set_by_index)."""
        node_ids = sorted(self.nodes.keys())
        idx_map = {nid: i for i, nid in enumerate(node_ids)}
        abstract_edges = frozenset(
            (idx_map[s], idx_map[t]) for s, t in self.edges
        )
        return (len(node_ids), abstract_edges)

    def topological_order(self) -> List[str]:
        """Return node_ids in dependency-respecting order (Kahn's algorithm)."""
        in_degree = {nid: 0 for nid in self.nodes}
        adjacency = {nid: [] for nid in self.nodes}
        for src, tgt in self.edges:
            adjacency[src].append(tgt)
            in_degree[tgt] += 1

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        order = []
        while queue:
            queue.sort()  # deterministic tie-breaking
            node = queue.pop(0)
            order.append(node)
            for neighbor in adjacency[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        return order

    def depth(self) -> int:
        """Longest path in the DAG (number of sequential reasoning steps)."""
        topo = self.topological_order()
        dist = {nid: 0 for nid in self.nodes}
        for nid in topo:
            for src, tgt in self.edges:
                if src == nid:
                    dist[tgt] = max(dist[tgt], dist[src] + 1)
        return max(dist.values()) if dist else 0

    def op_type_sequence(self) -> List[OpType]:
        """Operation types in topological order — the τ-signature."""
        return [self.nodes[nid].op_type for nid in self.topological_order()]

    def complexity_sequence(self) -> List[ComplexityWeight]:
        """Complexity weights in topological order — the φ-signature."""
        return [self.nodes[nid].complexity for nid in self.topological_order()]

    def structural_fingerprint(self) -> str:
        """
        A hash that is identical for τ-isomorphic graphs.
        Used for fast pre-screening before full VF2 verification.
        """
        sig = {
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
            "depth": self.depth(),
            "op_types": [op.value for op in self.op_type_sequence()],
            "complexities": [
                (c.n_operands, c.digit_class, c.nesting_depth,
                 c.involves_fraction, c.involves_negative)
                for c in self.complexity_sequence()
            ],
        }
        return hashlib.sha256(json.dumps(sig, sort_keys=True).encode()).hexdigest()[:16]


# ============================================================================
# Verification Function V
# ============================================================================

@dataclass
class VerificationFunction:
    """
    The V in (S, G, V): determines if an answer is correct.

    For math problems: V(answer) = (answer == expected_value)
    For logic problems: V(answer) = (answer in valid_conclusions)
    For code problems: V(output) = all(test_case(output) for test_case in suite)
    """
    expected_answer: Any
    answer_type: str = "numeric"   # "numeric", "categorical", "set", "code"
    tolerance: float = 1e-6        # for floating-point comparison
    test_cases: List[Dict] = field(default_factory=list)  # for code domain

    def verify(self, candidate_answer: Any) -> bool:
        if self.answer_type == "numeric":
            try:
                return abs(float(candidate_answer) - float(self.expected_answer)) < self.tolerance
            except (ValueError, TypeError):
                return False
        elif self.answer_type == "categorical":
            return str(candidate_answer).strip().lower() == str(self.expected_answer).strip().lower()
        return False


# ============================================================================
# The Complete Isomorphic Item
# ============================================================================

@dataclass
class IsomorphicItem:
    """
    The full (S, G, V) tuple with provenance tracking.

    An item knows its parent (the original it was derived from),
    its generation method, and its verification status.
    """
    item_id: str
    surface_text: str                       # S: what the model sees
    reasoning_graph: ReasoningGraph         # G: the structural skeleton
    verification: VerificationFunction      # V: correctness checker

    # Provenance
    parent_id: Optional[str] = None         # original item this was derived from
    generation_method: str = "original"     # "original", "graph_mutation", "llm_surface"
    mutation_seed: Optional[int] = None
    is_verified_isomorph: bool = False      # True after Stage A + B pass

    # Contamination metadata
    surface_hash: str = ""                  # for deduplication against known corpora
    novelty_score: float = 0.0             # 1 - max_similarity_to_known_text

    def __post_init__(self):
        if not self.surface_hash:
            self.surface_hash = hashlib.sha256(
                self.surface_text.encode()
            ).hexdigest()[:16]


# ============================================================================
# Mutation Specification
# ============================================================================

@dataclass
class MutationSpec:
    """
    Specifies what to change during graph mutation.

    The mutator uses this to know which dimensions are free
    (entities, values) and which are locked (topology, op_types).
    """
    # Entity swaps: role → new_entity
    entity_swaps: Dict[str, str] = field(default_factory=dict)

    # Value mutations: leaf_id → new_value
    value_mutations: Dict[str, float] = field(default_factory=dict)

    # Unit swaps: role → new_unit
    unit_swaps: Dict[str, str] = field(default_factory=dict)

    # Context theme (for surface generation)
    context_theme: str = ""   # e.g., "space_exploration", "marine_biology"

    # Constraints
    preserve_digit_class: bool = True
    preserve_sign: bool = True
    preserve_magnitude_order: bool = False  # stricter: keep same order of magnitude

    # Random seed for reproducibility
    seed: int = 42
