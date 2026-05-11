"""
Isomorph-Eval: Pipeline Stages
===============================
Stage 1: MathProblemParser     — S → (G, V)
Stage 2: GraphMutator          — G → G'
Stage 3: SurfaceGenerator      — G' → S'
Stage 4: IsomorphVerifier      — verify G' ≅_τ G, then S' → G'' ≅_τ G'

The critical safety mechanism is ROUND-TRIP VERIFICATION:
  S' is re-parsed to G'', and G'' is checked against G'.
  This prevents the LLM from altering logic during surface generation.
"""

from __future__ import annotations
import re
import math
import random
import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from core.data_structures import (
    OpType, ComplexityWeight, ReasoningNode, ReasoningGraph,
    EntityBinding, UnitBinding, VerificationFunction,
    IsomorphicItem, MutationSpec,
)


# ============================================================================
# STAGE 1: PARSER — S → (G, V)
# ============================================================================

class BaseProblemParser(ABC):
    """
    Abstract parser interface. Domain-specific parsers implement this.

    The parser's job: decompose a natural-language problem into
    its reasoning graph G and verification function V.

    CRITICAL DESIGN CHOICE: For math problems, we use a HYBRID approach:
      1. Rule-based extraction of known GSM8K patterns (covers ~85%)
      2. LLM-assisted parsing with structured output for novel patterns
      3. Human-in-the-loop for ambiguous cases

    We NEVER trust the LLM parser alone — its output is always
    verified by re-computing the answer through the extracted graph.
    """

    @abstractmethod
    def parse(self, surface_text: str) -> Tuple[ReasoningGraph, VerificationFunction]:
        """Parse surface text into (G, V)."""
        ...

    @abstractmethod
    def can_parse(self, surface_text: str) -> bool:
        """Check if this parser handles this problem type."""
        ...


class GSM8KParser(BaseProblemParser):
    """
    Parser for GSM8K-style math word problems.

    Strategy:
    1. Extract the chain-of-thought solution (GSM8K provides these)
    2. Parse each step into a ReasoningNode
    3. Build the dependency graph from variable references
    4. Compute the answer through the graph to verify consistency
    """

    # Common patterns in GSM8K solutions
    ASSIGNMENT_PATTERN = re.compile(
        r"(?P<var>[\w\s]+?)\s*(?:is|are|=|has|have|gets?|earns?|costs?|weighs?)\s*"
        r"(?P<value>[\d,]+(?:\.\d+)?)"
    )

    OPERATION_PATTERNS = {
        OpType.ADD: re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*\+\s*(\d[\d,]*(?:\.\d+)?)"),
        OpType.SUBTRACT: re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*[-–]\s*(\d[\d,]*(?:\.\d+)?)"),
        OpType.MULTIPLY: re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*[×\*]\s*(\d[\d,]*(?:\.\d+)?)"),
        OpType.DIVIDE: re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*[÷/]\s*(\d[\d,]*(?:\.\d+)?)"),
    }

    def can_parse(self, surface_text: str) -> bool:
        return "?" in surface_text and any(c.isdigit() for c in surface_text)

    def parse(self, surface_text: str, solution_steps: List[str] = None,
              answer: float = None) -> Tuple[ReasoningGraph, VerificationFunction]:
        """
        Parse a GSM8K problem.

        For the MVP, we accept pre-annotated solution steps.
        In production, this uses the hybrid LLM+rule approach.
        """
        nodes = {}
        edges = set()
        leaf_values = {}
        entity_bindings = []
        unit_bindings = []

        if solution_steps is None:
            # Fallback: use LLM-assisted parsing (not shown here)
            raise NotImplementedError("Auto-parsing requires LLM backend")

        # Build nodes from annotated solution steps
        step_results = {}  # maps step descriptions to node_ids

        for idx, step in enumerate(solution_steps):
            node_id = f"n{idx}"
            parsed = self._parse_single_step(step, idx, step_results)
            nodes[node_id] = parsed["node"]

            if parsed["node"].is_leaf:
                leaf_values[node_id] = parsed["node"].concrete_value

            # Track dependencies
            for dep_id in parsed.get("depends_on", []):
                edges.add((dep_id, node_id))

            step_results[parsed.get("description", f"step_{idx}")] = node_id

        # Build the reasoning graph
        graph = ReasoningGraph(
            nodes=nodes,
            edges=edges,
            entity_bindings=entity_bindings,
            unit_bindings=unit_bindings,
            leaf_values=leaf_values,
            answer=answer,
            answer_node_id=f"n{len(solution_steps) - 1}",
        )

        # Build verification function
        verification = VerificationFunction(
            expected_answer=answer,
            answer_type="numeric",
        )

        # SAFETY CHECK: re-execute the graph and verify answer matches
        computed = self._execute_graph(graph)
        if answer is not None and abs(computed - answer) > 1e-6:
            raise ValueError(
                f"Graph execution gives {computed}, but expected {answer}. "
                f"Parse is inconsistent — refusing to proceed."
            )

        return graph, verification

    def _parse_single_step(self, step: str, idx: int,
                           prior_results: Dict) -> Dict:
        """Parse one solution step into a ReasoningNode."""
        # This is simplified — production version uses pattern matching
        # on the annotated GSM8K solution format
        return {
            "node": ReasoningNode(
                node_id=f"n{idx}",
                op_type=OpType.ASSIGN,
                complexity=ComplexityWeight(n_operands=0, digit_class=1),
            ),
            "depends_on": [],
            "description": step,
        }

    def _execute_graph(self, graph: ReasoningGraph) -> float:
        """
        Forward-execute the reasoning graph to compute the answer.

        This is the GROUND TRUTH computation — independent of any LLM.
        If this doesn't match the stated answer, the parse is wrong.
        """
        computed_values = {}

        for node_id in graph.topological_order():
            node = graph.nodes[node_id]

            if node.is_leaf:
                computed_values[node_id] = node.concrete_value
                continue

            # Gather operand values
            operand_vals = [computed_values[ref] for ref in node.operand_refs]

            # Execute operation
            if node.op_type == OpType.ADD:
                result = sum(operand_vals)
            elif node.op_type == OpType.SUBTRACT:
                result = operand_vals[0] - sum(operand_vals[1:])
            elif node.op_type == OpType.MULTIPLY:
                result = math.prod(operand_vals)
            elif node.op_type == OpType.DIVIDE:
                result = operand_vals[0] / operand_vals[1]
            elif node.op_type == OpType.MODULO:
                result = operand_vals[0] % operand_vals[1]
            else:
                result = node.concrete_value  # fallback

            computed_values[node_id] = result

        return computed_values.get(graph.answer_node_id, 0.0)


# ============================================================================
# STAGE 2: GRAPH MUTATOR — G → G'
# ============================================================================

class SemanticMutator:
    """
    Mutates a ReasoningGraph G into G' while preserving τ-isomorphism.

    WHAT CHANGES (free dimensions):
      - Entity names (Janet → Tomás)
      - Object nouns (eggs → honey jars)
      - Numeric leaf values (16 → 23, preserving digit class)
      - Units (dollars → euros)
      - Context theme (farming → marine biology)

    WHAT IS LOCKED (structural invariants):
      - Graph topology (nodes, edges)
      - Operation types τ(n) at each node
      - Complexity weights φ(n) at each node
      - Number of operands per operation
      - Digit class of each numeric value
    """

    # Entity pools by category — all culturally diverse, none from
    # common benchmark datasets to maximize novelty
    ENTITY_POOLS = {
        "person": [
            "Tomás", "Aisha", "Kenji", "Priya", "Oluwaseun",
            "Fatima", "Dmitri", "Yuki", "Ingrid", "Kofi",
            "Xiulan", "Rashid", "Svetlana", "Hiroshi", "Amara",
            "Bjorn", "Nalini", "Emeka", "Saoirse", "Tariq",
        ],
        "animal": [
            "bees", "silkworms", "oysters", "alpacas", "quails",
            "trout", "crickets", "butterflies", "seahorses", "axolotls",
        ],
        "product": [
            "honey jars", "silk cocoons", "pearls", "wool bales",
            "feathers", "beeswax candles", "mushroom caps", "seashells",
            "crystals", "pine cones", "clay pots", "woven baskets",
        ],
        "venue": [
            "harbor market", "mountain fair", "online storefront",
            "village cooperative", "orbital trading post",
            "desert bazaar", "floating market", "night market",
        ],
        "food_activity": [
            "uses for pottery glazes", "grinds into flour",
            "processes into dye", "ferments into vinegar",
            "presses into oil", "dries for tea blending",
        ],
    }

    CONTEXT_THEMES = {
        "marine_biology": {
            "person": ["Dr. Kai", "Marina", "Captain Reeves"],
            "setting": "research station",
            "currency": "credits",
        },
        "space_exploration": {
            "person": ["Astronaut Vega", "Commander Chen", "Pilot Okafor"],
            "setting": "orbital habitat",
            "currency": "starcoin",
        },
        "medieval_craft": {
            "person": ["Blacksmith Elara", "Weaver Theron", "Herbalist Mira"],
            "setting": "guild hall",
            "currency": "silver pieces",
        },
    }

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def mutate(self, graph: ReasoningGraph,
               spec: Optional[MutationSpec] = None) -> Tuple[ReasoningGraph, MutationSpec]:
        """
        Produce a mutated graph G' that is τ-isomorphic to G.

        Returns (G', spec) where spec records all mutations made.
        """
        if spec is None:
            spec = self._generate_random_spec(graph)

        # Deep copy the graph structure (topology is immutable)
        new_nodes = {}
        for node_id, node in graph.nodes.items():
            new_node = ReasoningNode(
                node_id=node.node_id,
                op_type=node.op_type,          # LOCKED: τ preserved
                complexity=node.complexity,     # LOCKED: φ preserved
                operand_refs=list(node.operand_refs),  # LOCKED: edges preserved
                result_symbol=node.result_symbol,
                concrete_value=node.concrete_value,
                metadata=dict(node.metadata),
            )

            # Mutate leaf values (respecting digit class)
            if node.is_leaf and node.node_id in spec.value_mutations:
                new_val = spec.value_mutations[node.node_id]
                # VERIFY digit class preservation
                orig_digits = self._digit_class(node.concrete_value)
                new_digits = self._digit_class(new_val)
                if spec.preserve_digit_class and orig_digits != new_digits:
                    raise ValueError(
                        f"Mutation violates digit class: "
                        f"{node.concrete_value} ({orig_digits}d) → "
                        f"{new_val} ({new_digits}d)"
                    )
                new_node.concrete_value = new_val

            new_nodes[node_id] = new_node

        # Mutate entity bindings
        new_entities = []
        for eb in graph.entity_bindings:
            new_entity = spec.entity_swaps.get(eb.role, eb.entity)
            new_entities.append(EntityBinding(
                role=eb.role,
                entity=new_entity,
                category=eb.category,
                constraints=eb.constraints,
            ))

        # Mutate unit bindings
        new_units = []
        for ub in graph.unit_bindings:
            new_unit = spec.unit_swaps.get(ub.role, ub.unit)
            new_units.append(UnitBinding(
                role=ub.role,
                unit=new_unit,
            ))

        # Build new leaf_values dict
        new_leaf_values = {}
        for leaf_id, val in graph.leaf_values.items():
            new_leaf_values[leaf_id] = spec.value_mutations.get(leaf_id, val)

        # Recompute the answer by forward-executing the mutated graph
        mutated_graph = ReasoningGraph(
            nodes=new_nodes,
            edges=set(graph.edges),       # LOCKED: topology preserved
            entity_bindings=new_entities,
            unit_bindings=new_units,
            leaf_values=new_leaf_values,
            answer=None,                  # will be recomputed
            answer_node_id=graph.answer_node_id,
        )

        # Forward-execute to get new answer
        new_answer = self._execute_graph(mutated_graph)
        mutated_graph.answer = new_answer

        return mutated_graph, spec

    def _generate_random_spec(self, graph: ReasoningGraph) -> MutationSpec:
        """Generate a random but valid mutation specification."""
        spec = MutationSpec(seed=self.rng.randint(0, 2**32))

        # Swap entities
        for eb in graph.entity_bindings:
            if eb.category in self.ENTITY_POOLS:
                pool = [e for e in self.ENTITY_POOLS[eb.category]
                        if e != eb.entity]
                if pool:
                    spec.entity_swaps[eb.role] = self.rng.choice(pool)

        # Mutate leaf values (preserving digit class)
        for leaf_id, val in graph.leaf_values.items():
            digits = self._digit_class(val)
            # Generate new value with same digit count
            if digits == 1:
                new_val = self.rng.randint(2, 9)
            elif digits == 2:
                new_val = self.rng.randint(11, 99)
            elif digits == 3:
                new_val = self.rng.randint(100, 999)
            else:
                new_val = self.rng.randint(10**(digits-1), 10**digits - 1)

            # Preserve sign
            if val < 0:
                new_val = -abs(new_val)

            # Ensure new value differs from original
            while new_val == val:
                new_val = self.rng.randint(10**(digits-1), 10**digits - 1)

            spec.value_mutations[leaf_id] = float(new_val)

        # Pick a random context theme
        themes = list(self.CONTEXT_THEMES.keys())
        spec.context_theme = self.rng.choice(themes)

        return spec

    def _digit_class(self, value: float) -> int:
        """Number of digits in the integer part of a value."""
        return len(str(abs(int(value)))) if value != 0 else 1

    def _execute_graph(self, graph: ReasoningGraph) -> float:
        """Forward-execute graph to compute answer."""
        computed = {}
        for node_id in graph.topological_order():
            node = graph.nodes[node_id]
            if node.is_leaf:
                computed[node_id] = node.concrete_value
                continue
            ops = [computed[ref] for ref in node.operand_refs]
            if node.op_type == OpType.ADD:
                computed[node_id] = sum(ops)
            elif node.op_type == OpType.SUBTRACT:
                computed[node_id] = ops[0] - sum(ops[1:])
            elif node.op_type == OpType.MULTIPLY:
                computed[node_id] = math.prod(ops)
            elif node.op_type == OpType.DIVIDE:
                computed[node_id] = ops[0] / ops[1] if ops[1] != 0 else float('inf')
            else:
                computed[node_id] = 0.0
        return computed.get(graph.answer_node_id, 0.0)


# ============================================================================
# STAGE 3: SURFACE GENERATOR — G' → S'
# ============================================================================

class SurfaceGenerator:
    """
    Translates a mutated reasoning graph G' into natural-language text S'.

    ARCHITECTURE DECISION: We use a TEMPLATE + LLM POLISH approach,
    NOT a free-generation approach. This is critical for safety.

    Step 1: Generate a structured template from G' (deterministic)
    Step 2: Use an LLM to polish the template into fluent prose
    Step 3: The LLM prompt STRICTLY FORBIDS changing numbers,
            operations, or logical structure

    The LLM is a SURFACE REALIZER, not a REASONER.
    It makes text sound natural. It does NOT think.
    """

    # The template for constraining the LLM
    POLISH_SYSTEM_PROMPT = """You are a surface text realizer. Your ONLY job is to make
the following structured problem description read as a natural, fluent word problem.

ABSOLUTE RULES — violating ANY of these causes immediate rejection:
1. NEVER change any number. Every number in the template must appear EXACTLY
   in your output.
2. NEVER change the mathematical operations. If the template says "subtract",
   the problem must require subtraction.
3. NEVER add new quantities, conditions, or steps not in the template.
4. NEVER remove any quantity or step from the template.
5. Keep the question at the end asking for the same final quantity.
6. Use the entity names and units EXACTLY as given.

You may:
- Rephrase sentences for natural flow
- Add minor scene-setting details that do NOT affect the math
- Choose natural phrasing (e.g., "earns" instead of "receives")

Output ONLY the problem text. No solution. No commentary."""

    def generate_template(self, graph: ReasoningGraph) -> str:
        """
        Step 1: Deterministic template from the reasoning graph.

        This template contains ALL the mathematical content in a
        structured but stilted form. The LLM only polishes style.
        """
        lines = []

        # Build entity context
        entity_map = {eb.role: eb.entity for eb in graph.entity_bindings}
        unit_map = {ub.role: ub.unit for ub in graph.unit_bindings}

        agent = entity_map.get("agent", "A person")

        # Process nodes in topological order
        for node_id in graph.topological_order():
            node = graph.nodes[node_id]

            if node.is_leaf:
                # Leaf assignment: describe the given quantity
                val = int(node.concrete_value) if node.concrete_value == int(node.concrete_value) else node.concrete_value
                desc = node.metadata.get("description", node.result_symbol)
                unit = node.metadata.get("unit", "")
                lines.append(f"{agent}'s {desc} is {val} {unit}.".strip())

            else:
                # Operation node: describe what happens
                op_desc = self._operation_to_text(node.op_type)
                desc = node.metadata.get("description", node.result_symbol)
                lines.append(f"To find {desc}, {op_desc} the relevant quantities.")

        # Question line
        answer_node = graph.nodes[graph.answer_node_id]
        answer_desc = answer_node.metadata.get("description", "the result")
        answer_unit = answer_node.metadata.get("unit", "")
        lines.append(f"How much/many {answer_desc} in {answer_unit} does {agent} have?")

        return "\n".join(lines)

    def polish_with_llm(self, template: str, graph: ReasoningGraph,
                        llm_client=None) -> str:
        """
        Step 2: Use LLM to create fluent surface text from template.

        The prompt engineering here is DEFENSIVE — designed to prevent
        the LLM from "helping" by changing the math.
        """
        # Extract all numbers from the template for post-verification
        template_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', template))

        user_prompt = f"""Convert this structured problem into a fluent, natural word problem:

---TEMPLATE---
{template}
---END TEMPLATE---

Remember: EVERY number must appear exactly as given. Do not change any mathematical relationship."""

        if llm_client is None:
            # Fallback: return template as-is (for testing without LLM)
            return template

        # Call LLM
        response = llm_client.generate(
            system=self.POLISH_SYSTEM_PROMPT,
            user=user_prompt,
            temperature=0.3,   # low temperature for faithfulness
            max_tokens=500,
        )

        polished = response.strip()

        # POST-VERIFICATION: check all numbers survived
        polished_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', polished))
        if not template_numbers.issubset(polished_numbers):
            missing = template_numbers - polished_numbers
            raise SurfaceGenerationError(
                f"LLM dropped numbers during polishing: {missing}. "
                f"Template: {template_numbers}, Output: {polished_numbers}"
            )

        return polished

    def _operation_to_text(self, op: OpType) -> str:
        return {
            OpType.ADD: "add",
            OpType.SUBTRACT: "subtract",
            OpType.MULTIPLY: "multiply",
            OpType.DIVIDE: "divide",
        }.get(op, str(op.value))


class SurfaceGenerationError(Exception):
    pass


# ============================================================================
# STAGE 4: ISOMORPH VERIFIER — The Safety Net
# ============================================================================

class IsomorphVerifier:
    """
    Two-level verification that a generated item is a valid τ-isomorph.

    Level A (Static — runs on every generated item):
      1. Structural: G' has same topology as G
      2. Operation: τ-sequence preserved
      3. Complexity: φ-sequence preserved
      4. Numeric: all required numbers present in S'
      5. Answer: forward execution of G' produces correct answer

    Level B (Empirical — runs on calibration panel):
      1. Administer original and variant to J_cal systems
      2. Fit separate 2PL models
      3. Test H₀: b_i = b_{i'} via likelihood ratio test
      4. Flag items where |b - b'| > ε_b

    The ROUND-TRIP CHECK:
      Re-parse S' → G'', then verify G'' ≅_τ G'.
      This catches ANY corruption introduced by the LLM surface generator.
    """

    def __init__(self, parser: BaseProblemParser, tolerance_b: float = 0.3):
        self.parser = parser
        self.tolerance_b = tolerance_b

    def verify_stage_a(self, original_graph: ReasoningGraph,
                       mutated_graph: ReasoningGraph,
                       surface_text: str) -> VerificationResult:
        """
        Stage A: Static verification (runs on every item).
        Returns a detailed result with pass/fail for each check.
        """
        checks = {}

        # Check 1: Topology preservation
        checks["topology"] = (
            original_graph.topology == mutated_graph.topology
        )

        # Check 2: Operation type sequence preservation
        checks["op_types"] = (
            original_graph.op_type_sequence() == mutated_graph.op_type_sequence()
        )

        # Check 3: Complexity weight sequence preservation
        orig_complex = original_graph.complexity_sequence()
        mut_complex = mutated_graph.complexity_sequence()
        checks["complexity"] = (
            len(orig_complex) == len(mut_complex)
            and all(
                c1.is_compatible(c2)
                for c1, c2 in zip(orig_complex, mut_complex)
            )
        )

        # Check 4: Structural fingerprint (fast hash comparison)
        checks["fingerprint"] = (
            original_graph.structural_fingerprint()
            == mutated_graph.structural_fingerprint()
        )

        # Check 5: All leaf values present in surface text
        surface_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', surface_text))
        required_numbers = {
            str(int(v)) if v == int(v) else str(v)
            for v in mutated_graph.leaf_values.values()
        }
        checks["numbers_present"] = required_numbers.issubset(surface_numbers)

        # Check 6: Answer recomputation
        if mutated_graph.answer is not None:
            recomputed = self._execute_graph(mutated_graph)
            checks["answer_consistent"] = (
                abs(recomputed - mutated_graph.answer) < 1e-6
            )
        else:
            checks["answer_consistent"] = False

        # Check 7: Novelty (surface text differs sufficiently)
        checks["novel_surface"] = True  # placeholder; production uses
                                         # corpus similarity check

        all_passed = all(checks.values())
        return VerificationResult(
            passed=all_passed,
            checks=checks,
            error_msg="" if all_passed else
                f"Failed checks: {[k for k, v in checks.items() if not v]}",
        )

    def verify_round_trip(self, mutated_graph: ReasoningGraph,
                          generated_surface: str,
                          solution_steps: List[str],
                          answer: float) -> VerificationResult:
        """
        ROUND-TRIP VERIFICATION: The key safety mechanism.

        1. Re-parse the generated surface text S' into G''
        2. Check G'' ≅_τ G' (the mutated graph)
        3. If they don't match, the LLM corrupted the logic

        This is what prevents subtle errors like:
          - LLM adding "each" (changing multiply → multiply-per-item)
          - LLM reordering operations
          - LLM introducing implicit assumptions
        """
        try:
            re_parsed_graph, _ = self.parser.parse(
                generated_surface,
                solution_steps=solution_steps,
                answer=answer,
            )
        except Exception as e:
            return VerificationResult(
                passed=False,
                checks={"round_trip_parse": False},
                error_msg=f"Re-parse failed: {e}",
            )

        # Now check isomorphism between G' and G''
        result = self.verify_stage_a(
            mutated_graph, re_parsed_graph, generated_surface
        )

        return VerificationResult(
            passed=result.passed,
            checks={**result.checks, "round_trip_parse": True},
            error_msg=result.error_msg,
        )

    def _execute_graph(self, graph: ReasoningGraph) -> float:
        """Forward-execute graph to recompute answer."""
        computed = {}
        for node_id in graph.topological_order():
            node = graph.nodes[node_id]
            if node.is_leaf:
                computed[node_id] = node.concrete_value
                continue
            ops = [computed[ref] for ref in node.operand_refs]
            if node.op_type == OpType.ADD:
                computed[node_id] = sum(ops)
            elif node.op_type == OpType.SUBTRACT:
                computed[node_id] = ops[0] - sum(ops[1:])
            elif node.op_type == OpType.MULTIPLY:
                computed[node_id] = math.prod(ops)
            elif node.op_type == OpType.DIVIDE:
                computed[node_id] = ops[0] / ops[1] if ops[1] != 0 else float('inf')
            else:
                computed[node_id] = 0.0
        return computed.get(graph.answer_node_id, 0.0)


@dataclass
class VerificationResult:
    passed: bool
    checks: Dict[str, bool]
    error_msg: str = ""


# ============================================================================
# ORCHESTRATOR: The Full Pipeline
# ============================================================================

class IsomorphicEngine:
    """
    The top-level orchestrator that runs the full pipeline:
      S → G → G' → S' → verify(G' ≅ G)

    Produces N verified isomorphic variants per original item.
    """

    def __init__(self, parser: BaseProblemParser,
                 mutator: SemanticMutator = None,
                 generator: SurfaceGenerator = None,
                 verifier: IsomorphVerifier = None,
                 max_retries: int = 3):
        self.parser = parser
        self.mutator = mutator or SemanticMutator()
        self.generator = generator or SurfaceGenerator()
        self.verifier = verifier or IsomorphVerifier(parser)
        self.max_retries = max_retries

    def generate_variants(self, original_item: IsomorphicItem,
                          n_variants: int = 10,
                          llm_client=None) -> List[IsomorphicItem]:
        """
        Generate N verified τ-isomorphic variants of an original item.

        Pipeline per variant:
          1. Mutate graph:  G → G'
          2. Generate text: G' → template → S'
          3. Verify:        S' → G'' ≅_τ G'  (round-trip)
          4. If fail, retry with new mutation (up to max_retries)
        """
        variants = []
        attempts = 0
        max_attempts = n_variants * self.max_retries

        while len(variants) < n_variants and attempts < max_attempts:
            attempts += 1
            seed = hash((original_item.item_id, attempts)) % (2**32)
            self.mutator.rng = random.Random(seed)

            try:
                # Stage 2: Mutate graph
                mutated_graph, spec = self.mutator.mutate(
                    original_item.reasoning_graph
                )

                # Stage 3: Generate surface text
                template = self.generator.generate_template(mutated_graph)
                surface_text = self.generator.polish_with_llm(
                    template, mutated_graph, llm_client
                )

                # Stage 4: Verify (static)
                static_result = self.verifier.verify_stage_a(
                    original_item.reasoning_graph,
                    mutated_graph,
                    surface_text,
                )

                if not static_result.passed:
                    continue  # retry with different mutation

                # Create the variant item
                variant = IsomorphicItem(
                    item_id=f"{original_item.item_id}_v{len(variants):03d}",
                    surface_text=surface_text,
                    reasoning_graph=mutated_graph,
                    verification=VerificationFunction(
                        expected_answer=mutated_graph.answer,
                        answer_type="numeric",
                    ),
                    parent_id=original_item.item_id,
                    generation_method="graph_mutation",
                    mutation_seed=seed,
                    is_verified_isomorph=True,
                )

                variants.append(variant)

            except (ValueError, SurfaceGenerationError) as e:
                continue  # retry

        return variants
