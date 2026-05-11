"""
Isomorph-Eval: Production GSM8K Parser
=======================================
Automatically converts any GSM8K problem (surface text + chain-of-thought
solution) into a strict ReasoningGraph using a hybrid strategy:

  1. RULE-BASED: Parse the <<computation=result>> annotations that GSM8K
     solutions contain. This extracts the computation DAG with 100%
     arithmetic fidelity — no LLM involved, no hallucination possible.

  2. LLM STRUCTURED OUTPUT: Enrich the skeleton with semantic annotations
     (entity bindings, descriptions, units) using Pydantic-enforced schemas
     via the Anthropic API's tool_use.

  3. FORWARD EXECUTION: Recompute the answer from the extracted graph.
     If it doesn't match the ground-truth GSM8K answer, REJECT and RETRY.
     We NEVER trust extraction blindly.

Architecture:
  Rule-based extraction handles WHAT is computed (the τ-invariant).
  LLM extraction handles WHAT things are called (the surface layer).
  Forward execution is the non-negotiable safety net.

Dependencies: pydantic>=2.0, anthropic (optional — for LLM enrichment)
"""

from __future__ import annotations

import json
import math
import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator

# Internal data structures from Phase 1/2
from core.data_structures import (
    OpType,
    ComplexityWeight,
    ReasoningNode,
    ReasoningGraph,
    EntityBinding,
    UnitBinding,
    VerificationFunction,
    IsomorphicItem,
)

logger = logging.getLogger(__name__)


# ============================================================================
# SECTION 1: PYDANTIC SCHEMAS FOR STRUCTURED OUTPUT
# ============================================================================
# These schemas enforce our taxonomy at the API boundary.
# The LLM MUST return data matching these shapes exactly.
# Any deviation is caught by Pydantic validation before it
# reaches our internal logic.

class OpTypeEnum(str, Enum):
    """
    Strict subset of OpType for GSM8K math problems.
    We only expose arithmetic operations to the LLM —
    logical ops are irrelevant for this domain.
    """
    ASSIGN   = "assign"
    ADD      = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE   = "divide"
    MODULO   = "modulo"


class NodeSchema(BaseModel):
    """
    Pydantic schema for a single reasoning node.
    Enforced at the API boundary via tool_use / structured output.
    """
    node_id: str = Field(
        ...,
        pattern=r"^n\d+$",
        description="Unique ID: 'n0', 'n1', 'n2', etc. Sequential."
    )
    op_type: OpTypeEnum = Field(
        ...,
        description=(
            "The operation this node performs. "
            "'assign' for given quantities (leaf nodes with no operands). "
            "'add'/'subtract'/'multiply'/'divide' for computation steps."
        )
    )
    operand_refs: list[str] = Field(
        default_factory=list,
        description=(
            "List of node_ids this node depends on. "
            "EMPTY for 'assign' (leaf) nodes. "
            "For 'subtract': first element is the minuend, rest are subtrahends. "
            "For 'divide': first element is the dividend, second is the divisor."
        )
    )
    concrete_value: float = Field(
        ...,
        description=(
            "The numeric result of this node. "
            "For 'assign' nodes: the given number from the problem. "
            "For computation nodes: the computed result."
        )
    )
    result_symbol: str = Field(
        ...,
        description=(
            "A short snake_case variable name describing what this value "
            "represents. E.g., 'eggs_per_day', 'remaining_eggs', 'daily_revenue'."
        )
    )
    description: str = Field(
        ...,
        description=(
            "A brief natural-language description of this quantity. "
            "E.g., 'number of eggs laid per day', 'total daily revenue'."
        )
    )
    unit: str = Field(
        default="",
        description="The unit of measurement: 'eggs', 'dollars', 'miles', etc."
    )

    @field_validator("operand_refs")
    @classmethod
    def validate_operand_refs(cls, v, info):
        op = info.data.get("op_type")
        if op == OpTypeEnum.ASSIGN and len(v) > 0:
            raise ValueError("ASSIGN nodes must have empty operand_refs")
        if op != OpTypeEnum.ASSIGN and len(v) < 2:
            if op in (OpTypeEnum.SUBTRACT, OpTypeEnum.DIVIDE, OpTypeEnum.ADD,
                      OpTypeEnum.MULTIPLY):
                # Allow single-operand subtract for cases like "total - x"
                # where x is actually a multi-step subtraction chain
                pass
        return v


class EntityBindingSchema(BaseModel):
    """Schema for an entity binding (semantic role → surface string)."""
    role: str = Field(
        ...,
        description=(
            "The abstract semantic role: 'agent', 'item', 'recipient', "
            "'venue', 'activity', 'animal', 'object', etc."
        )
    )
    entity: str = Field(
        ...,
        description="The concrete surface string: 'Janet', 'eggs', 'farmers market'."
    )
    category: str = Field(
        default="",
        description="Ontological category: 'person', 'animal', 'product', 'place', etc."
    )


class UnitBindingSchema(BaseModel):
    """Schema for a unit binding."""
    role: str = Field(..., description="What this unit measures: 'price', 'weight', etc.")
    unit: str = Field(..., description="The unit: 'dollars', 'kg', 'miles', etc.")
    symbol: str = Field(default="", description="Symbol: '$', 'kg', etc.")
    position: str = Field(
        default="suffix",
        description="'prefix' for currencies like $, 'suffix' for units like kg"
    )


class ReasoningGraphSchema(BaseModel):
    """
    Complete Pydantic schema for a GSM8K reasoning graph.
    This is the shape the LLM must return via structured output.
    """
    nodes: list[NodeSchema] = Field(
        ...,
        min_length=2,
        description=(
            "ALL nodes in the reasoning graph, in topological order. "
            "Start with ASSIGN nodes for every given number, then "
            "computation nodes in the order they are computed."
        )
    )
    entity_bindings: list[EntityBindingSchema] = Field(
        default_factory=list,
        description="All named entities mentioned in the problem."
    )
    unit_bindings: list[UnitBindingSchema] = Field(
        default_factory=list,
        description="All units of measurement used in the problem."
    )
    answer_node_id: str = Field(
        ...,
        pattern=r"^n\d+$",
        description="The node_id of the final node that produces the answer."
    )

    @model_validator(mode="after")
    def validate_graph_consistency(self) -> "ReasoningGraphSchema":
        node_ids = {n.node_id for n in self.nodes}

        # Check answer_node_id exists
        if self.answer_node_id not in node_ids:
            raise ValueError(
                f"answer_node_id '{self.answer_node_id}' not found in nodes. "
                f"Available: {sorted(node_ids)}"
            )

        # Check all operand_refs point to existing nodes
        for node in self.nodes:
            for ref in node.operand_refs:
                if ref not in node_ids:
                    raise ValueError(
                        f"Node '{node.node_id}' references operand '{ref}' "
                        f"which does not exist. Available: {sorted(node_ids)}"
                    )

        # Check no forward references (operands must be defined before use)
        defined = set()
        for node in self.nodes:
            for ref in node.operand_refs:
                if ref not in defined:
                    raise ValueError(
                        f"Node '{node.node_id}' references '{ref}' which "
                        f"hasn't been defined yet. Nodes must be in "
                        f"topological order."
                    )
            defined.add(node.node_id)

        # Check unique node_ids
        if len(node_ids) != len(self.nodes):
            raise ValueError("Duplicate node_ids detected")

        return self


# ============================================================================
# SECTION 2: THE EXTRACTION PROMPT
# ============================================================================

SYSTEM_PROMPT = """You are a precise mathematical graph extractor. Your task is to
decompose a GSM8K math word problem and its solution into a REASONING GRAPH: a
directed acyclic graph of computation nodes.

STRICT RULES — violating any causes rejection:

1. ASSIGN NODES FIRST: Create one ASSIGN node for EVERY distinct given number
   in the problem. These are leaf nodes with op_type="assign" and EMPTY
   operand_refs. The concrete_value is the number itself.

2. COMPUTATION NODES NEXT: For each arithmetic step in the solution, create
   one computation node. The op_type must be exactly ONE of:
   "add", "subtract", "multiply", "divide".

3. OPERAND ORDER MATTERS:
   - For subtract: operand_refs[0] is the minuend (what you subtract FROM).
     All subsequent elements are subtrahends.
   - For divide: operand_refs[0] is the dividend, operand_refs[1] is the divisor.
   - For add/multiply: order does not matter mathematically.

4. EVERY NODE must have a concrete_value that is the EXACT numeric result
   of that step. No rounding. No approximation.

5. TOPOLOGICAL ORDER: Nodes must be listed so that every operand_ref refers
   to a node_id that was defined EARLIER in the list.

6. ANSWER NODE: The last computation node should produce the final answer.
   Set answer_node_id to that node's id.

7. DO NOT merge multiple operations into one node. If the solution says
   "16 - 3 - 4 = 9", that is ONE subtract node with three operands
   (n0 - n1 - n2), NOT two separate subtractions.

8. DO NOT create unnecessary nodes. If a number from the problem is used
   directly in a computation, reference its ASSIGN node — don't create
   a duplicate.

9. ENTITY BINDINGS: Extract every named entity (people, objects, animals,
   places) and assign a semantic role. Common roles: agent, item, recipient,
   venue, animal, object, activity.

10. UNIT BINDINGS: Extract units of measurement (dollars, eggs, miles, etc.)
    and note whether symbols are prefix ($) or suffix (kg).

EXAMPLE:

Problem: "Tom has 5 apples. He buys 3 more. How many apples does he have?"
Solution: "Tom starts with 5 apples. He buys 3 more. 5 + 3 = 8. The answer is 8."

Graph:
  nodes:
    - node_id: "n0", op_type: "assign", operand_refs: [], concrete_value: 5,
      result_symbol: "initial_apples", description: "apples Tom starts with"
    - node_id: "n1", op_type: "assign", operand_refs: [], concrete_value: 3,
      result_symbol: "bought_apples", description: "apples Tom buys"
    - node_id: "n2", op_type: "add", operand_refs: ["n0", "n1"],
      concrete_value: 8, result_symbol: "total_apples",
      description: "total apples after buying"
  answer_node_id: "n2"
  entity_bindings: [{role: "agent", entity: "Tom", category: "person"}]
  unit_bindings: [{role: "quantity", unit: "apples"}]"""


USER_PROMPT_TEMPLATE = """Extract the reasoning graph from this GSM8K problem.

PROBLEM:
{question}

SOLUTION:
{solution}

GROUND-TRUTH ANSWER: {answer}

Return the complete reasoning graph. Every number in the problem that
participates in the solution must appear as an ASSIGN node. Every
arithmetic step must be a separate computation node. The final node
must produce exactly {answer}."""


# ============================================================================
# SECTION 3: RULE-BASED ANNOTATION PARSER
# ============================================================================
# GSM8K solutions contain <<computation=result>> annotations.
# Example: "She has 16 - 3 - 4 = <<16-3-4=9>>9 remaining eggs"
# We parse these annotations to extract the computation DAG
# WITHOUT any LLM involvement — zero hallucination risk.

class AnnotationParser:
    """
    Extracts computation steps from GSM8K's <<expr=result>> annotations.

    This is the HIGH-FIDELITY path: the arithmetic graph is extracted
    entirely from deterministic pattern matching. No LLM, no risk.
    """

    # Pattern: <<expression=result>>
    ANNOTATION_RE = re.compile(r"<<([^>]+?)=([^>]+?)>>")

    # Pattern: #### final_answer
    ANSWER_RE = re.compile(r"####\s*(.+)")

    # Arithmetic expression tokenizer
    TOKEN_RE = re.compile(r"(\d+(?:\.\d+)?|[+\-*/()%])")

    def parse_annotations(
        self, solution_text: str
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Extract computation steps from <<>> annotations.

        Returns a list of dicts:
          [{"expression": "16-3-4", "result": 9.0, "raw": "<<16-3-4=9>>"},
           {"expression": "9*2", "result": 18.0, "raw": "<<9*2=18>>"}]

        Returns None if no annotations found.
        """
        matches = self.ANNOTATION_RE.findall(solution_text)
        if not matches:
            return None

        steps = []
        for expr, result_str in matches:
            try:
                result = self._parse_number(result_str.strip())
                steps.append({
                    "expression": expr.strip(),
                    "result": result,
                    "raw": f"<<{expr}={result_str}>>",
                })
            except ValueError:
                continue

        return steps if steps else None

    def build_graph_from_annotations(
        self,
        question: str,
        solution: str,
        answer: float,
    ) -> Optional[ReasoningGraphSchema]:
        """
        Build a complete graph schema from annotations alone.

        Strategy:
          1. Find all <<expr=result>> annotations
          2. Collect all leaf numbers (numbers that appear as literals)
          3. Build ASSIGN nodes for leaf numbers
          4. Build computation nodes for each step
          5. Wire dependencies by matching numeric values
        """
        steps = self.parse_annotations(solution)
        if not steps:
            return None

        # Collect all unique literal numbers from expressions
        all_literals: Dict[float, str] = {}  # value → first_occurrence_context
        node_list: List[NodeSchema] = []
        value_to_node: Dict[float, str] = {}  # maps a computed value to its node_id
        node_counter = 0

        # Phase 1: Identify all leaf values across all steps
        leaf_values_needed: List[float] = []
        for step in steps:
            tokens = self.TOKEN_RE.findall(step["expression"])
            for tok in tokens:
                if tok not in "+-*/()%":
                    val = self._parse_number(tok)
                    if val not in all_literals:
                        all_literals[val] = tok
                        leaf_values_needed.append(val)

        # Phase 2: Create ASSIGN nodes for leaf values
        # But only for values that aren't results of prior computations
        computed_results: Set[float] = set()

        for val in leaf_values_needed:
            nid = f"n{node_counter}"
            node_list.append(NodeSchema(
                node_id=nid,
                op_type=OpTypeEnum.ASSIGN,
                operand_refs=[],
                concrete_value=val,
                result_symbol=f"given_{node_counter}",
                description=f"given value {val}",
                unit="",
            ))
            value_to_node[val] = nid
            node_counter += 1

        # Phase 3: Create computation nodes for each annotation step
        for step in steps:
            expr = step["expression"]
            result = step["result"]

            # Parse the expression to determine operation and operands
            op_type, operand_values = self._parse_expression(expr)
            if op_type is None:
                continue

            # Resolve operand references
            operand_refs = []
            for oval in operand_values:
                if oval in value_to_node:
                    operand_refs.append(value_to_node[oval])
                else:
                    # This value wasn't a leaf — it might be a prior computation
                    # or a number we missed. Create an assign node for it.
                    nid = f"n{node_counter}"
                    node_list.append(NodeSchema(
                        node_id=nid,
                        op_type=OpTypeEnum.ASSIGN,
                        operand_refs=[],
                        concrete_value=oval,
                        result_symbol=f"given_{node_counter}",
                        description=f"given value {oval}",
                        unit="",
                    ))
                    value_to_node[oval] = nid
                    node_counter += 1
                    operand_refs.append(nid)

            # Create computation node
            comp_nid = f"n{node_counter}"
            node_list.append(NodeSchema(
                node_id=comp_nid,
                op_type=op_type,
                operand_refs=operand_refs,
                concrete_value=result,
                result_symbol=f"step_{node_counter}",
                description=f"computation step",
                unit="",
            ))
            value_to_node[result] = comp_nid
            computed_results.add(result)
            node_counter += 1

        if not node_list:
            return None

        # Determine answer node (the last computation node)
        answer_nid = node_list[-1].node_id

        # Remove ASSIGN nodes for values that are actually computed by prior steps
        # (a number that appears as a leaf in step 2 but was computed in step 1)
        final_nodes = []
        computed_nids_to_remove = set()
        for node in node_list:
            if (node.op_type == OpTypeEnum.ASSIGN
                    and node.concrete_value in computed_results
                    and any(
                        n.op_type != OpTypeEnum.ASSIGN
                        and n.concrete_value == node.concrete_value
                        for n in node_list
                    )):
                # This leaf's value is produced by a computation node —
                # redirect references to the computation node instead
                comp_node_id = next(
                    n.node_id for n in node_list
                    if n.op_type != OpTypeEnum.ASSIGN
                    and n.concrete_value == node.concrete_value
                )
                # Update all references
                for other in node_list:
                    other.operand_refs = [
                        comp_node_id if ref == node.node_id else ref
                        for ref in other.operand_refs
                    ]
                computed_nids_to_remove.add(node.node_id)
            else:
                final_nodes.append(node)

        # Re-number nodes sequentially
        old_to_new = {}
        renumbered = []
        for i, node in enumerate(final_nodes):
            new_id = f"n{i}"
            old_to_new[node.node_id] = new_id
            node.node_id = new_id
            node.operand_refs = [old_to_new.get(r, r) for r in node.operand_refs]
            node.result_symbol = f"step_{i}" if node.op_type != OpTypeEnum.ASSIGN else f"given_{i}"
            renumbered.append(node)

        answer_nid = old_to_new.get(answer_nid, answer_nid)

        try:
            return ReasoningGraphSchema(
                nodes=renumbered,
                entity_bindings=[],
                unit_bindings=[],
                answer_node_id=answer_nid,
            )
        except Exception as e:
            logger.warning(f"Annotation-based graph failed validation: {e}")
            return None

    def _parse_expression(
        self, expr: str
    ) -> Tuple[Optional[OpTypeEnum], List[float]]:
        """
        Parse a simple arithmetic expression into (op_type, operand_values).

        Handles: "16-3-4", "9*2", "5+3+7", "100/4", "16-3-4+2"
        For mixed operations, identifies the DOMINANT operation.
        """
        expr = expr.strip()

        # Tokenize
        tokens = self.TOKEN_RE.findall(expr)
        if not tokens:
            return None, []

        numbers = []
        operators = []
        for tok in tokens:
            if tok in "+-*/%":
                operators.append(tok)
            elif tok not in "()":
                numbers.append(self._parse_number(tok))

        if not operators:
            return None, numbers

        # Determine dominant operation
        unique_ops = set(operators)

        if unique_ops == {"*"} or unique_ops == {"×"}:
            return OpTypeEnum.MULTIPLY, numbers
        elif unique_ops == {"/"} or unique_ops == {"÷"}:
            return OpTypeEnum.DIVIDE, numbers
        elif unique_ops == {"+"}:
            return OpTypeEnum.ADD, numbers
        elif unique_ops == {"-"}:
            return OpTypeEnum.SUBTRACT, numbers
        elif unique_ops == {"%"}:
            return OpTypeEnum.MODULO, numbers
        elif unique_ops <= {"+", "-"}:
            # Mixed add/subtract: treat as subtract from first operand
            # "16 - 3 - 4 + 2" → subtract(16, 3, 4) then add 2
            # For simplicity, we treat the dominant sign
            if operators.count("-") >= operators.count("+"):
                return OpTypeEnum.SUBTRACT, numbers
            else:
                return OpTypeEnum.ADD, numbers
        else:
            # Mixed operations — try eval for result, pick dominant
            if "*" in unique_ops or "/" in unique_ops:
                if "*" in unique_ops:
                    return OpTypeEnum.MULTIPLY, numbers
                return OpTypeEnum.DIVIDE, numbers
            return OpTypeEnum.ADD, numbers

    def _parse_number(self, s: str) -> float:
        """Parse a number string, handling commas and currency symbols."""
        s = s.strip().replace(",", "").replace("$", "").replace("€", "")
        return float(s)

    def extract_answer(self, text: str) -> Optional[float]:
        """Extract the final answer from '#### answer' pattern."""
        match = self.ANSWER_RE.search(text)
        if match:
            try:
                return self._parse_number(match.group(1))
            except ValueError:
                return None
        return None


# ============================================================================
# SECTION 4: THE PARSER — parse() IMPLEMENTATION
# ============================================================================

class ParseError(Exception):
    """Raised when graph extraction fails verification."""
    pass


class GSM8KParser:
    """
    Production parser for GSM8K math word problems.

    Converts (surface_text, solution) → (ReasoningGraph, VerificationFunction)

    Strategy:
      1. Try rule-based extraction from <<>> annotations (zero hallucination)
      2. Use LLM structured output for semantic enrichment (or full extraction
         if annotations are missing)
      3. Forward-execute the graph and verify against ground-truth answer
      4. On failure, retry with stricter prompting (up to max_retries)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        max_retries: int = 3,
        tolerance: float = 0.01,
    ):
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.tolerance = tolerance
        self.annotation_parser = AnnotationParser()
        self._client = None

    @property
    def client(self):
        """Lazy-load the Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package required for LLM extraction. "
                    "Install with: pip install anthropic"
                )
        return self._client

    def parse(
        self,
        question: str,
        solution: str,
        answer: Optional[float] = None,
    ) -> Tuple[ReasoningGraph, VerificationFunction]:
        """
        Parse a GSM8K problem into (ReasoningGraph, VerificationFunction).

        Parameters
        ----------
        question : str
            The problem text (what the model sees)
        solution : str
            The chain-of-thought solution with <<>> annotations
        answer : float, optional
            The ground-truth answer. Extracted from #### if not provided.

        Returns
        -------
        (ReasoningGraph, VerificationFunction)

        Raises
        ------
        ParseError
            If the graph cannot be extracted and verified after max_retries.
        """
        # Extract answer from solution if not provided
        if answer is None:
            answer = self.annotation_parser.extract_answer(solution)
            if answer is None:
                raise ParseError(
                    "Could not extract answer from solution. "
                    "Provide the answer explicitly."
                )

        errors: List[str] = []

        # ---- ATTEMPT 1: Rule-based annotation parsing ----
        annotation_schema = self.annotation_parser.build_graph_from_annotations(
            question, solution, answer
        )

        if annotation_schema is not None:
            try:
                graph = self._schema_to_internal(annotation_schema, answer)
                computed = self._execute_graph(graph)
                if self._answer_matches(computed, answer):
                    logger.info("Rule-based parsing succeeded on first try")
                    # Enrich with LLM semantics if available
                    graph = self._enrich_semantics(graph, question, solution)
                    verification = VerificationFunction(
                        expected_answer=answer, answer_type="numeric",
                        tolerance=self.tolerance,
                    )
                    return graph, verification
                else:
                    errors.append(
                        f"Rule-based graph computed {computed}, "
                        f"expected {answer}"
                    )
            except Exception as e:
                errors.append(f"Rule-based conversion failed: {e}")

        # ---- ATTEMPT 2+: LLM structured extraction with retries ----
        for attempt in range(self.max_retries):
            try:
                schema = self._extract_via_llm(
                    question, solution, answer,
                    prior_errors=errors if attempt > 0 else None,
                )
                graph = self._schema_to_internal(schema, answer)
                computed = self._execute_graph(graph)

                if self._answer_matches(computed, answer):
                    logger.info(
                        f"LLM extraction succeeded on attempt {attempt + 1}"
                    )
                    verification = VerificationFunction(
                        expected_answer=answer, answer_type="numeric",
                        tolerance=self.tolerance,
                    )
                    return graph, verification
                else:
                    errors.append(
                        f"Attempt {attempt + 1}: graph computed {computed}, "
                        f"expected {answer}"
                    )

            except Exception as e:
                errors.append(f"Attempt {attempt + 1} failed: {e}")

        # All attempts exhausted
        raise ParseError(
            f"Failed to extract valid graph after {self.max_retries} attempts. "
            f"Errors:\n" + "\n".join(f"  [{i+1}] {e}" for i, e in enumerate(errors))
        )

    # ----------------------------------------------------------------
    # LLM Structured Extraction
    # ----------------------------------------------------------------

    def _extract_via_llm(
        self,
        question: str,
        solution: str,
        answer: float,
        prior_errors: Optional[List[str]] = None,
    ) -> ReasoningGraphSchema:
        """
        Call the Anthropic API with tool_use to extract a structured graph.

        Uses the Pydantic schema as the tool definition, ensuring the LLM
        returns data that passes all our validators.
        """
        user_prompt = USER_PROMPT_TEMPLATE.format(
            question=question,
            solution=solution,
            answer=answer,
        )

        # If retrying, append error feedback
        if prior_errors:
            user_prompt += "\n\nPRIOR EXTRACTION FAILED. Errors:\n"
            for err in prior_errors[-3:]:  # last 3 errors
                user_prompt += f"  - {err}\n"
            user_prompt += (
                "\nFix these issues. Ensure the graph forward-executes to "
                f"exactly {answer}. Double-check operand_refs ordering for "
                "subtract and divide operations."
            )

        # Build tool definition from Pydantic schema
        tool_schema = ReasoningGraphSchema.model_json_schema()

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            tools=[{
                "name": "extract_reasoning_graph",
                "description": (
                    "Extract the reasoning graph from a GSM8K problem. "
                    "Returns a structured graph with nodes, edges, "
                    "entity bindings, and the answer node."
                ),
                "input_schema": tool_schema,
            }],
            tool_choice={"type": "tool", "name": "extract_reasoning_graph"},
        )

        # Extract the tool use result
        for block in response.content:
            if block.type == "tool_use":
                # Validate through Pydantic
                return ReasoningGraphSchema.model_validate(block.input)

        raise ParseError("LLM did not return a tool_use block")

    # ----------------------------------------------------------------
    # Semantic Enrichment (LLM adds descriptions to rule-based skeleton)
    # ----------------------------------------------------------------

    def _enrich_semantics(
        self,
        graph: ReasoningGraph,
        question: str,
        solution: str,
    ) -> ReasoningGraph:
        """
        Use LLM to add semantic annotations to a rule-based skeleton.

        The arithmetic is already verified — we only ask for:
        - result_symbol (variable names)
        - descriptions
        - entity_bindings
        - unit_bindings

        If LLM is unavailable, return the graph with generic labels.
        """
        if self.api_key is None:
            return graph  # no LLM available, keep generic labels

        enrich_prompt = (
            "Given this math problem, provide semantic labels for each "
            "computation step. I'll give you the node values — you provide "
            "a result_symbol (snake_case variable name), a description, "
            "and a unit for each.\n\n"
            f"PROBLEM: {question}\n\n"
            f"NODES (by value):\n"
        )
        for nid in graph.topological_order():
            node = graph.nodes[nid]
            enrich_prompt += (
                f"  {nid}: {node.op_type.value}, value={node.concrete_value}\n"
            )

        enrich_prompt += (
            "\nAlso extract entity_bindings (role, entity, category) "
            "and unit_bindings (role, unit, symbol, position).\n"
            "Return as JSON with keys: node_labels (dict of node_id → "
            "{result_symbol, description, unit}), entity_bindings (list), "
            "unit_bindings (list)."
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                messages=[{"role": "user", "content": enrich_prompt}],
            )

            text = response.content[0].text

            # Extract JSON from response
            json_match = re.search(r"\{[\s\S]*\}", text)
            if json_match:
                data = json.loads(json_match.group())

                # Apply labels
                if "node_labels" in data:
                    for nid, labels in data["node_labels"].items():
                        if nid in graph.nodes:
                            node = graph.nodes[nid]
                            node.result_symbol = labels.get(
                                "result_symbol", node.result_symbol
                            )
                            node.metadata["description"] = labels.get(
                                "description", ""
                            )
                            node.metadata["unit"] = labels.get("unit", "")

                # Apply entity bindings
                if "entity_bindings" in data:
                    graph.entity_bindings = [
                        EntityBinding(
                            role=eb.get("role", ""),
                            entity=eb.get("entity", ""),
                            category=eb.get("category", ""),
                        )
                        for eb in data["entity_bindings"]
                    ]

                # Apply unit bindings
                if "unit_bindings" in data:
                    graph.unit_bindings = [
                        UnitBinding(
                            role=ub.get("role", ""),
                            unit=ub.get("unit", ""),
                            symbol=ub.get("symbol", ""),
                            position=ub.get("position", "suffix"),
                        )
                        for ub in data["unit_bindings"]
                    ]

        except Exception as e:
            logger.warning(f"Semantic enrichment failed (non-fatal): {e}")

        return graph

    # ----------------------------------------------------------------
    # Schema → Internal Conversion
    # ----------------------------------------------------------------

    def _schema_to_internal(
        self,
        schema: ReasoningGraphSchema,
        answer: float,
    ) -> ReasoningGraph:
        """
        Convert Pydantic schema objects to our internal dataclasses.

        This is where we compute ComplexityWeight φ(n) for each node.
        """
        nodes: Dict[str, ReasoningNode] = {}
        edges: Set[Tuple[str, str]] = set()
        leaf_values: Dict[str, float] = {}

        for ns in schema.nodes:
            # Map OpTypeEnum → OpType
            op = OpType(ns.op_type.value)

            # Compute ComplexityWeight
            n_operands = len(ns.operand_refs)

            # Digit class: max digits among this node's value and operands
            all_values = [abs(ns.concrete_value)]
            for ref in ns.operand_refs:
                ref_node = next(
                    (n for n in schema.nodes if n.node_id == ref), None
                )
                if ref_node:
                    all_values.append(abs(ref_node.concrete_value))

            digit_class = max(
                len(str(int(abs(v)))) if v != 0 else 1
                for v in all_values
            )

            involves_fraction = any(
                v != int(v) for v in all_values
            )
            involves_negative = any(
                v < 0 for v in all_values
            )

            complexity = ComplexityWeight(
                n_operands=n_operands,
                digit_class=digit_class,
                involves_fraction=involves_fraction,
                involves_negative=involves_negative,
            )

            node = ReasoningNode(
                node_id=ns.node_id,
                op_type=op,
                complexity=complexity,
                operand_refs=list(ns.operand_refs),
                result_symbol=ns.result_symbol,
                concrete_value=ns.concrete_value,
                metadata={
                    "description": ns.description,
                    "unit": ns.unit,
                },
            )
            nodes[ns.node_id] = node

            # Build edges
            for ref in ns.operand_refs:
                edges.add((ref, ns.node_id))

            # Track leaf values
            if op == OpType.ASSIGN and n_operands == 0:
                leaf_values[ns.node_id] = ns.concrete_value

        # Entity bindings
        entity_bindings = [
            EntityBinding(
                role=eb.role,
                entity=eb.entity,
                category=eb.category,
            )
            for eb in schema.entity_bindings
        ]

        # Unit bindings
        unit_bindings = [
            UnitBinding(
                role=ub.role,
                unit=ub.unit,
                symbol=ub.symbol,
                position=ub.position,
            )
            for ub in schema.unit_bindings
        ]

        return ReasoningGraph(
            nodes=nodes,
            edges=edges,
            entity_bindings=entity_bindings,
            unit_bindings=unit_bindings,
            leaf_values=leaf_values,
            answer=answer,
            answer_node_id=schema.answer_node_id,
        )

    # ----------------------------------------------------------------
    # Forward Execution — THE SAFETY NET
    # ----------------------------------------------------------------

    def _execute_graph(self, graph: ReasoningGraph) -> float:
        """
        Forward-execute the reasoning graph to recompute the answer.

        This is the NON-NEGOTIABLE safety check. If the graph's
        computed answer doesn't match ground truth, the parse is wrong.

        Raises
        ------
        ParseError
            If the graph contains cycles or undefined references.
        """
        computed: Dict[str, float] = {}

        try:
            order = graph.topological_order()
        except Exception as e:
            raise ParseError(f"Graph has invalid topology: {e}")

        if len(order) != len(graph.nodes):
            raise ParseError(
                f"Topological sort returned {len(order)} nodes but graph "
                f"has {len(graph.nodes)} — likely contains a cycle."
            )

        for node_id in order:
            node = graph.nodes[node_id]

            if node.is_leaf:
                if node.concrete_value is None:
                    raise ParseError(
                        f"Leaf node {node_id} has no concrete_value"
                    )
                computed[node_id] = node.concrete_value
                continue

            # Gather operand values
            operand_vals = []
            for ref in node.operand_refs:
                if ref not in computed:
                    raise ParseError(
                        f"Node {node_id} references {ref} which has no "
                        f"computed value. Available: {list(computed.keys())}"
                    )
                operand_vals.append(computed[ref])

            if not operand_vals:
                raise ParseError(
                    f"Non-leaf node {node_id} ({node.op_type}) has no operands"
                )

            # Execute the operation
            if node.op_type == OpType.ADD:
                result = sum(operand_vals)

            elif node.op_type == OpType.SUBTRACT:
                result = operand_vals[0] - sum(operand_vals[1:])

            elif node.op_type == OpType.MULTIPLY:
                result = math.prod(operand_vals)

            elif node.op_type == OpType.DIVIDE:
                if len(operand_vals) < 2:
                    raise ParseError(
                        f"DIVIDE node {node_id} needs 2 operands, "
                        f"got {len(operand_vals)}"
                    )
                if operand_vals[1] == 0:
                    raise ParseError(f"Division by zero in node {node_id}")
                result = operand_vals[0] / operand_vals[1]

            elif node.op_type == OpType.MODULO:
                if len(operand_vals) < 2 or operand_vals[1] == 0:
                    raise ParseError(f"Invalid modulo in node {node_id}")
                result = operand_vals[0] % operand_vals[1]

            else:
                raise ParseError(
                    f"Unsupported op_type {node.op_type} in node {node_id}"
                )

            computed[node_id] = result

        answer_nid = graph.answer_node_id
        if answer_nid not in computed:
            raise ParseError(
                f"Answer node '{answer_nid}' was not computed. "
                f"Available: {list(computed.keys())}"
            )

        return computed[answer_nid]

    def _answer_matches(self, computed: float, expected: float) -> bool:
        """Check if computed answer matches expected within tolerance."""
        if expected == 0:
            return abs(computed) < self.tolerance
        return abs(computed - expected) / max(abs(expected), 1e-10) < self.tolerance

    # ----------------------------------------------------------------
    # Batch Processing
    # ----------------------------------------------------------------

    def parse_dataset(
        self,
        dataset: List[Dict[str, str]],
        progress: bool = True,
    ) -> Tuple[List[IsomorphicItem], List[Dict]]:
        """
        Parse an entire GSM8K dataset.

        Parameters
        ----------
        dataset : list of dicts
            Each dict has keys: "question", "answer" (full solution text)
        progress : bool
            Print progress updates

        Returns
        -------
        (successful_items, failed_items)
        """
        successful = []
        failed = []

        for idx, item in enumerate(dataset):
            question = item["question"]
            solution = item["answer"]  # GSM8K uses "answer" for full solution

            if progress and idx % 50 == 0:
                logger.info(
                    f"Parsing {idx}/{len(dataset)} "
                    f"({len(successful)} OK, {len(failed)} failed)"
                )

            try:
                graph, verification = self.parse(
                    question=question,
                    solution=solution,
                )

                iso_item = IsomorphicItem(
                    item_id=f"gsm8k_{idx:04d}",
                    surface_text=question,
                    reasoning_graph=graph,
                    verification=verification,
                    generation_method="original",
                    is_verified_isomorph=True,  # original is trivially isomorphic to itself
                )
                successful.append(iso_item)

            except (ParseError, Exception) as e:
                failed.append({
                    "index": idx,
                    "question": question[:100] + "...",
                    "error": str(e),
                })

        if progress:
            total = len(dataset)
            logger.info(
                f"Parsing complete: {len(successful)}/{total} succeeded "
                f"({len(successful)/total:.1%}), {len(failed)} failed"
            )

        return successful, failed


# ============================================================================
# SECTION 5: SELF-TEST — Verify the parser on known GSM8K examples
# ============================================================================

def self_test():
    """
    Run the parser on 3 representative GSM8K problems using ONLY
    the rule-based annotation parser (no LLM required).
    """
    print("=" * 78)
    print("GSM8K PARSER SELF-TEST (rule-based, no LLM required)")
    print("=" * 78)

    test_cases = [
        {
            "name": "Janet's ducks (3-step: subtract then multiply)",
            "question": (
                "Janet's ducks lay 16 eggs per day. She eats three for "
                "breakfast every morning and bakes muffins for her friends "
                "every day with four. She sells every remaining egg at the "
                "farmers' market daily for $2 per egg. How much in dollars "
                "does she make every day at the farmers' market?"
            ),
            "solution": (
                "Janet eats 3 + 4 = <<3+4=7>>7 eggs per day for meals and baking.\n"
                "She has 16 - 7 = <<16-7=9>>9 eggs remaining.\n"
                "She earns 9 * 2 = <<9*2=18>>18 dollars per day.\n"
                "#### 18"
            ),
            "expected_answer": 18,
        },
        {
            "name": "Simple addition chain",
            "question": (
                "A store has 40 red apples, 25 green apples, and 15 yellow "
                "apples. How many apples are there in total?"
            ),
            "solution": (
                "The store has 40 + 25 + 15 = <<40+25+15=80>>80 apples.\n"
                "#### 80"
            ),
            "expected_answer": 80,
        },
        {
            "name": "Multi-step with division",
            "question": (
                "A baker makes 120 cookies. She puts them equally into 8 boxes. "
                "She then eats 3 cookies from each box. How many cookies are "
                "left in each box?"
            ),
            "solution": (
                "Each box has 120 / 8 = <<120/8=15>>15 cookies.\n"
                "After eating, each box has 15 - 3 = <<15-3=12>>12 cookies.\n"
                "#### 12"
            ),
            "expected_answer": 12,
        },
    ]

    parser = GSM8KParser(api_key=None, max_retries=0)

    for tc in test_cases:
        print(f"\n{'─' * 60}")
        print(f"TEST: {tc['name']}")
        print(f"{'─' * 60}")

        try:
            graph, verification = parser.parse(
                question=tc["question"],
                solution=tc["solution"],
                answer=tc["expected_answer"],
            )

            # Display the graph
            print(f"\n  Nodes ({len(graph.nodes)}):")
            for nid in graph.topological_order():
                node = graph.nodes[nid]
                if node.is_leaf:
                    print(f"    {nid}: ASSIGN = {node.concrete_value}")
                else:
                    refs = ", ".join(node.operand_refs)
                    print(
                        f"    {nid}: {node.op_type.value.upper()}({refs}) "
                        f"= {node.concrete_value}"
                    )

            print(f"\n  Edges ({len(graph.edges)}):")
            for src, tgt in sorted(graph.edges):
                print(f"    {src} → {tgt}")

            print(f"\n  DAG depth: {graph.depth()}")
            print(f"  τ-signature: {[o.value for o in graph.op_type_sequence()]}")
            print(f"  Fingerprint: {graph.structural_fingerprint()}")

            # Forward execution
            computed = parser._execute_graph(graph)
            match = parser._answer_matches(computed, tc["expected_answer"])
            print(f"\n  Forward execution: {computed}")
            print(f"  Expected:          {tc['expected_answer']}")
            print(f"  Match: {'✓ PASS' if match else '✗ FAIL'}")

        except ParseError as e:
            print(f"\n  ✗ PARSE ERROR: {e}")
        except Exception as e:
            print(f"\n  ✗ UNEXPECTED ERROR: {type(e).__name__}: {e}")

    print(f"\n{'=' * 78}")
    print("Self-test complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    self_test()
