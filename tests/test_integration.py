"""Integration tests for the complete single-round tuning flow.

Tests the integration of exam agent, evaluator, and checkpoint
management with mocked OpenAI API responses.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm_hpt.evaluation.evaluator import evaluate
from vllm_hpt.exam.loader import Question
from vllm_hpt.model.client import LLMClient
from vllm_hpt.model.exam_agent import ExamAgent
from vllm_hpt.orchestrator.checkpoint import Checkpoint, CheckpointManager
from vllm_hpt.tuning.history import HistoryEntry, OptimizationHistory
from vllm_hpt.tuning.params import SamplingParams


@pytest.fixture
def sample_questions():
    """Create sample questions for testing."""
    return [
        Question(
            id="q1",
            question="What is 2+2?",
            choices=["A. 3", "B. 4", "C. 5", "D. 6"],
            answer_key="B",
            original_label="B",
        ),
        Question(
            id="q2",
            question="What color is the sky?",
            choices=["A. Red", "B. Blue", "C. Green", "D. Yellow"],
            answer_key="B",
            original_label="B",
        ),
        Question(
            id="q3",
            question="How many legs does a dog have?",
            choices=["A. 2", "B. 3", "C. 4", "D. 5"],
            answer_key="C",
            original_label="C",
        ),
    ]


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = MagicMock(spec=LLMClient)
    client.model = "test-model"
    client.base_url = "http://test:8000/v1"
    client.api_key = "test-key"
    return client


@pytest.mark.asyncio
async def test_single_round_flow(sample_questions, mock_llm_client):
    """Test exam agent + evaluation flow with mocked API responses."""
    # Step 1: Setup - we already have sample_questions
    assert len(sample_questions) == 3
    
    # Step 2: Create mock exam agent that returns fixed answers
    # Mock the chat method to return "The best answer is A" for first question,
    # "The best answer is B" for second, "The best answer is C" for third
    mock_responses = [
        "The best answer is B",  # Correct for q1
        "The best answer is B",  # Correct for q2
        "The best answer is A",  # Wrong for q3 (should be C)
    ]
    
    # Create async mock for chat method
    async def mock_chat_side_effect(messages, sampling_params=None):
        # Return responses in order
        return mock_responses.pop(0)
    
    mock_llm_client.chat = AsyncMock(side_effect=mock_chat_side_effect)
    
    # Create exam agent with mock client
    exam_agent = ExamAgent(mock_llm_client, concurrency=5)
    
    # Create sampling params
    params = SamplingParams(temperature=0.7, top_p=0.9, top_k=50)
    
    # Answer questions
    raw_outputs = await exam_agent.answer_questions(sample_questions, params)
    
    # Verify we got 3 responses
    assert len(raw_outputs) == 3
    assert raw_outputs[0] == "The best answer is B"
    assert raw_outputs[1] == "The best answer is B"
    assert raw_outputs[2] == "The best answer is A"
    
    # Step 3: Evaluate results
    eval_result = evaluate(sample_questions, raw_outputs)
    
    # Verify evaluation results
    assert eval_result.total_count == 3
    assert eval_result.correct_count == 2  # q1 and q2 correct, q3 wrong
    assert eval_result.accuracy == pytest.approx(2/3)
    assert len(eval_result.wrong_questions) == 1
    assert eval_result.wrong_questions[0].question_id == "q3"
    assert eval_result.wrong_questions[0].model_answer == "A"
    assert eval_result.wrong_questions[0].correct_answer == "C"


@pytest.mark.asyncio
async def test_checkpoint_roundtrip(tmp_path):
    """Test checkpoint save and load roundtrip.
    
    Creates a checkpoint with real data, saves it, loads it back,
    and verifies all fields match.
    """
    # Create checkpoint with real data
    params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        top_k=60,
        repetition_penalty=1.2,
        max_tokens=256
    )
    
    history = OptimizationHistory()
    history.add_entry(HistoryEntry(
        round_num=1,
        params=params,
        train_accuracy=0.75,
        validation_accuracy=0.70
    ))
    history.add_entry(HistoryEntry(
        round_num=2,
        params=SamplingParams(temperature=0.6, top_p=0.9),
        train_accuracy=0.80,
        validation_accuracy=0.78
    ))
    
    checkpoint = Checkpoint(
        run_id="test_run_123",
        current_round=2,
        total_rounds=10,
        history=history,
        best_params=params,
        best_validation_accuracy=0.78,
        strategy_name="tpe",
    )
    
    # Save checkpoint
    manager = CheckpointManager()
    checkpoint_path = tmp_path / "test_checkpoint.json"
    manager.save(checkpoint, str(checkpoint_path))
    
    # Verify file exists
    assert checkpoint_path.exists()
    
    # Load checkpoint
    loaded_checkpoint = manager.load(str(checkpoint_path))
    
    # Verify all fields match
    assert loaded_checkpoint.run_id == "test_run_123"
    assert loaded_checkpoint.current_round == 2
    assert loaded_checkpoint.total_rounds == 10
    assert loaded_checkpoint.best_validation_accuracy == 0.78
    
    # Verify best params
    assert loaded_checkpoint.best_params.temperature == 0.8
    assert loaded_checkpoint.best_params.top_p == 0.95
    assert loaded_checkpoint.best_params.top_k == 60
    assert loaded_checkpoint.best_params.repetition_penalty == 1.2
    assert loaded_checkpoint.best_params.max_tokens == 256
    
    # Verify history
    assert len(loaded_checkpoint.history.entries) == 2
    
    entry1 = loaded_checkpoint.history.entries[0]
    assert entry1.round_num == 1
    assert entry1.train_accuracy == 0.75
    assert entry1.validation_accuracy == 0.70
    assert entry1.params.temperature == 0.8
    
    entry2 = loaded_checkpoint.history.entries[1]
    assert entry2.round_num == 2
    assert entry2.train_accuracy == 0.80
    assert entry2.validation_accuracy == 0.78
    assert entry2.params.temperature == 0.6
    assert entry2.params.top_p == 0.9
    
    # Verify timestamps exist
    assert loaded_checkpoint.created_at is not None
    assert loaded_checkpoint.updated_at is not None


@pytest.mark.asyncio
async def test_evaluate_flow():
    """Test the evaluate() function with known inputs.
    
    Verifies accuracy calculation with various answer patterns.
    """
    # Create test questions
    questions = [
        Question(
            id="q1",
            question="Test question 1",
            choices=["A. Option 1", "B. Option 2", "C. Option 3"],
            answer_key="A",
            original_label="A"
        ),
        Question(
            id="q2",
            question="Test question 2",
            choices=["A. Option 1", "B. Option 2", "C. Option 3"],
            answer_key="B",
            original_label="B"
        ),
        Question(
            id="q3",
            question="Test question 3",
            choices=["A. Option 1", "B. Option 2", "C. Option 3"],
            answer_key="C",
            original_label="C"
        ),
        Question(
            id="q4",
            question="Test question 4",
            choices=["A. Option 1", "B. Option 2", "C. Option 3"],
            answer_key="A",
            original_label="A"
        ),
    ]
    
    # Test case 1: All correct
    raw_outputs_all_correct = [
        "The answer is A",
        "The answer is B",
        "The answer is C",
        "The answer is A",
    ]
    
    result = evaluate(questions, raw_outputs_all_correct)
    assert result.total_count == 4
    assert result.correct_count == 4
    assert result.accuracy == 1.0
    assert len(result.wrong_questions) == 0
    
    # Test case 2: Half correct
    raw_outputs_half_correct = [
        "The answer is A",  # Correct
        "The answer is A",  # Wrong (should be B)
        "The answer is C",  # Correct
        "The answer is B",  # Wrong (should be A)
    ]
    
    result = evaluate(questions, raw_outputs_half_correct)
    assert result.total_count == 4
    assert result.correct_count == 2
    assert result.accuracy == 0.5
    assert len(result.wrong_questions) == 2
    assert result.wrong_questions[0].question_id == "q2"
    assert result.wrong_questions[0].model_answer == "A"
    assert result.wrong_questions[0].correct_answer == "B"
    assert result.wrong_questions[1].question_id == "q4"
    assert result.wrong_questions[1].model_answer == "B"
    assert result.wrong_questions[1].correct_answer == "A"
    
    # Test case 3: All wrong
    raw_outputs_all_wrong = [
        "The answer is B",  # Wrong
        "The answer is C",  # Wrong
        "The answer is A",  # Wrong
        "The answer is C",  # Wrong
    ]
    
    result = evaluate(questions, raw_outputs_all_wrong)
    assert result.total_count == 4
    assert result.correct_count == 0
    assert result.accuracy == 0.0
    assert len(result.wrong_questions) == 4
    
    # Test case 4: Parse failures (unparseable answers)
    raw_outputs_with_failures = [
        "The answer is A",  # Correct
        "I don't know",     # Parse failure
        "The answer is C",  # Correct
        "Maybe it's X",     # Parse failure
    ]
    
    result = evaluate(questions, raw_outputs_with_failures)
    assert result.total_count == 4
    assert result.correct_count == 2  # Only q1 and q3 correct
    assert result.accuracy == 0.5
    assert len(result.wrong_questions) == 2  # q2 and q4 are wrong (parse failures)
    assert result.parse_failure_count == 2
    
    # Verify parse failure details
    wrong_q2 = next(wq for wq in result.wrong_questions if wq.question_id == "q2")
    assert wrong_q2.model_answer is None  # Parse failed
    assert wrong_q2.correct_answer == "B"
    
    wrong_q4 = next(wq for wq in result.wrong_questions if wq.question_id == "q4")
    assert wrong_q4.model_answer is None  # Parse failed
    assert wrong_q4.correct_answer == "A"
