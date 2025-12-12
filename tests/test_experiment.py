import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.experiment import run_batch_experiment

class TestExperiment(unittest.TestCase):

    @patch('src.utils.experiment.load_document')
    @patch('src.utils.experiment.split_documents')
    @patch('src.utils.experiment.create_vectorstore')
    @patch('src.utils.experiment.get_llm')
    @patch('src.utils.experiment.get_rag_chain')
    @patch('src.utils.experiment.get_judge_chain')
    def test_run_batch_experiment(self, mock_judge, mock_rag, mock_llm, mock_create_vs, mock_split, mock_load):
        # Setup Mocks
        mock_load.return_value = ["doc"]
        mock_split.return_value = ["chunk1", "chunk2"]
        mock_vs = MagicMock()
        mock_create_vs.return_value = mock_vs
        
        mock_rag_chain_instance = MagicMock()
        mock_rag_chain_instance.invoke.return_value = {
            "answer": "Test Answer",
            "context": [MagicMock(page_content="ctx")]
        }
        mock_rag.return_value = mock_rag_chain_instance
        
        mock_judge_chain_instance = MagicMock()
        mock_judge_chain_instance.invoke.return_value = {
            "accuracy": 10,
            "faithfulness": 10,
            "relevance": 10,
            "explanation": "Great"
        }
        mock_judge.return_value = mock_judge_chain_instance
        
        # Config
        config = {
            "model_name": "TestModel",
            "judge_model": "TestJudge",
            "temperature": 0.7,
            "top_p": 0.9,
            "chunk_sizes": [1000],
            "chunk_overlaps": [100],
            "k_retrievals": [3]
        }
        
        # Run
        results = run_batch_experiment("test.pdf", config, "Question?")
        
        # Assertions
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["Chunk Size"], 1000)
        self.assertEqual(results[0]["Top-K"], 3)
        self.assertEqual(results[0]["Answer"], "Test Answer")
        
        # Verify calls
        mock_create_vs.assert_called_once() # Should be called once per chunk config
        mock_rag_chain_instance.invoke.assert_called_once()

    @patch('src.utils.experiment.load_document')
    @patch('src.utils.experiment.split_documents')
    @patch('src.utils.experiment.create_vectorstore')
    @patch('src.utils.experiment.get_llm')
    @patch('src.utils.experiment.get_rag_chain')
    @patch('src.utils.experiment.get_judge_chain')
    def test_run_batch_experiment_multiple(self, mock_judge, mock_rag, mock_llm, mock_create_vs, mock_split, mock_load):
        # Setup Mocks
        mock_load.return_value = ["doc"]
        mock_split.return_value = ["chunk"]
        
        # Mock retrieval response
        mock_rag_chain_instance = MagicMock()
        mock_rag_chain_instance.invoke.return_value = {"answer": "A", "context": []}
        mock_rag.return_value = mock_rag_chain_instance
        
        mock_judge_chain_instance = MagicMock()
        mock_judge_chain_instance.invoke.return_value = {}
        mock_judge.return_value = mock_judge_chain_instance
        
        # Config with 2 chunk sizes, 1 overlap, 2 K values = 4 runs
        config = {
            "model_name": "TestModel",
            "judge_model": "TestJudge",
            "temperature": 0.7,
            "top_p": 0.9,
            "chunk_sizes": [500, 1000],
            "chunk_overlaps": [50],
            "k_retrievals": [1, 3]
        }
        
        # Run
        results = run_batch_experiment("test.pdf", config, "Q")
        
        # Assertions
        self.assertEqual(len(results), 4)
        self.assertEqual(mock_create_vs.call_count, 2) # Once for 500, once for 1000

if __name__ == '__main__':
    unittest.main()
