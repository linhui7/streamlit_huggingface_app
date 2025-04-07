import unittest
from unittest.mock import patch, MagicMock
from model import HuggingFaceModel


class TestHuggingFaceModel(unittest.TestCase):

    def test_init(self):
        """Test model initialization"""
        model = HuggingFaceModel()
        self.assertEqual(model.model_name, "HuggingFaceM4/tiny-random-LlamaForCausalLM")
        self.assertFalse(model.initialized)
        self.assertIsNone(model.model)
        self.assertIsNone(model.tokenizer)

    @patch("model.AutoTokenizer.from_pretrained")
    @patch("model.AutoModelForCausalLM.from_pretrained")
    def test_load_model_success(self, mock_model, mock_tokenizer):
        """Test successful model loading"""
        mock_tokenizer.return_value = MagicMock()
        mock_model.return_value = MagicMock()

        model = HuggingFaceModel()
        result = model.load_model()

        self.assertTrue(result)
        self.assertTrue(model.initialized)
        mock_tokenizer.assert_called_once_with(
            "HuggingFaceM4/tiny-random-LlamaForCausalLM"
        )
        mock_model.assert_called_once()

    @patch("model.AutoTokenizer.from_pretrained")
    def test_load_model_failure(self, mock_tokenizer):
        """Test model loading failure"""
        mock_tokenizer.side_effect = Exception("Test error")

        model = HuggingFaceModel()
        result = model.load_model()

        self.assertFalse(result)
        self.assertFalse(model.initialized)


if __name__ == "__main__":
    unittest.main()
