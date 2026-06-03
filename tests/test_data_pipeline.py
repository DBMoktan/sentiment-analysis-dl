import unittest
from src.data_pipeline import NepaliTextNormalizer

class TestNepaliTextNormalizer(unittest.TestCase):
    def setUp(self):
        self.normalizer = NepaliTextNormalizer()

    def test_url_and_mention_removal(self):
        text = "Check this out https://example.com @user_name test"
        expected = "check this out test"
        self.assertEqual(self.normalizer.normalize_text(text), expected)

    def test_devanagari_numbers_and_text(self):
        text = "यो १२३४ ५६७८ ९० phone"
        expected = "यो १२३४ ५६७८ ९० phone"
        self.assertEqual(self.normalizer.normalize_text(text), expected)

    def test_emoji_and_punctuation_removal(self):
        text = "Yo phone danger babal chha! 😍🔥 #best-phone !!!"
        expected = "yo phone danger babal chha best phone"
        self.assertEqual(self.normalizer.normalize_text(text), expected)

    def test_spaces_collapsing(self):
        text = "  yo    phone   babal  chha   "
        expected = "yo phone babal chha"
        self.assertEqual(self.normalizer.normalize_text(text), expected)

if __name__ == "__main__":
    unittest.main()
