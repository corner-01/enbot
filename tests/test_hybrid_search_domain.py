import unittest

from rag.hybrid_search import HybridSearchEngine


class HybridSearchDomainTests(unittest.TestCase):
    def test_envalt_brand_and_platform_questions_are_in_domain(self):
        queries = (
            "EnValt是什么公司？",
            "无锡英弗特的主营业务是什么？",
            "iEnOS是什么平台？",
            "myEnOS面向哪些用户？",
        )

        for query in queries:
            with self.subTest(query=query):
                self.assertTrue(HybridSearchEngine._is_in_domain(query))

    def test_unrelated_company_question_remains_out_of_domain(self):
        self.assertFalse(HybridSearchEngine._is_in_domain("苹果公司是什么？"))


if __name__ == "__main__":
    unittest.main()
