"""推荐问答必须与当前语料的显式标注对应。"""

from rag_eval.evaluate import EvalItem
from rag_eval.ingest import Document
from rag_eval.recommend import supported_sample_questions


def test_only_questions_with_declared_documents_are_recommended() -> None:
    items = [
        EvalItem(question="安全带怎么佩戴？", ground_truth="...", relevant_doc_ids=["doc-seatbelt"], item_id="q1"),
        EvalItem(question="座椅加热怎么开？", ground_truth="...", relevant_doc_ids=["doc-seat-heat"], item_id="q2"),
    ]
    docs = [Document("doc-seatbelt", "安全带手册", "安全带应贴紧髋部。")]
    result = supported_sample_questions(items, docs)
    assert [item.item_id for item in result] == ["q1"]


def test_no_matching_documents_means_no_recommendations() -> None:
    items = [EvalItem(question="安全带怎么佩戴？", ground_truth="...", relevant_doc_ids=["doc-seatbelt"])]
    docs = [Document("web-battery", "电池博客", "磷酸铁锂电池安全性较高。")]
    assert supported_sample_questions(items, docs) == []
