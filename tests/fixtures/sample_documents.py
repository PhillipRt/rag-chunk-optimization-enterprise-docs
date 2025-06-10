"""
Sample documents for testing retrievers and other components.
"""

SAMPLE_DOCUMENTS = [
    {
        "content": "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically-typed and garbage-collected.",
        "metadata": {
            "source": "test_doc_1",
            "title": "Python Language",
            "category": "programming"
        }
    },
    {
        "content": "JavaScript, often abbreviated as JS, is a programming language that is one of the core technologies of the World Wide Web, alongside HTML and CSS. 98% of websites use JavaScript on the client side for webpage behavior.",
        "metadata": {
            "source": "test_doc_2",
            "title": "JavaScript Language",
            "category": "programming"
        }
    },
    {
        "content": "Machine learning (ML) is a field of inquiry devoted to understanding and building methods that 'learn', that is, methods that leverage data to improve performance on some set of tasks.",
        "metadata": {
            "source": "test_doc_3",
            "title": "Machine Learning",
            "category": "artificial_intelligence"
        }
    },
    {
        "content": "Retrieval-Augmented Generation (RAG) is an AI framework that enhances large language model (LLM) outputs by incorporating relevant information retrieved from external sources.",
        "metadata": {
            "source": "test_doc_4",
            "title": "RAG Framework",
            "category": "artificial_intelligence"
        }
    },
    {
        "content": "A knowledge graph is a knowledge base that uses a graph-structured data model or topology to integrate data. Knowledge graphs are often used to store interlinked descriptions of entities with free-form semantics.",
        "metadata": {
            "source": "test_doc_5",
            "title": "Knowledge Graphs",
            "category": "data_structures"
        }
    }
]

SAMPLE_QUERIES = [
    "What is Python programming language?",
    "How does JavaScript relate to web development?",
    "Explain machine learning in simple terms",
    "What are the components of a RAG system?",
    "How do knowledge graphs store information?"
] 