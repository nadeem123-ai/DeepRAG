"""
test_cases.py
-------------
Ground truth question-answer pairs for evaluating the RAG pipeline.
These are based on the resume document content.

To add more test cases, just add more dicts to the TEST_CASES list.
Each case needs:
    - question:        What you ask the pipeline
    - expected_answer: The correct answer you expect
    - category:        Topic area (for grouping results)
"""

TEST_CASES = [
    {
        "question": "What is Muhammad Nadeem's email address?",
        "expected_answer": "nadeemfarooq317@gmail.com",
        "category": "Personal Info",
    },
    {
        "question": "What is Muhammad Nadeem's current job title?",
        "expected_answer": "Machine Learning Engineer at GrowSofTec",
        "category": "Experience",
    },
    {
        "question": "What deep learning frameworks does Muhammad Nadeem know?",
        "expected_answer": "PyTorch and TensorFlow",
        "category": "Technical Skills",
    },
    {
        "question": "What vector databases has Muhammad Nadeem worked with?",
        "expected_answer": "FAISS and Pinecone",
        "category": "Technical Skills",
    },
    {
        "question": "Where did Muhammad Nadeem complete his M.Phil?",
        "expected_answer": "University of the Punjab, Lahore, Pakistan",
        "category": "Education",
    },
    {
        "question": "What object detection model did Muhammad Nadeem use at INTECH?",
        "expected_answer": "YOLOv8",
        "category": "Experience",
    },
    {
        "question": "What optimization techniques did Muhammad Nadeem experiment with?",
        "expected_answer": "TensorRT, quantization, and LoRA fine-tuning",
        "category": "Technical Skills",
    },
    {
        "question": "What was Muhammad Nadeem's role at University of the Punjab?",
        "expected_answer": "Machine Learning Researcher",
        "category": "Experience",
    },
    {
        "question": "Which IBM certification does Muhammad Nadeem have?",
        "expected_answer": "Generative AI: Introduction and Applications",
        "category": "Certifications",
    },
    {
        "question": "What workflow automation tool has Muhammad Nadeem used?",
        "expected_answer": "n8n",
        "category": "Technical Skills",
    },
]