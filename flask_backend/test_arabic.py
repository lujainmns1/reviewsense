#!/usr/bin/env python3
# test_arabic.py - Test script for Arabic sentiment analysis

from arabic_model_service import analyze_arabic_review, clean_arabic_text

def test_arabic_sentiment():
    """Test the Arabic sentiment analysis with sample reviews"""

    # Sample Arabic reviews for testing
    test_reviews = [
        "المنتج ممتاز جداً وأنا راضي عنه تماماً",  # Excellent product, very satisfied
        "خدمة العملاء سيئة والمنتج لم يعجبني",  # Bad customer service, didn't like the product
        "المنتج متوسط ولا يستحق الشراء",  # Average product, not worth buying
        "شكراً لكم على الخدمة الجيدة",  # Thank you for the good service
        "سعر مرتفع جداً مقارنة بالجودة",  # Price is too high compared to quality
        "",  # Empty string
        "123456",  # Numbers only
        "Hello world"  # English text
    ]

    print("Testing Arabic Sentiment Analysis")
    print("=" * 50)

    for i, review in enumerate(test_reviews, 1):
        print(f"\nTest {i}:")
        print(f"Original: '{review}'")

        # Test cleaning
        cleaned = clean_arabic_text(review)
        print(f"Cleaned: '{cleaned}'")

        # Test analysis
        result = analyze_arabic_review(review)
        print(f"Sentiment: {result['label']} (confidence: {result['score']:.3f})")

    print("\nTesting completed!")

if __name__ == "__main__":
    test_arabic_sentiment()

 
