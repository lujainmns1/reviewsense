from app import app, db
from models import User, AnalysisSession, Review, ModelResult, Topic

def test_database():
    with app.app_context():
        # Create tables
        db.create_all()
        
        # Create a test user
        test_user = User(email='test@example.com')
        test_user.set_password('password123')
        db.session.add(test_user)
        db.session.commit()
        
        # Create a test session
        test_session = AnalysisSession(
            user_id=test_user.id,
            country_code='US'
        )
        db.session.add(test_session)
        db.session.commit()
        
        # Create a test review
        test_review = Review(
            session_id=test_session.id,
            review_text='This is a test review',
            language='en'
        )
        db.session.add(test_review)
        db.session.commit()
        
        # Create a test model result
        test_result = ModelResult(
            session_id=test_session.id,
            review_id=test_review.id,
            model_name='test_model',
            sentiment='positive',
            sentiment_score=0.9
        )
        db.session.add(test_result)
        db.session.commit()
        
        # Create a test topic
        test_topic = Topic(
            review_id=test_review.id,
            topic_text='test topic',
            score=0.8
        )
        db.session.add(test_topic)
        db.session.commit()
        
        print("Database test completed successfully!")
        
if __name__ == '__main__':
    test_database()