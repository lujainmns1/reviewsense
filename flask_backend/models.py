from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from datetime import datetime

db = SQLAlchemy()
bcrypt = Bcrypt()

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow)
    sessions = db.relationship('AnalysisSession', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)

class AnalysisSession(db.Model):
    __tablename__ = 'analysis_sessions'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    name = db.Column(db.String(255), nullable=True)
    country_code = db.Column(db.String(2))
    detected_dialect = db.Column(db.String(50))
    created_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow)
    reviews = db.relationship('Review', backref='session', lazy=True)

    # Add model results relationship
    model_results = db.relationship('ModelResult', backref='session', lazy=True)

class Review(db.Model):
    __tablename__ = 'reviews'
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('analysis_sessions.id'), nullable=False)
    review_text = db.Column(db.Text, nullable=False)
    cleaned_text = db.Column(db.Text)
    language = db.Column(db.String(10))
    created_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow)
    
    # Analysis results for this review
    model_results = db.relationship('ModelResult', backref='review', lazy=True)
    topics = db.relationship('Topic', backref='review', lazy=True)

class ModelResult(db.Model):
    __tablename__ = 'model_results'
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('analysis_sessions.id'), nullable=False)
    review_id = db.Column(db.Integer, db.ForeignKey('reviews.id'), nullable=False)
    model_name = db.Column(db.String(100), nullable=False)
    sentiment = db.Column(db.String(20), nullable=False)
    sentiment_score = db.Column(db.Float)
    created_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow)

class Topic(db.Model):
    __tablename__ = 'topics'
    id = db.Column(db.Integer, primary_key=True)
    review_id = db.Column(db.Integer, db.ForeignKey('reviews.id'), nullable=False)
    topic_text = db.Column(db.String(255), nullable=False)
    score = db.Column(db.Float)
    created_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow)

# Create indexes
def create_indexes():
    db.Index('idx_reviews_session', Review.session_id)
    db.Index('idx_model_results_review', ModelResult.review_id)
    db.Index('idx_model_results_session', ModelResult.session_id)
    db.Index('idx_topics_review', Topic.review_id)
    db.Index('idx_sessions_user', AnalysisSession.user_id)