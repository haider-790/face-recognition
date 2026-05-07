from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()


class Student(db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    student_id    = db.Column(db.String(50), unique=True, nullable=False)
    name          = db.Column(db.String(100), nullable=False)
    registered_at = db.Column(db.DateTime, default=datetime.utcnow)


class Attendance(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(50), nullable=False)
    name       = db.Column(db.String(100), nullable=False)
    subject    = db.Column(db.String(100), nullable=False)
    date       = db.Column(db.String(20), nullable=False)
    time       = db.Column(db.String(20), nullable=False)
    status     = db.Column(db.String(20), default='Present')