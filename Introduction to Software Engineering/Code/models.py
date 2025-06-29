# models.py
class User:
    def __init__(self, user_id, password, role):
        self.user_id = user_id
        self.password = password
        self.role = role  # 'student', 'reviewer', 'admin'
        
    def is_admin(self):
        return self.role == 'admin'
    
    def is_student(self):
        return self.role == 'student'
    
    def is_reviewer(self):
        return self.role == 'reviewer'

class Teacher:
    def __init__(self, teacher_id, name):
        self.teacher_id = teacher_id
        self.name = name

class Evaluation:
    STATUS = {
        'abnormal': 0,
        'unsubmitted': 1,
        'pending_review': 2,
        'student_reject': 3,
        'admin_reject': 4,
        'approved': 5,
        'featured': 6
    }
    
    RATING = {
        'very_poor': 1,
        'poor': 2,
        'medium': 3,
        'good': 4,
        'very_good': 5
    }
    
    def __init__(self, evaluation_id, teacher_id, content, rating, materials=None):
        self.evaluation_id = evaluation_id
        self.teacher_id = teacher_id
        self.content = content
        self.rating = rating
        self.materials = materials or []
        self.status = self.STATUS['unsubmitted']
        self.review_score = 0.0

class ReviewFeedback:
    def __init__(self, evaluation_id, reviewer_id, score):
        self.evaluation_id = evaluation_id
        self.reviewer_id = reviewer_id
        self.score = score

class AdminFeedback:
    FEEDBACK = {
        'reject': 0,
        'approve': 1,
        'feature': 2
    }
    
    def __init__(self, evaluation_id, admin_id, feedback_status):
        self.evaluation_id = evaluation_id
        self.admin_id = admin_id
        self.feedback_status = feedback_status