# app.py
from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from models import User, Teacher, Evaluation
from review_system import ReviewSystem
from admin_review import AdminReviewSystem
from utils import *
import random

app = Flask(__name__)
app.secret_key = 'secure_secret_key'
login_manager = LoginManager(app)

# 伪数据库
users_db = {
    'stu001': User('stu001', 'Pass@1234word567', 'student'),
    'rev001': User('rev001', 'Rev@1234word567', 'reviewer'),
    'adm001': User('G00001', 'Admin@1234pass', 'admin')
}

teachers_db = [
    Teacher('TA001', '张教授'),
    Teacher('TB002', '李教授'),
    Teacher('TC003', '王教授')
]

evaluations_db = []

review_system = ReviewSystem()
admin_review = AdminReviewSystem()

@login_manager.user_loader
def load_user(user_id):
    return users_db.get(user_id)

# ====================== 用户端Web界面模块 ======================
@app.route('/login', methods=['GET', 'POST'])
def login():
    """登录验证 (PAD图: 登录验证)"""
    if request.method == 'POST':
        user_id = request.form['user_id']
        password = request.form['password']
        
        user = users_db.get(user_id)
        if user and user.password == password:
            login_user(user)
            return redirect(url_for('index'))
        return jsonify({'status': 'error', 'message': '用户名或密码错误'})
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    if current_user.is_student():
        return redirect(url_for('submit_evaluation'))
    elif current_user.is_reviewer():
        return redirect(url_for('review_evaluations'))
    elif current_user.is_admin():
        return redirect(url_for('admin_review'))
    return "欢迎页面"

@app.route('/evaluations')
@login_required
def show_evaluations():
    """评价展示 (PAD图: 评价展示)"""
    # 1. 获取精选评价 (状态码6)
    featured_evals = [e for e in evaluations_db if e.status == Evaluation.STATUS['featured']]
    
    # 2. 渲染模板
    return render_template('evaluations.html', evaluations=featured_evals)

# ====================== 学生评价模块 ======================
@app.route('/submit_evaluation', methods=['GET', 'POST'])
@login_required
def submit_evaluation():
    if not current_user.is_student():
        return "无权限", 403
    
    if request.method == 'POST':
        # 1. 获取表单数据
        teacher_id = request.form['teacher_id']
        content = request.form['content']
        rating = request.form['rating']
        
        # 2. 验证数据
        if not validate_teacher_id(teacher_id):
            return "教师ID格式错误", 400
        if not validate_evaluation_content(content):
            return "评价内容格式错误", 400
        
        # 3. 生成评价ID
        eval_id = generate_evaluation_id()
        
        # 4. 创建评价对象
        materials = request.form.getlist('materials')
        valid_materials = [m for m in materials if validate_material_path(m)]
        
        evaluation = Evaluation(
            eval_id,
            teacher_id,
            content,
            rating,
            valid_materials
        )
        evaluation.status = Evaluation.STATUS['pending_review']
        
        # 5. 保存评价
        evaluations_db.append(evaluation)
        return jsonify({
            'status': 'success',
            'evaluation_id': eval_id
        })
    
    # GET请求：显示评价提交表单
    return render_template('submit_evaluation.html', teachers=teachers_db)

# ====================== 审查评价模块 ======================
@app.route('/review', methods=['POST'])
@login_required
def submit_review():
    """审查反馈提交接口"""
    if not current_user.is_reviewer():
        return "无权限", 403
    
    data = request.json
    evaluation_id = data.get('evaluation_id')
    rating = data.get('rating')
    
    if not validate_evaluation_id(evaluation_id):
        return "评价ID格式错误", 400
    
    result = review_system.submit_review(
        evaluation_id,
        current_user.user_id,
        rating
    )
    
    return jsonify(result)

# ====================== 主管审核模块 ======================
@app.route('/admin/review', methods=['POST'])
@login_required
def admin_review():
    if not current_user.is_admin():
        return "无权限", 403
    
    data = request.json
    evaluation_id = data.get('evaluation_id')
    feedback_status = data.get('feedback_status')
    
    if not validate_evaluation_id(evaluation_id):
        return "评价ID格式错误", 400
    
    # 处理主管反馈
    status = admin_review.process_admin_feedback(
        evaluation_id,
        current_user.user_id,
        feedback_status
    )
    
    return jsonify({
        'status': 'success',
        'new_status': status
    })

if __name__ == '__main__':
    app.run(debug=True)