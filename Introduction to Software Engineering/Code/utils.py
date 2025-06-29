# utils.py
import re
import random
import string

def validate_user_id(user_id):
    """验证用户ID格式: [小写字母|除特殊外字母|数字]+4{大小写字母和数字组合}9"""
    pattern = r'^[a-z0-9A-Z]{4}[a-zA-Z0-9]{9}$'
    return re.match(pattern, user_id) is not None

def validate_password(password):
    """验证密码格式: 必须包含特殊字符、大写字母、小写字母、数字各1个，总长16位"""
    if len(password) != 16:
        return False
    if not re.search(r'[@*&<>]', password):
        return False
    if not re.search(r'[A-Z]', password):
        return False
    if not re.search(r'[a-z]', password):
        return False
    if not re.search(r'[0-9]', password):
        return False
    return True

def validate_evaluation_id(eval_id):
    """验证评价ID格式: PJ+4大写字母+8数字"""
    pattern = r'^PJ[A-Z]{4}\d{8}$'
    return re.match(pattern, eval_id) is not None

def validate_teacher_id(teacher_id):
    """验证教师ID格式: T+1大写字母+3数字"""
    pattern = r'^T[A-Z]\d{3}$'
    return re.match(pattern, teacher_id) is not None

def validate_teacher_name(name):
    """验证教师姓名: 1-10个中文字符"""
    return 1 <= len(name) <= 10 and all('\u4e00' <= char <= '\u9fff' for char in name)

def validate_evaluation_content(content):
    """验证评价内容: 10-1000个中文字符"""
    return 10 <= len(content) <= 1000 and all('\u4e00' <= char <= '\u9fff' for char in content)

def validate_material_path(path):
    """验证材料路径格式: ./material/[10大小写字母].(pdf|mp4|mp3)"""
    pattern = r'^\./material/[a-zA-Z]{10}\.(pdf|mp4|mp3)$'
    return re.match(pattern, path) is not None

def generate_evaluation_id():
    """生成评价ID: PJAAAA00000001格式"""
    letters = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSUVWXYZ', k=4))
    numbers = ''.join(random.choices(string.digits, k=8))
    return f'PJ{letters}{numbers}'

def map_rating_to_score(rating):
    """将评价等第映射为分数"""
    mapping = {
        'very_poor': 1,
        'poor': 2,
        'medium': 3,
        'good': 4,
        'very_good': 5
    }
    return mapping.get(rating, 3)