# review_system.py
from models import ReviewFeedback, Evaluation
from utils import map_rating_to_score

class ReviewSystem:
    def __init__(self, threshold=3.5, required_reviews=5):
        self.threshold = threshold  # 基准线
        self.required_reviews = required_reviews  # 所需审查人数
        
    def submit_review(self, evaluation_id, reviewer_id, rating):
        """审查反馈提交 (PAD图: 审查反馈提交)"""
        # 伪代码实现
        # 1. 验证评价ID格式
        # 2. 验证审查员ID格式
        # 3. 将评价等第映射为分数
        score = map_rating_to_score(rating)
        
        # 4. 创建审查反馈记录
        feedback = ReviewFeedback(evaluation_id, reviewer_id, score)
        
        # 5. 存储到审查反馈表(D2)
        # db.session.add(feedback)
        # db.session.commit()
        
        # 6. 检查是否达到审查要求
        feedback_count = self.get_feedback_count(evaluation_id)
        if feedback_count >= self.required_reviews:
            return self.calculate_review_result(evaluation_id)
        
        return {"status": "pending", "feedback_count": feedback_count}
    
    def get_feedback_count(self, evaluation_id):
        """获取某评价的反馈数量 (伪实现)"""
        # 实际查询数据库: SELECT COUNT(*) FROM review_feedback WHERE evaluation_id = ?
        return 3  # 示例值
    
    def calculate_review_result(self, evaluation_id):
        """计算审查结果 (PAD图: 计算审查结果)"""
        # 1. 查询所有反馈
        # feedbacks = ReviewFeedback.query.filter_by(evaluation_id=evaluation_id).all()
        feedbacks = [ReviewFeedback(evaluation_id, f"reviewer{i}", 4.0) for i in range(5)]  # 示例数据
        
        # 2. 计算平均分
        total_score = sum(fb.score for fb in feedbacks)
        avg_score = total_score / len(feedbacks)
        
        # 3. 查找对应的评价
        # evaluation = Evaluation.query.get(evaluation_id)
        evaluation = Evaluation(evaluation_id, "T001", "评价内容", "good")  # 示例数据
        
        # 4. 比较基准线
        if avg_score >= self.threshold:
            evaluation.status = Evaluation.STATUS['approved']
            result = "approved"
        else:
            evaluation.status = Evaluation.STATUS['student_reject']
            result = "rejected"
        
        # 5. 更新评价状态
        self.update_evaluation_info(evaluation)
        
        return {
            "status": result,
            "avg_score": avg_score,
            "evaluation_id": evaluation_id
        }
    
    def update_evaluation_info(self, evaluation):
        """更新评价信息 (PAD图: 更新评价信息)"""
        # 伪代码实现
        # 1. 更新评价表(D1)中的状态码
        # 2. 记录审查平均分
        # 3. 如果通过审查，创建审核申请记录
        print(f"更新评价 {evaluation.evaluation_id} 状态为 {evaluation.status}")
        
        # 如果状态为approved，触发主管审核
        if evaluation.status == Evaluation.STATUS['approved']:
            self.trigger_admin_review(evaluation.evaluation_id)
    
    def trigger_admin_review(self, evaluation_id):
        """触发主管审核流程 (伪实现)"""
        print(f"评价 {evaluation_id} 已通过审查，等待主管审核")
        # 在实际系统中会创建审核申请记录