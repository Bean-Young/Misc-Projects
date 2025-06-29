# admin_review.py
from models import Evaluation, AdminFeedback
from utils import validate_evaluation_content

class AdminReviewSystem:
    def preliminary_review(self, evaluation):
        """初步审核 (PAD图: 初步审核)"""
        # 1. 检查评价质量
        if not self.is_high_quality(evaluation):
            evaluation.status = Evaluation.STATUS['admin_reject']
            return False, "质量不合格"
        
        # 2. 检查材料真实性
        if evaluation.materials and not self.verify_materials(evaluation):
            evaluation.status = Evaluation.STATUS['abnormal']
            return False, "材料验证失败"
        
        # 3. 筛查精选评价
        if self.is_featured_material(evaluation):
            self.screen_featured(evaluation)
            return True, "精选评价"
        
        evaluation.status = Evaluation.STATUS['approved']
        return True, "审核通过"
    
    def is_high_quality(self, evaluation):
        """评价质量检查 (伪实现)"""
        # 1. 检查评价长度
        if len(evaluation.content) < 50:
            return False
        
        # 2. 检查具体性
        if "很好" in evaluation.content or "非常好" in evaluation.content:
            return False
        
        # 3. 检查语言质量
        if any(word in evaluation.content for word in ["垃圾", "废物", "去死"]):
            return False
        
        return True
    
    def verify_materials(self, evaluation):
        """验证材料真实性 (伪实现)"""
        # 在实际系统中会进行图像/视频验证
        return True
    
    def is_featured_material(self, evaluation):
        """是否精选材料 (伪实现)"""
        # 1. 检查材料数量
        if len(evaluation.materials) >= 2:
            return True
        
        # 2. 检查评价详细程度
        if len(evaluation.content) > 200:
            return True
        
        return False
    
    def screen_featured(self, evaluation):
        """筛查精选 (PAD图: 筛查精选)"""
        # 1. 更新评价状态
        evaluation.status = Evaluation.STATUS['featured']
        
        # 2. 创建主管反馈记录
        admin_id = "G00001"  # 假设当前主管ID
        feedback = AdminFeedback(
            evaluation.evaluation_id,
            admin_id,
            AdminFeedback.FEEDBACK['feature']
        )
        
        # 3. 存储到主管反馈表(D3)
        # db.session.add(feedback)
        # db.session.commit()
        
        print(f"评价 {evaluation.evaluation_id} 已被标记为精选")
    
    def process_admin_feedback(self, evaluation_id, admin_id, feedback_status):
        """处理主管反馈 (完整流程)"""
        # 1. 获取评价
        # evaluation = Evaluation.query.get(evaluation_id)
        evaluation = Evaluation(evaluation_id, "T001", "评价内容", "good")  # 示例数据
        
        # 2. 创建反馈记录
        feedback = AdminFeedback(evaluation_id, admin_id, feedback_status)
        # db.session.add(feedback)
        
        # 3. 更新评价状态
        if feedback_status == AdminFeedback.FEEDBACK['feature']:
            evaluation.status = Evaluation.STATUS['featured']
        elif feedback_status == AdminFeedback.FEEDBACK['approve']:
            evaluation.status = Evaluation.STATUS['approved']
        else:
            evaluation.status = Evaluation.STATUS['admin_reject']
        
        # 4. 保存更改
        # db.session.commit()
        
        return evaluation.status