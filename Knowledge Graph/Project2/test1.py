import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# 数据准备 - 生成模拟数据
def generate_data(num_samples=1000):
    np.random.seed(42)
    temperatures = np.random.uniform(-5, 40, num_samples)  # 温度范围：-5°C 到 40°C
    rain_probs = np.random.uniform(0, 100, num_samples)    # 降水概率范围：0% 到 100%
    
    # 根据规则生成衣物标签
    labels = []
    for temp, rain in zip(temperatures, rain_probs):
        # 根据温度选择衣物
        if temp < 0:
            clothing = '厚羽绒服、毛衣、保暖裤、围巾、手套、帽子'
        elif 0 <= temp < 10:
            clothing = '厚外套、毛衣、长裤、围巾、手套'
        elif 10 <= temp < 16:
            clothing = '外套、毛衣、长裤'
        elif 16 <= temp < 21:
            clothing = '薄外套、长袖衬衫、长裤'
        elif 21 <= temp < 26:
            clothing = '长袖T恤、长裤或裙子'
        elif 26 <= temp < 31:
            clothing = '短袖T恤、短裤或裙子'
        elif 31 <= temp < 36:
            clothing = '短袖T恤、短裤、凉鞋'
        else:
            clothing = '短袖T恤、短裤、凉鞋、太阳镜'
        
        # 根据降水概率添加雨具
        if rain > 50:
            clothing += ' + 雨衣或加固伞'
        elif rain > 20:
            clothing += ' + 折叠伞'
        
        labels.append(clothing)
    
    return np.column_stack((temperatures, rain_probs)), np.array(labels)

# 机器学习模型训练
def train_model(X, y):
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 创建并训练决策树模型
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # 评估模型
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"模型训练完成 - 训练集准确率: {train_acc:.2f}, 测试集准确率: {test_acc:.2f}")
    return model

# 产生式规则系统
def apply_rules(temperature, rain_prob, model_recommendation):
    final_recommendation = model_recommendation
    
    # 规则1: 极寒天气特殊处理
    if temperature < -10:
        final_recommendation = "极厚羽绒服、保暖内衣、加绒裤、防寒手套、雪地靴、防寒面罩"
    
    # 规则2: 高温天气强制添加太阳镜
    elif temperature >= 35 and "太阳镜" not in final_recommendation:
        final_recommendation += " + 太阳镜"
    
    # 规则3: 暴雨天气特殊处理
    elif rain_prob > 70 and "雨衣" not in final_recommendation:
        if "加固伞" in final_recommendation:
            final_recommendation = final_recommendation.replace("加固伞", "雨衣")
        else:
            final_recommendation += " + 雨衣"
    
    # 规则4: 寒冷雨天特殊处理
    elif temperature < 10 and rain_prob > 50 and "防水" not in final_recommendation:
        final_recommendation = final_recommendation.replace("外套", "防水外套")
    
    return final_recommendation

# 集成推荐系统
def recommend_clothing(temperature, rain_prob, model):
    # 使用机器学习模型进行初步预测
    input_data = np.array([[temperature, rain_prob]])
    model_recommendation = model.predict(input_data)[0]
    
    # 应用产生式规则进行调整
    final_recommendation = apply_rules(temperature, rain_prob, model_recommendation)
    
    return {
        "temperature": temperature,
        "rain_probability": rain_prob,
        "model_recommendation": model_recommendation,
        "final_recommendation": final_recommendation
    }

# 主程序
if __name__ == "__main__":
    # 1. 生成并准备数据
    X, y = generate_data(2000)
    
    # 2. 训练机器学习模型
    model = train_model(X, y)
    
    # 3. 测试推荐系统
    test_cases = [
        (38, 5),    # 高温晴天
        (15, 60),   # 凉爽雨天
        (-5, 30),   # 寒冷阴天
        (25, 80),   # 温暖暴雨
        (-15, 10)   # 极寒天气
    ]
    
    print("\n衣物推荐测试:")
    for temp, rain_prob in test_cases:
        result = recommend_clothing(temp, rain_prob, model)
        print(f"\n预测天气: {temp}°C, 降水概率 {rain_prob}%")
        print(f"模型推荐: {result['model_recommendation']}")
        print(f"最终推荐: {result['final_recommendation']}")
        print("-" * 60)