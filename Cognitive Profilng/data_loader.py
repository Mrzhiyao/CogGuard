import pandas as pd
import json
from typing import List, Dict
import os


class EduDataLoader:
    """Edu数据加载器"""
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"文件不存在: {self.csv_path}")
        
        self.data = pd.read_csv(self.csv_path)
        return self.data
    
    def format_for_kg(self, knowledge_points: List[str] = None) -> List[Dict]:
        if self.data is None:
            self.load_data()
        
        formatted_data = []
        
        for idx, row in self.data.iterrows():
            item = {
                "question": row.get("题干", ""),
                "desc": self._generate_desc(row, knowledge_points),
                "program": row.get("错误答案", ""),
                "error_desc": self._generate_error_desc(row),
                "correct_answer": row.get("正确答案", ""),
                "score": row.get("得分", 0),
                "result": row.get("结果", ""),
                "problem_name": row.get("题目", ""),
                "user_id": row.get("用户", ""),
                "timestamp": row.get("时间", ""),
            }
            formatted_data.append(item)
        
        return formatted_data
    
    def _generate_desc(self, row: pd.Series, knowledge_points: List[str] = None) -> str:
        question = row.get("题干", "")
        problem_name = row.get("题目", "")
        
        # 如果有预定义知识点，可以在这里提示LLM优先考虑这些知识点
        desc = f"题目：{problem_name}\n\n{question}"
        
        if knowledge_points:
            desc += f"\n\n参考知识点列表：{', '.join(knowledge_points)}"
        
        return desc
    
    def _generate_error_desc(self, row: pd.Series) -> str:
        result = row.get("结果", "")
        score_raw = row.get("得分", 0)
        # 确保 score 是数字类型（处理 CSV 中可能是字符串的情况）
        try:
            score = float(score_raw) if score_raw else 0
        except (ValueError, TypeError):
            score = 0
        wrong_answer = row.get("错误答案", "")
        correct_answer = row.get("正确答案", "")
        
        error_desc = []
        
        if result == "答案错误" or score == 0:
            error_desc.append("学生的答案不正确")
            if wrong_answer and correct_answer:
                error_desc.append(f"错误答案与正确答案存在差异")
        elif "编译错误" in str(result) or "运行错误" in str(result):
            error_desc.append(f"代码存在{result}")
        elif score > 0 and score < 100:
            error_desc.append(f"部分正确，得分：{score}")
        
        if not error_desc:
            error_desc.append("需要进一步分析代码问题")
        
        return "\n".join([f"{i+1}. {desc}" for i, desc in enumerate(error_desc)])
    
    def get_knowledge_points_from_file(self, file_path: str) -> List[str]:
        if not os.path.exists(file_path):
            return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        knowledge_points = []
        lines = content.strip().split('\n')
        
        for line in lines:
            if ':' in line:
                category, points = line.split(':', 1)
                points_list = [p.strip() for p in points.split() if p.strip()]
                knowledge_points.extend(points_list)
        
        return knowledge_points
    
    def save_formatted_data(self, formatted_data: List[Dict], output_path: str):
        """保存格式化后的数据"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, ensure_ascii=False, indent=2)
        
        print(f"已保存 {len(formatted_data)} 条格式化数据到 {output_path}")

    def format_dataframe(self, dataframe: pd.DataFrame, knowledge_points: List[str] = None) -> List[Dict]:
        """将指定DataFrame格式化为知识图谱构建所需的格式"""
        original_data = self.data
        self.data = dataframe.reset_index(drop=True)
        formatted = self.format_for_kg(knowledge_points)
        self.data = original_data
        return formatted

    def split_by_ratio(self, ratio: float = 0.7):
        """按照时间顺序将数据划分为两部分"""
        if self.data is None:
            self.load_data()
        df = self.data.copy()
        if "时间" in df.columns:
            try:
                df["时间"] = pd.to_datetime(df["时间"])
                df = df.sort_values(by="时间")
            except Exception:
                df = df.reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)
        split_index = int(len(df) * ratio)
        train_df = df.iloc[:split_index].reset_index(drop=True)
        test_df = df.iloc[split_index:].reset_index(drop=True)
        return train_df, test_df


