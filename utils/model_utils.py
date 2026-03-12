#!/usr/bin/env python3
"""
模型工具

功能：
- 模型保存/加载
- 模型评估
- 预测结果处理
"""

import pickle
import json
from datetime import datetime
from pathlib import Path


class ModelUtils:
    """模型工具类"""
    
    @staticmethod
    def save_model(model, model_name, model_dir='model', metadata=None):
        """
        保存模型
        
        Args:
            model: 训练好的模型
            model_name: 模型名称
            model_dir: 模型保存目录
            metadata: 模型元数据（训练参数、性能等）
        """
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # 保存模型文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_file = model_path / f"{model_name}_{timestamp}.pkl"
        
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        
        # 保存元数据
        if metadata:
            meta_file = model_path / f"{model_name}_{timestamp}_meta.json"
            metadata['saved_at'] = timestamp
            metadata['model_file'] = str(model_file.name)
            
            with open(meta_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"模型已保存: {model_file}")
        return model_file
    
    @staticmethod
    def load_model(model_file):
        """
        加载模型
        
        Args:
            model_file: 模型文件路径
            
        Returns:
            加载的模型
        """
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        print(f"模型已加载: {model_file}")
        return model
    
    @staticmethod
    def list_models(model_dir='model'):
        """列出所有保存的模型"""
        model_path = Path(model_dir)
        if not model_path.exists():
            return []
        
        models = []
        for pkl_file in model_path.glob('*.pkl'):
            meta_file = pkl_file.with_suffix('').with_name(pkl_file.stem + '_meta.json')
            model_info = {
                'file': str(pkl_file),
                'name': pkl_file.stem,
                'created': pkl_file.stat().st_mtime
            }
            
            if meta_file.exists():
                with open(meta_file) as f:
                    model_info['metadata'] = json.load(f)
            
            models.append(model_info)
        
        return sorted(models, key=lambda x: x['created'], reverse=True)


if __name__ == '__main__':
    # 测试
    import sklearn.ensemble as ensemble
    
    # 创建一个简单模型
    model = ensemble.RandomForestRegressor(n_estimators=10)
    
    # 保存
    ModelUtils.save_model(
        model, 
        'test_model',
        metadata={
            'model_type': 'RandomForest',
            'n_estimators': 10,
            'description': '测试模型'
        }
    )
    
    # 列出模型
    models = ModelUtils.list_models()
    print(f"\n共有 {len(models)} 个模型")
    for m in models:
        print(f"  - {m['name']}")
