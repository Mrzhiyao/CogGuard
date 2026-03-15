#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
去除all_merged文件中(src, e)重复的记录，只保留label最高的记录
"""

import json
from collections import defaultdict
import os


def remove_duplicate_src_e(input_file: str, output_file: str):
    """
    去除文件中(src, e)重复的记录，只保留label最高的记录
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
    """
    print(f"处理文件: {input_file}")
    print(f"输出文件: {output_file}")
    
    # 使用字典记录每个(src, e)组合对应的最佳记录
    src_e_to_best_record = {}
    
    total_records = 0
    duplicate_count = 0
    
    print("  正在读取文件...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                src = record.get('src', '')
                e = record.get('e', '')
                label = record.get('label', None)
                
                # 使用(src, e)作为key
                key = (src, e)
                
                total_records += 1
                
                # 如果该key不存在，或当前记录的label更高，则更新
                if key not in src_e_to_best_record:
                    src_e_to_best_record[key] = record
                else:
                    duplicate_count += 1
                    existing_label = src_e_to_best_record[key].get('label', None)
                    
                    # 比较label，保留更高的
                    if label is not None and (existing_label is None or label > existing_label):
                        src_e_to_best_record[key] = record
                    # 如果label相同，保留现有的（或可以选择保留第一条，这里保留现有的）
                    
            except json.JSONDecodeError as e:
                print(f"    警告: 第{line_num}行JSON解析失败: {e}")
                continue
            except Exception as e:
                print(f"    警告: 第{line_num}行处理失败: {e}")
                continue
    
    print(f"  读取完成: {total_records} 条记录")
    print(f"  去重后: {len(src_e_to_best_record)} 条记录")
    print(f"  去除重复记录: {duplicate_count} 条")
    print(f"  实际减少: {total_records - len(src_e_to_best_record)} 条")
    
    # 保存结果
    print(f"  正在保存到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in src_e_to_best_record.values():
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"  ✓ 保存完成")
    print()
    
    return {
        'input_file': input_file,
        'output_file': output_file,
        'total_records': total_records,
        'output_records': len(src_e_to_best_record),
        'removed_records': total_records - len(src_e_to_best_record)
    }


def main():
    """处理所有文件"""
    files_to_process = [
        ('./datasets/all_merged_20.jsonl',
         './datasets/all_merged_20_new.jsonl'),
        ('./datasets/all_merged_30.jsonl',
         './datasets/all_merged_30_new.jsonl'),
        ('./datasets/all_merged_40.jsonl',
         './datasets/all_merged_40_new.jsonl'),
    ]
    
    print("=" * 70)
    print("去除all_merged文件中(src, e)重复的记录（保留label最高的）")
    print("=" * 70)
    print()
    
    results = []
    
    for input_file, output_file in files_to_process:
        # 检查输入文件是否存在
        if not os.path.exists(input_file):
            print(f"警告: 输入文件不存在，跳过: {input_file}")
            print()
            continue
        
        # 处理文件
        result = remove_duplicate_src_e(input_file, output_file)
        results.append(result)
    
    # 汇总统计
    print("=" * 70)
    print("处理完成汇总")
    print("=" * 70)
    print()
    
    total_input = 0
    total_output = 0
    total_removed = 0
    
    for result in results:
        print(f"文件: {os.path.basename(result['input_file'])}")
        print(f"  输入记录数: {result['total_records']}")
        print(f"  输出记录数: {result['output_records']}")
        print(f"  去除记录数: {result['removed_records']}")
        print(f"  保留比例: {result['output_records']/result['total_records']*100:.2f}%")
        print()
        
        total_input += result['total_records']
        total_output += result['output_records']
        total_removed += result['removed_records']
    
    print("总计:")
    print(f"  输入记录数: {total_input}")
    print(f"  输出记录数: {total_output}")
    print(f"  去除记录数: {total_removed}")
    print(f"  保留比例: {total_output/total_input*100:.2f}%")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()

