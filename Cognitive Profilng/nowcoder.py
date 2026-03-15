import requests
from bs4 import BeautifulSoup
import csv
import time
import re
import html as html_module
from numpy.core.numeric import cross
import pandas as pd


def html_to_plaintext(elem, drop_hspace=True) -> str:
    if hasattr(elem, "decode_contents"):
        html = elem.decode_contents()
    else:
        html = str(elem)
    soup = BeautifulSoup(html, "lxml")
    for bad in soup(["script", "style"]):
        bad.decompose()
    for img in soup.find_all("img"):
        alt = img.get("alt")
        img.replace_with(soup.new_string(alt))

    for br in soup.find_all("br"):
        br.replace_with(soup.new_string("\n"))

    for a in soup.find_all("a"):
        text = a.get_text(separator=" ", strip=True)
        href = a.get("href")
        replacement = f"{text} ({href})" if href else text
        a.replace_with(soup.new_string(replacement))

    block_tags = ["p", "div", "blockquote", "center",
                  "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6"]
    for name in block_tags:
        for t in soup.find_all(name):
            if t.previous_sibling is None or not (isinstance(t.previous_sibling, str) and t.previous_sibling.endswith("\n")):
                t.insert_before(soup.new_string("\n"))
            t.insert_after(soup.new_string("\n"))

    text = soup.get_text()
    text = html_module.unescape(text)
    lines = [ln.strip() for ln in text.splitlines()]
    cleaned_lines = []
    prev_empty = False
    for ln in lines:
        if ln == "":
            if not prev_empty:
                cleaned_lines.append("")
                prev_empty = True
        else:
            cleaned_lines.append(ln)
            prev_empty = False
    text = "\n".join(cleaned_lines).strip()

    if drop_hspace:
        text = re.sub(r'\\hspace\{.*?\}', '', text)

    return text

class NowcoderScraper:
    def __init__(self, cookies_str):
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36',
            'Referer': 'https://ac.nowcoder.com/',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        }
        self.session.headers.update(self.headers)

        cookies_dict = {}
        for item in cookies_str.split('; '):
            if '=' in item:
                key, value = item.split('=', 1)
                cookies_dict[key] = value
        self.session.cookies.update(cookies_dict)
        print(f"已加载 {len(cookies_dict)} 个cookie\n")

    def get_submissions(self, contest_id, page=1, page_size=50):

        api_url = 'https://ac.nowcoder.com/acm/contest/status-list'
        params = {
            'id': contest_id,
            'page': page,
            'limit': page_size,
        }

        try:
            response = self.session.get(api_url, params=params)
            data = response.json()
            if data.get('code') == 0:
                submissions = data.get('data', {}).get('data', [])
                total = data.get('data', {}).get('total', 0)
                return submissions, total
            else:
                print(f"API返回错误：{data.get('msg')}")
                return [], 0
        except Exception as e:
            print(f"获取提交记录时发生错误：{e}")
            return [], 0

    def get_submission_detail(self, submission_id):

        url = f'https://ac.nowcoder.com/acm/contest/view-submission?submissionId={submission_id}'

        try:
            response = self.session.get(url)
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.text, 'html.parser')
            # 提取题干描述
            problem_description = ''
            # 可能的题干区域选择器
            for_choose = [
                'div[class="result-question-box"]',
                'div[class="subject-question"]',
            ]
            for selector in for_choose:
                desc_elem = soup.select_one(selector)
                if desc_elem:
                    problem_description = html_to_plaintext(str(desc_elem))
                    break

            user_code = ''
            selectors = [
                'pre[class*="prettyprint"][class*="lang-cpp"]',
                'pre[class*="prettyprint"][class*="lang-cpp"]',
                'pre.prettyprint.lang-cpp',
                'pre[class~="prettyprint"][class~="lang-cpp"]',
                'pre[class*="cpp"]',
            ]

            for selector in selectors:
                try:
                    code_elem = soup.select_one(selector)
                    if code_elem:
                        user_code = html_module.unescape(code_elem.get_text(strip=False))
                        break
                except Exception as e:
                    # 如果 class 名非法（包含换行符），则进行特殊修复
                    pre_tags = soup.find_all("pre")
                    for pre in pre_tags:
                        class_attr = " ".join(re.split(r"\s+", pre.get("class", ""))) if pre.get("class") else ""
                        if "prettyprint" in class_attr and "lang-cpp" in class_attr:
                            user_code = html_module.unescape(pre.get_text(strip=False))
                            break
                    if user_code:
                        break

            return {
                'description': problem_description,
                'code': user_code
            }

        except Exception as e:
            print()
            print(f"获取提交详情 {submission_id} 时出错：{e}")
            return {
                'description': '获取失败',
                'code': '获取失败'
            }

    def scrape_wrong_submissions(self, contest_id, output_file='wrong_submissions.csv', max_pages=None):
        print(f"开始筛选并记录比赛{contest_id}的错误和正确提交数据")

        # 收集所有错误提交
        wrong_submissions = []

        # 收集正确答案,只保存第一个
        correct_submissions = []
        correct_list = []

        page = 1
        while True:
            submissions, total = self.get_submissions(contest_id, page)

            if not submissions:
                break

            print(f"第 {page} 页：找到 {len(submissions)} 条提交记录")
            pre_submmit = {
                'name' : None,
                'problemId' : None,
            }
            count = 0
            for sub in submissions:
                status = sub.get('statusMessage', 0)
                languages_type = sub.get('languageName')
                if pre_submmit['name'] == sub.get('userName') and pre_submmit['problemId'] == sub.get('problemId') :
                    if count > 5:
                        print(f"{sub.get('userName')} {sub.get('problemId')}skip")
                        continue
                    else:
                        count += 1
                else:
                    count = 0
                if sub.get("userName") == '皓哥86886':
                    continue
                if status != "答案正确" and 'C++' in languages_type:

                    wrong_submissions.append({
                            'user_name': sub.get('userName', ''),
                            'submit_id': sub.get('submissionId', ''),
                            'score': sub.get('score', 0),
                            'problem_index': sub.get('index', ''),
                            'problem_id' : sub.get('problemId', ''),
                            'submit_time': sub.get('submitTime', ''),
                    })
                elif status == "答案正确" and 'C++' in languages_type:
                    problem_id = sub.get('problemId', '')
                    problem_index = sub.get('index', '')
                    submit_id = sub.get('submissionId', '')
                    problem_dict = {
                            'problem_id': problem_id,
                            'problem_index':problem_index,
                            'submit_id': submit_id,
                            "score" : sub.get('score', 100),
                    }
                    if problem_id not in correct_list:
                        print(f"收集到问题{problem_id},题号{problem_index}的答案提交,提交代号{submit_id}")
                        correct_submissions.append(problem_dict)
                        correct_list.append(problem_id)
                pre_submmit['name'] = sub.get('userName')
                pre_submmit['problemId'] = sub.get('problemId')

            page += 1
            if max_pages and page > max_pages:
                break

            time.sleep(0.5)

        print(f"共找到 {len(wrong_submissions)} 条错误提交,和{len(correct_submissions)}条正确提交")

        #获取正确答案
        correct_answer = {

        }
        correct_score = {

        }
        for i ,sub in enumerate(correct_submissions):

            try:
                detail = self.get_submission_detail(sub['submit_id'])
                correct_answer.update({
                    sub['problem_id']: detail['code'],
                })
                correct_score.update({
                    sub['problem_id']: sub['score'],
                })
                time.sleep(0.2)

            except Exception as e:
                print(f"  错误：{e}")
                continue

        # 获取每个错误提交的详细信息
        with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['用户ID','时间戳', '题目', '题干', '错误答案', '正确答案', '分数',"满分"])

            for i, sub in enumerate(wrong_submissions, 1):
                try:
                    print(f"[{i}/{len(wrong_submissions)}] 处理运行id {sub['submit_id']} (用户: {sub['user_name']}) (问题编号:{sub['problem_index']})")
                    title = f"编号:{sub['problem_index']} id:{sub['problem_id']}"
                    # 从提交详情页获取题目信息和代码
                    detail = self.get_submission_detail(sub['submit_id'])

                    if sub['problem_id'] in correct_answer:
                        writer.writerow([
                            sub['user_name'],
                            str(sub['submit_time']),
                            title,
                            detail['description'],
                            detail['code'],
                            correct_answer[sub['problem_id']],
                            sub['score'],
                            correct_score[sub.get('problem_id')],
                        ])

                        time.sleep(0.2)  # 避免请求过快

                    else:
                        print(f"手动输入问题{sub['problem_index']}的正确例子运行id   https://ac.nowcoder.com/acm/contest/{contest_id}#submit")
                        c_id = input()
                        c_detail = self.get_submission_detail(c_id)
                        fin_ans = c_detail['code']
                        s = 100

                        writer.writerow([
                            sub['user_name'],
                            str(sub['submit_time']),
                            title,
                            detail['description'],
                            detail['code'],
                            fin_ans,
                            sub['score'],
                            s,
                        ])
                        correct_answer[sub['problem_id']] = fin_ans
                        correct_list.append(sub['problem_id'])
                        correct_score[sub['problem_id']] = s


                except Exception as e:
                    print(f"  错误：{e}")
                    continue

        print(f"完成！共保存 {len(wrong_submissions)} 条错误提交记录")
        print(f"数据已保存到: {output_file}")
        return output_file

def clean_csv(input_file, output_file):
    print(f"开始清洗文件: {input_file}")
    # 读取 CSV 文件
    df = pd.read_csv(input_file, dtype=str)

    # 检查时间戳列是否存在
    if "时间戳" in df.columns:
        df["时间戳"] = df["时间戳"].astype(str)
        df = df.sort_values(by=["用户ID", "题目", "时间戳"], ascending=True)
        df = df.groupby(["用户ID", "题目"], as_index=False).head(10)

    else:
        print("⚠️ 警告：未找到“时间戳”列，跳过时间排序和过滤。")

    # 保存清洗后的文件
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"清洗完成！已保存至 {output_file}")

def main():
    #cookies
    cookies_str = ""
    # 创建实例
    scraper = NowcoderScraper(cookies_str)

    # ID
    contest_ids = [
        '79664','97666','93268','106333',
        '27302','','11746' 
        '108305',
        '108199','117503','117504','108215','108216'
    ]

    file_list = []

    for contest_id in contest_ids:
        print(f"现在进行对比赛id :https://ac.nowcoder.com/acm/contest/{contest_id}#submit 的提取")

        output_file = scraper.scrape_wrong_submissions(
            contest_id,
            output_file=f'contest_{contest_id}_wrong.csv',
            max_pages=30
        )
        clean_csv(f'contest_{contest_id}_wrong.csv',f'{contest_id}.csv')


if __name__ == '__main__':
    main()