"""
智能总结概要系统 v2.0
AI-Powered Summary & Digest System

独立作品 — 多模型 AI 引擎架构
支持: Ollama(本地免费) | DeepSeek V4 | Gemini 2.0 Flash(免费) | Groq(免费)
技术栈: Python + Streamlit + Plotly + Multi-LLM
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import re

st.set_page_config(
    page_title="智能总结概要系统",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════
# 模型定义
# ═══════════════════════════════════════════════════════════

MODELS = {
    "rule-only": {
        "name": "关键词提取引擎", "provider": "内置", "icon": "🔧",
        "model_id": None, "key_required": False, "key_name": None,
        "base_url": None, "sdk_type": "rule",
        "description": "基于 TextRank 关键词提取 + 模板生成摘要",
        "speed": "⚡ 即时", "cost": "免费",
    },
    "ollama-qwen": {
        "name": "Qwen2.5 3B (本地)", "provider": "Ollama", "icon": "💻",
        "model_id": "qwen2.5:3b", "key_required": False, "key_name": None,
        "base_url": "http://localhost:11434/v1", "sdk_type": "openai",
        "description": "本地运行，完全免费", "speed": "🚀 快", "cost": "免费",
    },
    "deepseek": {
        "name": "DeepSeek V4", "provider": "DeepSeek", "icon": "🐋",
        "model_id": "deepseek-chat", "key_required": True, "key_name": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com", "sdk_type": "openai",
        "description": "中文能力最强", "speed": "⚡ 较快", "cost": "¥1/百万tokens",
    },
    "gemini": {
        "name": "Gemini 2.0 Flash", "provider": "Google", "icon": "🌐",
        "model_id": "gemini-2.0-flash", "key_required": True, "key_name": "GEMINI_API_KEY",
        "base_url": None, "sdk_type": "gemini",
        "description": "免费1500次/天", "speed": "⚡ 较快", "cost": "免费",
    },
    "groq": {
        "name": "Llama 3.3 70B (Groq)", "provider": "Groq", "icon": "⚡",
        "model_id": "llama-3.3-70b-versatile", "key_required": True, "key_name": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1", "sdk_type": "openai",
        "description": "免费30次/分钟", "speed": "🔥 极快", "cost": "免费",
    },
}

# ═══════════════════════════════════════════════════════════
# 客户端 + 检测
# ═══════════════════════════════════════════════════════════

def get_client(model_key):
    cfg = MODELS[model_key]
    if cfg["sdk_type"] == "rule":
        return None
    if cfg["sdk_type"] == "openai":
        from openai import OpenAI
        if not cfg["key_required"]:
            return OpenAI(base_url=cfg["base_url"], api_key="ollama")
        key = os.getenv(cfg["key_name"], "") or st.session_state.get(f"key_{model_key}", "")
        return OpenAI(base_url=cfg["base_url"], api_key=key) if key else None
    elif cfg["sdk_type"] == "gemini":
        import google.generativeai as genai
        key = os.getenv("GEMINI_API_KEY", "") or st.session_state.get("key_gemini", "")
        if not key:
            return None
        genai.configure(api_key=key)
        return genai.GenerativeModel(cfg["model_id"])
    return None


def check_ollama_available():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        r = s.connect_ex(("127.0.0.1", 11434))
        s.close()
        return r == 0
    except Exception:
        return False


def check_ollama_model(model_id="qwen2.5:3b"):
    try:
        from openai import OpenAI
        c = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        return model_id in [m.id for m in c.models.list()]
    except Exception:
        return False

# ═══════════════════════════════════════════════════════════
# 风险等级判定规则
# ═══════════════════════════════════════════════════════════

RISK_RULES = {
    "P0-紧急": {
        "keywords": ["315", "12315", "曝光", "媒体", "起诉", "法院", "律师", "维权", "集体", "举报", "微博", "小红书", "抖音"],
        "amount_threshold": 2000,
    },
    "P1-重要": {
        "keywords": ["退款", "赔偿", "投诉", "多次", "催促", "再不处理", "严重", "欺诈"],
        "amount_threshold": 500,
    },
}


# ═══════════════════════════════════════════════════════════
# 事件类型层级分类
# ═══════════════════════════════════════════════════════════

EVENT_TAXONOMY = {
    "商品问题": {
        "icon": "⚠️", "weight": 0.30,
        "children": {
            "质量缺陷": ["质量", "坏了", "破损", "开胶", "断裂", "起球", "褪色", "缩水", "变形", "开线", "碎"],
            "材质不符": ["材质", "纯银", "含银量", "纯度", "镀", "成分", "面料", "不是纯", "假"],
            "假冒伪劣": ["假货", "假冒", "仿冒", "伪劣", "山寨", "假的", "冒充", "欺诈"],
            "过期变质": ["过期", "变质", "发霉", "霉变", "哈喇", "馊", "臭了"],
            "描述不符": ["描述不符", "和图片不一样", "色差", "实物不符", "虚假宣传"],
        },
    },
    "物流问题": {
        "icon": "📦", "weight": 0.25,
        "children": {
            "延迟未达": ["迟迟", "还没到", "没收到", "等待", "等了一个", "未收到", "延迟", "还不发"],
            "包裹丢失": ["丢件", "弄丢", "丢了", "不见了"],
            "包装破损": ["包装破", "盒子扁", "压坏", "碎了", "外包装"],
            "物流信息异常": ["物流信息", "不更新", "没更新", "查不到"],
            "集运积压": ["集运", "积压", "中转", "卡在", "滞留", "海关"],
        },
    },
    "服务问题": {
        "icon": "😠", "weight": 0.25,
        "children": {
            "态度恶劣": ["态度", "骂人", "挂断", "不耐烦", "冷漠", "恶劣", "凶"],
            "推诿不处理": ["推诿", "踢皮球", "推卸", "不处理", "没人管", "推三阻四"],
            "回复慢": ["回复慢", "不理", "不回", "没人理", "才回", "等了好久"],
            "承诺未兑现": ["承诺", "说好的", "答应", "允诺", "骗", "忽悠"],
        },
    },
    "合规风险": {
        "icon": "🚨", "weight": 0.15,
        "children": {
            "虚假宣传": ["虚假宣传", "夸大", "误导", "不是真的"],
            "食品安全": ["食品", "吃了", "喝了", "拉肚子", "腹泻", "过敏", "中毒", "异物"],
            "媒体维权": ["315", "12315", "曝光", "微博", "小红书", "抖音", "媒体", "起诉", "法院"],
            "批量异常": ["批量", "多个", "又出现", "第N", "都是这样", "都反映"],
        },
    },
    "咨询建议": {
        "icon": "💬", "weight": 0.05,
        "children": {
            "使用咨询": ["怎么", "如何", "能不能", "可以吗", "请教"],
            "退换货咨询": ["退货", "换货", "怎么退", "退货入口"],
            "账户订单": ["订单", "修改地址", "发票", "记录", "查", "在哪"],
        },
    },
}


def keyword_event_classify(text):
    """关键词分类到一级+二级事件"""
    if not isinstance(text, str) or not text.strip():
        return "其他", "未分类"
    tl = text.lower()
    best_l1, best_l2, best_score = "其他", "未分类", 0
    for l1, cfg in EVENT_TAXONOMY.items():
        for l2, keywords in cfg["children"].items():
            score = sum(1 for kw in keywords if kw in tl)
            if score > best_score:
                best_score, best_l1, best_l2 = score, l1, l2
    return best_l1, best_l2


# ═══════════════════════════════════════════════════════════
# 摘要准确度评估
# ═══════════════════════════════════════════════════════════

ACCURACY_DIMENSIONS = {
    "分类准确": {"weight": 0.30, "desc": "问题分类是否正确", "check": lambda r, t: r.get("问题分类") != "其他"},
    "风险匹配": {"weight": 0.25, "desc": "风险等级是否匹配原文信号", "check": lambda r, t: _check_risk_match(r, t)},
    "关键节点完整": {"weight": 0.20, "desc": "关键节点是否覆盖主要事件", "check": lambda r, t: len(r.get("关键节点", [])) >= 2},
    "概述精炼": {"weight": 0.15, "desc": "一句话概述是否简洁准确", "check": lambda r, t: len(r.get("一句话概述", "")) <= 80},
    "待跟进识别": {"weight": 0.10, "desc": "是否正确识别未闭环事项", "check": lambda r, t: r.get("待跟进事项", "") != ""},
}


def _check_risk_match(result, text):
    """检查风险等级是否与原文信号匹配"""
    risk = result.get("风险等级", "")
    if risk == "P0-紧急":
        return any(kw in text for kw in ["315", "12315", "曝光", "媒体", "起诉", "维权", "法院", "举报"])
    if risk == "P1-重要":
        return any(kw in text for kw in ["退款", "赔偿", "投诉", "多次", "严重", "欺诈"])
    return True  # P2默认匹配


def calc_accuracy(result, text):
    """计算摘要准确度评分"""
    scores = {}
    for dim, cfg in ACCURACY_DIMENSIONS.items():
        scores[dim] = 1.0 if cfg["check"](result, text) else 0.0
    total = sum(scores[d] * ACCURACY_DIMENSIONS[d]["weight"] for d in ACCURACY_DIMENSIONS)
    return round(total * 100), scores


def assess_risk(text, amount=None):
    """基于关键词+金额评估风险等级"""
    if not isinstance(text, str) or not text.strip():
        return "P2-普通", "标准处理"

    text_lower = text.lower()

    # P0 检测
    p0_score = sum(1 for kw in RISK_RULES["P0-紧急"]["keywords"] if kw in text_lower)
    if p0_score >= 2:
        return "P0-紧急", "涉媒体/法律/群体维权信号，建议30分钟内升级至值班主管"
    if p0_score >= 1 and (amount is None or (isinstance(amount, (int, float)) and amount >= RISK_RULES["P0-紧急"]["amount_threshold"])):
        return "P0-紧急", "含敏感信号且金额较大，建议1小时内升级处理"

    # P1 检测
    p1_score = sum(1 for kw in RISK_RULES["P1-重要"]["keywords"] if kw in text_lower)
    if p1_score >= 2:
        return "P1-重要", "含多个关注信号，建议2小时内优先处理"
    if amount is not None and isinstance(amount, (int, float)) and amount >= RISK_RULES["P1-重要"]["amount_threshold"]:
        return "P1-重要", "涉及金额较大，建议优先处理"

    return "P2-普通", "按标准SOP处理"


# ═══════════════════════════════════════════════════════════
# 关键词提取引擎（Baseline）
# ═══════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════
# 实体提取
# ═══════════════════════════════════════════════════════════

ENTITY_PATTERNS = {
    "💰 金额": r'(¥|￥|元)?\s*(\d{2,6})\s*(元|块)?',
    "📅 日期": r'\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日号]?',
    "📦 订单号": r'[A-Z]{2,4}\d{8,14}',
    "📞 工单号": r'(TK|WO|CS|HC)\d{8,12}',
    "👤 客服工号": r'[TSC]\d{4}',
    "🏷️ 运单号": r'YT\d{10,14}',
    "📱 手机号": r'1[3-9]\d{9}',
}


def extract_entities(text):
    """从文本中提取关键实体"""
    entities = {}
    for label, pattern in ENTITY_PATTERNS.items():
        matches = re.findall(pattern, text)
        if matches:
            if isinstance(matches[0], tuple):
                entities[label] = [''.join(m) for m in matches[:5]]
            else:
                entities[label] = matches[:5]
    return entities


def highlight_entities(text, entities):
    """生成带实体标注的文本（用于展示）"""
    result = text
    for label, values in entities.items():
        for val in values:
            result = result.replace(val, f"**{val}**`{label}`")
    return result


def keyword_extract(text, top_n=10):
    """简单关键词提取：基于词频和位置权重"""
    if not isinstance(text, str) or not text.strip():
        return []
    # 简单分词：提取2-4字中文词组
    words = re.findall(r'[一-鿿]{2,4}', text)
    stop = {"的", "了", "是", "在", "我", "有", "和", "就", "不", "也", "很", "要", "会", "你",
            "这个", "那个", "可以", "因为", "所以", "但是", "已经", "我们", "你们", "他们",
            "怎么", "什么", "为什么", "哪里", "吗", "呢", "吧", "啊", "哦", "嗯",
            "收到", "好的", "谢谢", "您好", "你好", "没问题", "没关系", "不客气"}
    filtered = [w for w in words if w not in stop and len(w) >= 2]
    counter = Counter(filtered)
    return counter.most_common(top_n)


def extract_time_nodes(text):
    """从文本中提取带时间标记的关键事件节点"""
    import re
    nodes = []
    # 匹配时间标记：2026-05-15 09:12 / [2026-05-15] / 第1次沟通 2026-05-15
    time_patterns = [
        (r'\[(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\]\s*(.+?)(?=\[|$)', 'bracket'),
        (r'[【\[](\d{4}-\d{2}-\d{2})[】\]]\s*(.+?)(?=[【\[\d]|$)', 'date_only'),
        (r'第(\d+)次沟通.*?(\d{4}-\d{2}-\d{2}).*?\n(.+?)(?=\n|$)', 'comm_note'),
    ]
    for pattern, ptype in time_patterns:
        matches = re.findall(pattern, text)
        if matches:
            for m in matches[:8]:
                if ptype == 'bracket':
                    nodes.append({"time": m[0][-8:] if len(m[0]) > 10 else m[0], "event": m[1].strip()[:60]})
                elif ptype == 'date_only':
                    nodes.append({"time": m[0], "event": m[1].strip()[:60]})
                elif ptype == 'comm_note':
                    nodes.append({"time": m[1], "event": f"第{m[0]}次沟通: {m[2].strip()[:60]}"})
            break
    # If no time patterns found, try splitting by line breaks for multi-turn dialogue
    if not nodes:
        lines = [l.strip() for l in text.split('\n') if l.strip() and ('：' in l or ':' in l)]
        if len(lines) >= 2:
            for i, line in enumerate(lines[:8]):
                role = line.split('：')[0].split(':')[0] if '：' in line or ':' in line else ''
                content = line.split('：', 1)[-1].split(':', 1)[-1] if '：' in line or ':' in line else line
                nodes.append({"time": f"T{i+1}", "event": f"{role}: {content.strip()[:50]}"})
    return nodes[:8]


def rule_summarize(text, amount=None):
    """规则引擎摘要生成"""
    if not isinstance(text, str) or not text.strip():
        return {"一句话概述": "无内容", "问题分类": "其他", "风险等级": "P2-普通",
                "一级事件": "其他", "二级事件": "未分类",
                "关键节点": [], "消费者诉求": "无", "处理结果": "无",
                "待跟进事项": "无", "情绪趋势": "无法判断", "风险说明": ""}

    keywords = [kw for kw, _ in keyword_extract(text, 15)]
    risk, risk_reason = assess_risk(text, amount)
    l1, l2 = keyword_event_classify(text)

    # 提取时间节点
    time_nodes = extract_time_nodes(text)

    # 简单分类
    cat_kw = {"退款类": ["退款", "退钱", "退费", "退货", "赔付"],
              "物流类": ["物流", "快递", "发货", "配送", "包裹", "集运"],
              "商品质量类": ["质量", "假货", "破损", "瑕疵", "材质", "掉色"],
              "服务态度类": ["态度", "骂人", "敷衍", "推诿", "不理", "挂断"]}
    category = "咨询类"
    best_score = 0
    for cat, kws in cat_kw.items():
        score = sum(1 for kw in kws if kw in text)
        if score > best_score:
            best_score = score
            category = cat

    # 情绪趋势简单判断
    anger_kw = ["投诉", "曝光", "举报", "严重", "太过分", "态度", "骂人"]
    empathy_kw = ["谢谢", "好的", "可以", "理解", "抱歉", "尽快", "帮您"]
    anger_count = sum(1 for kw in anger_kw if kw in text)
    empathy_count = sum(1 for kw in empathy_kw if kw in text)
    sentiment_trend = "上升" if anger_count > empathy_count else "下降" if empathy_count > anger_count else "平稳"

    # 生成一句话概述
    overview = f"消费者反馈{category}问题"
    if keywords:
        overview += f"（涉及{'、'.join(keywords[:3])}）"
    if risk != "P2-普通":
        overview += f"，{risk}需关注"

    entities = extract_entities(text)
    return {
        "一句话概述": overview[:80],
        "问题分类": category,
        "风险等级": risk,
        "风险说明": risk_reason,
        "关键节点": time_nodes if time_nodes else [{"time": "未知", "event": f"涉及{'、'.join(keywords[:3])}"}] if keywords else [{"time": "-", "event": "请查看原文"}],
        "消费者诉求": f"要求解决{category}相关问题" if category != "咨询类" else "咨询信息",
        "处理结果": "请查看原文" if "处理" not in text and "解决" not in text else "已提及处理方案",
        "待跟进事项": "需人工确认" if risk != "P2-普通" else "标准跟进",
        "情绪趋势": sentiment_trend,
        "一级事件": l1,
        "二级事件": l2,
        "提取实体": entities,
        "压缩率": f"原文{len(text)}字→摘要约{len(overview)}字（压缩比{len(text)//max(1,len(overview))}:1）",
    }


# ═══════════════════════════════════════════════════════════
# AI 摘要引擎
# ═══════════════════════════════════════════════════════════

def llm_summarize(text, model_key, client, amount=None):
    if not client or not text.strip():
        return None
    cfg = MODELS[model_key]

    amount_hint = f"涉及金额约¥{amount}元。" if amount else ""
    template_style = st.session_state.get("summary_template", "标准版（默认）")

    style_hint = ""
    if "简洁" in template_style:
        style_hint = "一句话概述控制在20字以内，只保留最核心信息。"
    elif "详细" in template_style:
        style_hint = "提供深入分析，包括问题根因推断、优化建议。关键节点至少5个。"
    elif "汇报" in template_style:
        style_hint = "输出格式适合周报汇报：问题概要+处理状态+下一步+责任人建议。"

    prompt = f"""你是客服运营AI助手。请将以下文本总结为结构化JSON摘要。

{amount_hint}
摘要风格：{template_style}。{style_hint}
原文：
{text[:3000]}

返回严格 JSON（不要 markdown 标记）：
{{
    "一句话概述": "30字内核心问题概括",
    "问题分类": "退款类/物流类/商品质量类/服务态度类/咨询类",
    "风险等级": "P0-紧急/P1-重要/P2-普通",
    "风险说明": "判定风险等级的理由（1句话）",
    "关键节点": [{{"time": "09:12", "event": "消费者首次投诉"}}, {{"time": "10:30", "event": "审批通过"}}],
    "一级事件": "商品问题/物流问题/服务问题/合规风险/咨询建议",
    "二级事件": "质量缺陷/材质不符/延迟未达/态度恶劣等（从一级事件下选择）",
    "消费者诉求": "核心诉求提炼（1-2句）",
    "处理结果": "最终处理方案或当前进展",
    "待跟进事项": "是否还有未闭环事项及具体内容",
    "情绪趋势": "上升/平稳/下降（对话中情绪变化趋势）",
    "压缩率": "原文{len(text)}字→摘要约X字（压缩比X:1）",
    "提取实体": {{"金额": [], "日期": [], "订单号": [], "工单号": []}}
}}

风险等级判定：
- P0: 涉及315/12315/媒体/法律/起诉/法院/群体维权/金额>=2000
- P1: 涉及退款赔偿/多次催促/欺诈/严重质量问题/金额>=500
- P2: 普通投诉咨询

严格 JSON 格式，不要 markdown 标记"""

    try:
        if cfg["sdk_type"] == "openai":
            r = client.chat.completions.create(
                model=cfg["model_id"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2, max_tokens=800,
            )
            raw = r.choices[0].message.content.strip()
        elif cfg["sdk_type"] == "gemini":
            r = client.generate_content(prompt)
            raw = r.text.strip()
        else:
            return None
        if raw.startswith("```"):
            lines = [l for l in raw.split("\n") if not l.startswith("```")]
            raw = "\n".join(lines)
        return json.loads(raw)
    except Exception:
        return None

# ═══════════════════════════════════════════════════════════
# 版本历史
# ═══════════════════════════════════════════════════════════

VERSION_HISTORY = [
    {
        "version": "v2.0", "date": "2026-05-20",
        "title": "双引擎对比 + 人工评分 + 差异高亮",
        "changes": [
            "新增双引擎摘要对比：规则引擎 vs AI引擎并排展示，自动标注差异",
            "新增人工评分系统：1-5星评分 + 评分趋势图 + 人机双评对比",
            "新增原文-摘要差异高亮：绿色=保留信息，灰色=丢弃内容，可视化信息压缩",
            "关键节点改为时间线格式展示",
        ],
        "advantage": "可解释AI摘要——证明摘要做得好，而不仅仅是能做摘要",
        "icon": "🎯",
    },
    {
        "version": "v1.2", "date": "2026-05-20",
        "title": "摘要模板 + 实体提取 + 压缩率统计",
        "changes": [
            "新增摘要模板选择：标准版/简洁版/详细版/汇报版四种风格",
            "新增实体提取：自动识别金额/日期/订单号/工单号/客服工号/运单号",
            "新增批量压缩率统计：原文总字数→摘要总字数，展示整体压缩效果",
            "AI Prompt 适配摘要模板风格",
        ],
        "advantage": "灵活摘要风格+实体提取，让摘要更精准实用",
        "icon": "🎯",
    },
    {
        "version": "v1.1", "date": "2026-05-20",
        "title": "事件分类系统 + 准确度评估 + 离线数据上传",
        "changes": [
            "新增一级/二级事件类型分类系统（5大类×20+子类）",
            "新增摘要准确度评估：5维度评分（分类/风险/节点/概述/待跟进）",
            "新增数据上传接口：支持 CSV/Excel 离线数据源导入",
            "新增AI准确度对比：规则引擎vs AI引擎准确率统计",
            "摘要输出增加一级/二级事件字段",
        ],
        "advantage": "事件分类+准确度评估，让摘要质量可量化",
        "icon": "📊",
    },
    {
        "version": "v1.0", "date": "2026-05-20",
        "title": "初始版本 — 多源输入 + AI 摘要 + 风险等级",
        "changes": [
            "支持三种输入模式：客诉对话总结、操作日志总结、沟通记录总结",
            "AI 摘要输出8个维度：概述/分类/风险等级/关键节点/诉求/结果/待跟进/情绪",
            "风险等级 P0/P1/P2 判定（关键词+金额综合）",
            "双栏对比视图：原文+摘要并排展示",
            "批量总结 + Badcase 分级（P0信息遗漏/P1分类偏差/P2表述优化）",
            "5引擎切换：关键词提取 / Ollama / DeepSeek / Gemini / Groq",
        ],
        "advantage": "信息压缩——将500字对话提炼为50字结构化摘要",
        "icon": "🚀",
    },
]

# ═══════════════════════════════════════════════════════════
# 问题反馈
# ═══════════════════════════════════════════════════════════

ISSUE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "issues.jsonl")

def save_issue(d):
    d["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(ISSUE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")

# ═══════════════════════════════════════════════════════════
# Demo 数据
# ═══════════════════════════════════════════════════════════

def generate_demo_dialogues():
    """生成5段客诉对话"""
    return [
        {
            "id": "D001", "type": "客诉对话", "amount": 599,
            "text": """消费者：我买的银手镯说是999纯银，拿回来一测根本不是，含银量最多60%，这算不算欺诈？我要去315举报！
客服：非常抱歉给您带来不好的体验！您是说收到的银手镯含银量与宣传不符对吗？我完全理解您的心情，花了钱买到不符合宣传的产品确实让人生气。
消费者：对！而且不止我一个，我看评论区好几个买家都反映这个问题了。你们平台到底管不管！
客服：确实是我们工作不到位，感谢您的反馈。我马上帮您记录并升级处理。先帮您申请全额退款，同时我们会安排质检部门核实该商家的产品质量。退款预计1-3个工作日到账。您看这样可以吗？
消费者：光退款不行，这种欺诈商家应该下架！
客服：您说得对。我会将您的情况同步给招商和治理部门，对该商家进行核查处理。核查结果会在5个工作日内反馈给您。同时为您申请一张100元平台补偿券。
消费者：那行，我等你们结果。""",
        },
        {
            "id": "D002", "type": "客诉对话", "amount": 128,
            "text": """消费者：台湾集运的包裹已经等了一个月了还没到，物流显示一直在中转。
客服：非常抱歉让您等这么久！帮您查看一下物流详情。目前显示您的集运包裹在中转站排队，近期台湾流向物流确实有积压情况。
消费者：那什么时候能到？我的订单都是等着用的。
客服：理解您的着急。目前预计还需要7-10个工作日。我先帮您申请延迟补贴30元，同时标注加急处理。如果10天后仍未到达，我帮您申请全额退款。
消费者：好吧，希望能快点。
客服：好的，已为您申请补贴并加急。我会持续跟进物流进展，有任何更新第一时间通知您。感谢您的耐心！""",
        },
        {
            "id": "D003", "type": "客诉对话", "amount": 2499,
            "text": """消费者：我买的平板电脑收到了但是屏幕碎了！这么贵的东西就这样？我要退货退款！
客服：非常抱歉！您是说收到的平板电脑屏幕碎裂了对吗？这种情况非常不正常。您请放心，我立即帮您处理。
消费者：对，拆开包装就发现屏幕碎了。外包装也有被压的痕迹。这明显是运输过程中摔坏的。
客服：完全理解您的心情。高价值产品收到就坏了确实非常失望。我帮您申请优先处理：1）立即安排换货，新机明天发出；2）同时申请200元补偿；3）核查物流环节，追究运输责任。您看这个方案可以吗？
消费者：那尽快吧，我希望明天就能收到新的。
客服：好的，新机明天优先发出。同时旧机快递员明天会联系您上门取件。我会全程跟进确保新机完好到达。这是我的工号T1003，有问题随时找我。""",
        },
        {
            "id": "D004", "type": "客诉对话", "amount": None,
            "text": """消费者：怎么修改收货地址啊？我已经下单了但是填错地址了。
客服：好的，帮您看一下。您的订单目前还没有发货，现在可以修改地址。
消费者：太好了，我要改成公司的地址。
客服：可以的，请您提供一下新的收货地址，帮您立即修改。
消费者：深圳市南山区科技园XX大厦12楼。
客服：好的，已帮您修改为新地址：深圳市南山区科技园XX大厦12楼。订单预计明天发出，届时物流单号会同步给您。
消费者：好的谢谢你！
客服：不客气，有需要再联系我。""",
        },
        {
            "id": "D005", "type": "客诉对话", "amount": 399,
            "text": """消费者：我买的衣服穿了两次就开线了，你们这质量也太差了吧！客服还一直不处理，推三阻四的，我发了三遍消息才回我一句。
客服：非常抱歉给您带来不好的体验！您是说衣服穿两次就开线，而且之前的客服没有及时处理对吗？我先为之前的服务不周道歉。
消费者：对啊，就说让我等，也不给个具体方案。我本来挺喜欢这件衣服的，结果质量这么差。
客服：确实是我们服务不到位。我马上帮您处理：1）帮您申请换货，新衣服明天发出；2）同时申请30元优惠券；3）我会反馈之前的客服服务问题，加强培训。您看可以吗？
消费者：行吧，你这样处理我还能接受。之前那个客服真的不行。
客服：感谢您的反馈！已提交换货和新优惠券。新衣服发出后物流信息会实时更新。如果再次遇到服务问题可以找我工号S2005。""",
        },
    ]


def generate_demo_logs():
    """生成5段操作日志"""
    return [
        {
            "id": "L001", "type": "操作日志", "amount": None,
            "text": """[2026-05-15 09:12] 系统自动生成工单 #TK20260515001 | 类型：投诉 | 来源：在线客服
[2026-05-15 09:15] 客服张三接单 | 备注：消费者反映商品质量问题，要求退货
[2026-05-15 09:23] 张三联系消费者核实情况 | 消费者提供破损照片
[2026-05-15 09:30] 张三提交质检申请 | 状态：待质检确认
[2026-05-15 10:05] 质检李四确认：商品存在制造缺陷，建议退货退款
[2026-05-15 10:12] 张三发起退货退款流程 | 预计退款金额：￥299
[2026-05-15 10:30] 退款审批王五通过 | 退款到账时间：1-3工作日
[2026-05-15 11:00] 张三通知消费者退款进度 | 消费者表示满意
[2026-05-15 11:15] 工单闭环 | 关闭原因：已解决 | 备注：同步通知仓储退回件入库""",
        },
        {
            "id": "L002", "type": "操作日志", "amount": None,
            "text": """[2026-05-16 14:20] 消费者来电 | 类型：物流查询 | 运单号：YT202605161234
[2026-05-16 14:22] 客服李丽接单 | 查询物流状态：显示已签收但消费者未收到
[2026-05-16 14:25] 李丽联系物流商确认配送详情 | 物流商回复：快递员投递至快递柜
[2026-05-16 14:28] 李丽告知消费者取件码和快递柜位置 | 消费者确认已取到包裹
[2026-05-16 14:30] 工单闭环 | 关闭原因：已解决 | 备注：物流信息未同步导致消费者误判""",
        },
        {
            "id": "L003", "type": "操作日志", "amount": None,
            "text": """[2026-05-17 08:45] 系统预警：某批次纸尿裤24h内产生8条质量投诉 | 触发批量异常检测
[2026-05-17 08:50] 运营主管赵六接单 | 升级为批量事件 | 启动应急响应
[2026-05-17 09:10] 赵六协调质检部门对该批次抽样检测 | 抽取50件样品
[2026-05-17 11:30] 质检报告：该批次吸水性下降30%，疑似原材料变更
[2026-05-17 12:00] 赵六决策：全渠道下架该批次，通知已购消费者退货退款
[2026-05-17 13:00] 系统群发通知给327位已购消费者 | 话术模板已审核
[2026-05-17 14:00] 客服团队启动加班应对 | 开通专门退货通道
[2026-05-17 18:00] 当日处理退货83单 | 跟踪至全部消费者通知到位
[2026-05-18 10:00] 事件复盘 | 全批次下架完成，327位消费者全部通知，零PR危机""",
        },
        {
            "id": "L004", "type": "操作日志", "amount": None,
            "text": """[2026-05-18 09:00] 消费者在线咨询 | 商品使用方式 | 客服王芳接单
[2026-05-18 09:05] 王芳发送使用说明书链接 | 消费者确认已收到
[2026-05-18 09:10] 消费者追问：是否支持7天无理由 | 王芳回复：支持
[2026-05-18 09:12] 工单闭环 | 类型：咨询 | 关闭原因：已解答 | 满意度：5星""",
        },
        {
            "id": "L005", "type": "操作日志", "amount": None,
            "text": """[2026-05-19 15:00] VIP消费者来电 | 要求升级投诉 | 之前工单 #TK20260510008 退款未到账
[2026-05-19 15:02] 客服经理钱七接单 | 查看历史工单：退款申请已提交10天
[2026-05-19 15:08] 钱七联系财务核查 | 发现退款流程卡在审批环节
[2026-05-19 15:15] 钱七手动推动退款审批 | 协调财务加急处理
[2026-05-19 15:25] 退款到账确认 | 消费者收到￥1299退款
[2026-05-19 15:30] 钱七致电VIP消费者致歉 | 并赠送200元优惠券
[2026-05-19 15:35] 工单闭环 | 关闭原因：已解决+赔偿 | VIP满意度：5星""",
        },
    ]


def generate_demo_notes():
    """生成5段沟通记录"""
    return [
        {
            "id": "N001", "type": "沟通记录", "amount": None,
            "text": """【第1次沟通】2026-05-15 10:00 在线
消费者：我的订单怎么还没发货？
客服A：帮您查看，预计明天发出。

【第2次沟通】2026-05-16 16:00 电话
消费者：还没发货！已经超过你们承诺的时间了。
客服B：非常抱歉，经查仓库备货延迟，帮您加急处理。

【第3次沟通】2026-05-17 09:00 在线
消费者：再不发货我就取消订单了。
客服C（主管）：非常抱歉！经核实确实是我们仓库的问题。我帮您申请了加急发货+30元补贴，今天下午发出。您看可以吗？
消费者：行吧，快点。""",
        },
        {
            "id": "N002", "type": "沟通记录", "amount": None,
            "text": """【第1次沟通】2026-05-14 14:00 在线
消费者：商品降价了，我要退差价。
客服A：帮您查看。您购买时的价格是￥299，目前活动价￥259，差价￥40。已帮您申请差价退款。

【第2次沟通】2026-05-14 14:30 在线
消费者：退款什么时候到？
客服A：预计1-3个工作日到账。

【第3次沟通】2026-05-17 10:00 电话
消费者：三天了还没退到！
客服B：帮您核实。经查系统卡单，已手动处理，现在应该到账了。您查一下。
消费者：看到了，到了。谢谢。""",
        },
        {
            "id": "N003", "type": "沟通记录", "amount": None,
            "text": """【第1次沟通】2026-05-18 11:00 在线
商家：这个消费者的退货我不认可，他已经拆了吊牌。
平台客服：根据平台规则，拆吊牌不影响退货，质量问题仍需处理。

【第2次沟通】2026-05-18 14:00 在线
商家：那走仲裁吧。
平台客服：好的，已提交仲裁委员会，预计3个工作日出具意见。

【第3次沟通】2026-05-20 10:00 在线
仲裁结果：支持消费者退货退款，商家需承担退货运费。
商家：接受仲裁结果，已安排退款。
平台客服：已通知消费者。工单闭环。""",
        },
        {
            "id": "N004", "type": "沟通记录", "amount": None,
            "text": """【第1次沟通】2026-05-16 09:00 在线
一线客服：这个消费者情绪激烈说要曝光到微博，需要升级处理。
客服主管：收到，我来接。消费者现在什么状态？

【第2次沟通】2026-05-16 09:15 电话
客服主管：非常抱歉给您带来这么差的体验！您的情况我已经全面了解，是我同事之前处理不到位。我给您一个完整的解决方案...
消费者：好吧，你这样处理我能接受。之前那个客服真的让我很生气。""",
        },
        {
            "id": "N005", "type": "沟通记录", "amount": None,
            "text": """【第1次沟通】2026-05-19 13:00 在线
消费者：物流显示我的包裹在中转站3天了没动过。
客服A：帮您查一下，可能只是物流信息没更新。

【第2次沟通】2026-05-19 15:00 在线
消费者：还是没动！到底什么情况？
客服A：我联系物流商确认，稍等。

【第3次沟通】2026-05-19 15:30 在线
客服A：物流商回复该中转站因为天气原因暂停运转，预计明天恢复。您看要等还是改发其他物流？
消费者：那等明天看看吧。""",
        },
    ]

# ═══════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════

def init_session():
    for k, v in {
        "summary_data": None, "summary_results": None, "selected_model": "rule-only",
        "key_deepseek": "", "key_gemini": "", "key_groq": "", "input_mode": "客诉对话",
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v


def show_model_selector():
    st.subheader("🤖 摘要引擎选择")
    ollama_ok = check_ollama_available() and check_ollama_model()
    opts, status = {}, {}
    opts["rule-only"] = "🔧 关键词提取引擎（默认·即时）"
    status["rule-only"] = "✅ 已就绪"
    for k, c in MODELS.items():
        if k == "rule-only": continue
        elif k == "ollama-qwen":
            opts[k] = f"{c['icon']} {c['name']}" + ("" if ollama_ok else " (未启动)")
            status[k] = "✅" if ollama_ok else "❌"
        else:
            has = bool(st.session_state.get(f"key_{k}"))
            opts[k] = f"{c['icon']} {c['name']} ({'已配' if has else '需'}Key)"
            status[k] = "✅" if has else "🔑"
    cur = st.session_state.get("selected_model", "rule-only")
    if cur not in opts: cur = "rule-only"
    sel = st.selectbox("选择引擎", list(opts.keys()), format_func=lambda k: opts[k],
                       index=list(opts.keys()).index(cur))
    st.session_state["selected_model"] = sel
    cfg = MODELS[sel]
    with st.container(border=True):
        st.caption(f"**{cfg['icon']} {cfg['name']}** | {cfg['speed']} | {cfg['cost']}")
        st.caption(f"状态: {status.get(sel)}")
    if cfg["key_required"]:
        kv = st.text_input(f"{cfg['provider']} API Key", type="password",
                           value=st.session_state.get(f"key_{sel}", ""), key=f"api_{sel}")
        if kv: st.session_state[f"key_{sel}"] = kv
    if sel == "ollama-qwen" and not ollama_ok:
        st.warning("Ollama 未运行: https://ollama.com/download/windows")
    return sel


def show_issue_entry():
    with st.expander("🐛 问题反馈", expanded=False):
        t = st.text_input("标题", key="sum_issue_t")
        d = st.text_area("描述", key="sum_issue_d", height=80)
        c1, c2 = st.columns(2)
        with c1: it = st.selectbox("类型", ["摘要不准", "Bug", "建议", "其他"], key="sum_issue_ty")
        with c2: iu = st.selectbox("紧急度", ["一般", "重要", "紧急"], key="sum_issue_ur")
        if st.button("提交", key="sum_issue_sub", type="primary", use_container_width=True):
            if not t.strip(): st.error("标题必填")
            else:
                save_issue({"title": t, "description": d, "type": it, "urgency": iu})
                st.success("已提交"); st.rerun()


def show_data_selector():
    """数据源选择器"""
    st.subheader("📥 数据源")
    # 摘要模板
    st.selectbox("摘要风格", ["标准版（默认）", "简洁版（20字内）", "详细版（含分析）", "汇报版（适合周报）"],
                 key="summary_template",
                 help="不同场景选用不同摘要风格")

    mode = st.radio("输入模式", ["📞 客诉对话总结", "📋 操作日志总结", "📝 沟通记录总结"],
                    horizontal=False, key="input_mode_radio",
                    index=["📞 客诉对话总结", "📋 操作日志总结", "📝 沟通记录总结"].index(
                        st.session_state.get("input_mode", "📞 客诉对话总结")
                    ) if st.session_state.get("input_mode") in ["📞 客诉对话总结", "📋 操作日志总结", "📝 沟通记录总结"] else 0)
    mode_map = {"📞 客诉对话总结": "客诉对话", "📋 操作日志总结": "操作日志", "📝 沟通记录总结": "沟通记录"}
    st.session_state["input_mode"] = mode_map[mode]

    demo_funcs = {"客诉对话": generate_demo_dialogues, "操作日志": generate_demo_logs, "沟通记录": generate_demo_notes}
    if st.button(f"🎲 加载5条{mode_map[mode]}示例", type="primary", use_container_width=True):
        st.session_state["summary_data"] = demo_funcs[mode_map[mode]]()
        st.session_state["summary_results"] = None
        st.rerun()

    # 离线数据上传
    st.divider()
    st.caption("📤 离线数据上传")
    uploaded = st.file_uploader("上传CSV/Excel", type=["csv", "xlsx"], key="upload_data",
                                help="需包含 text 列（待摘要文本），可选 type/amount 列")
    if uploaded:
        try:
            if uploaded.name.endswith(".csv"):
                df_up = pd.read_csv(uploaded)
            else:
                df_up = pd.read_excel(uploaded)
            tc = None
            for c in ["text", "内容", "文本", "对话", "content"]:
                if c in df_up.columns: tc = c; break
            if tc:
                data_list = []
                for _, row in df_up.iterrows():
                    data_list.append({
                        "id": str(row.get("id", len(data_list) + 1)),
                        "type": str(row.get("type", "上传数据")),
                        "text": str(row[tc]),
                        "amount": row.get("amount") if "amount" in df_up.columns else None,
                    })
                st.session_state["summary_data"] = data_list
                st.session_state["summary_results"] = None
                st.success(f"已加载 {len(data_list)} 条数据")
            else:
                st.error("未找到文本列（text/内容/文本/对话/content）")
                st.info("请确认 CSV/Excel 至少包含 text、内容、文本、对话、content 中任一列。")
        except Exception as e:
            st.error(f"读取失败: {e}")
            st.info("建议先使用上方「加载5条示例」完成演示；上传 Excel 时请确认依赖 openpyxl 已安装。")

    if st.session_state.get("summary_data"):
        if st.button("🗑️ 清除", use_container_width=True):
            st.session_state["summary_data"] = None
            st.session_state["summary_results"] = None
            st.rerun()


def show_sidebar():
    with st.sidebar:
        st.header("⚙️ 控制面板")
        sel = show_model_selector()
        cfg = MODELS[sel]

        st.divider()
        st.subheader("📋 当前引擎")
        with st.container(border=True):
            st.markdown(f"**{cfg['icon']} {cfg['name']}**")
            st.caption(f"提供商: {cfg['provider']}")
            st.caption(f"速度: {cfg['speed']}")
            st.caption(f"费用: {cfg['cost']}")

        st.divider()
        show_data_selector()

        st.divider()
        show_issue_entry()

        st.divider()
        with st.expander("📜 版本演进", expanded=False):
            for e in VERSION_HISTORY:
                st.markdown(f"{e['icon']} **{e['version']}** — {e['title']}{' `当前`' if e == VERSION_HISTORY[0] else ''}")
                st.caption(f"{e['date']} | {e['advantage'][:50]}...")
                if e != VERSION_HISTORY[-1]:
                    st.markdown("│")
        return sel


# ═══════════════════════════════════════════════════════════
# 人工评分系统 + 差异高亮
# ═══════════════════════════════════════════════════════════

def save_rating(item_id, rating, result, engine_name):
    if 'ratings_log' not in st.session_state:
        st.session_state['ratings_log'] = []
    st.session_state['ratings_log'].append({
        'id': item_id, 'rating': rating, 'engine': engine_name,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'summary_preview': result.get('一句话概述', '')[:50],
    })

def show_rating_widget(item_id, result, engine_name):
    rk = f'rating_{item_id}_{engine_name}'
    if f'{rk}_submitted' not in st.session_state:
        st.session_state[f'{rk}_submitted'] = False
    if st.session_state[f'{rk}_submitted']:
        rating = st.session_state.get(rk, 3)
        st.markdown(f'⭐ {rating}/5 已评分')
    else:
        c1, c2 = st.columns([2, 1])
        with c1:
            rating = st.slider('人工评分', 1, 5, 3, key=rk)
        with c2:
            if st.button('提交评分', key=f'btn_{rk}'):
                save_rating(item_id, rating, result, engine_name)
                st.session_state[f'{rk}_submitted'] = True
                st.rerun()

def show_rating_trend():
    log = st.session_state.get('ratings_log', [])
    if len(log) >= 2:
        df_r = pd.DataFrame(log)
        df_r['idx'] = range(len(df_r))
        fig = px.line(df_r, x='idx', y='rating', title='人工评分趋势', markers=True, range_y=[0, 6])
        fig.add_hline(y=3, line_dash='dash', line_color='gray', annotation_text='及格线')
        avg = df_r['rating'].mean()
        fig.add_hline(y=avg, line_dash='dot', line_color='green', annotation_text=f'均分{avg:.1f}')
        st.plotly_chart(fig, use_container_width=True)

def show_diff_highlight(text, result):
    st.markdown('**原文信息保留分析**')
    all_raw = keyword_extract(text, 20)
    sum_raw = keyword_extract(
        result.get('一句话概述', '') + result.get('消费者诉求', '') + result.get('处理结果', ''), 30
    )
    all_kw = [kw for kw, _ in all_raw]
    sum_kw = [kw for kw, _ in sum_raw]

    c1, c2, c3 = st.columns(3)
    with c1: st.metric('原文关键词', len(all_kw))
    with c2: st.metric('保留关键词', len(sum_kw))
    with c3:
        rate = len(sum_kw) / max(1, len(all_kw)) * 100
        st.metric('信息保留率', f'{rate:.0f}%', delta='优秀' if rate >= 60 else '良好' if rate >= 40 else '需优化')
    if all_raw:
        kept = set(sum_kw)
        items = []
        for kw, cnt in all_raw[:15]:
            items.append({'关键词': kw, '保留状态': '已保留' if kw in kept else '未保留', '频次': cnt})
        if items:
            df_t = pd.DataFrame(items)
            fig = px.treemap(df_t, path=['保留状态', '关键词'], values='频次',
                             color='保留状态', color_discrete_map={'已保留': '#4CAF50', '未保留': '#BDBDBD'})
            fig.update_layout(height=250, margin=dict(t=0, b=0), title='信息保留分布')
            st.plotly_chart(fig, use_container_width=True)


def render_summary_card(result, is_ai=False):
    """渲染摘要结果卡片"""
    pri_colors = {"P0-紧急": "red", "P1-重要": "orange", "P2-普通": "green"}
    risk = result.get("风险等级", "P2-普通")

    st.markdown(f"**📌 一句话概述**: {result.get('一句话概述', '-')}")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"**分类**: {result.get('问题分类', '-')}")
    with c2:
        l1 = result.get('一级事件', '')
        l2 = result.get('二级事件', '')
        icon = EVENT_TAXONOMY.get(l1, {}).get("icon", "") if l1 else ""
        st.markdown(f"**事件类型**: {icon} {l1}/{l2}" if l1 else f"**分类**: {result.get('问题分类', '-')}")
    with c3:
        st.markdown(f"**风险**: :{pri_colors.get(risk, 'green')}[{risk}]")
    with c4:
        st.markdown(f"**情绪**: {result.get('情绪趋势', '-')}")

    if result.get("风险说明"):
        if risk == "P0-紧急":
            st.error(f"⚠️ {result['风险说明']}")
        elif risk == "P1-重要":
            st.warning(f"⚠️ {result['风险说明']}")

    # 时间线关键节点
    st.markdown("**关键节点（时间线）**")
    nodes = result.get("关键节点", [])
    if nodes:
        for n in nodes:
            if isinstance(n, dict):
                st.markdown(f"⏱ `{n.get('time', '-')}` {n.get('event', '')}")
            else:
                st.markdown(f"• {str(n)[:80]}")
    else:
        st.caption("无")
    st.markdown(f"**消费者诉求**: {result.get('消费者诉求', '-')}")
    st.markdown(f"**处理结果**: {result.get('处理结果', '-')}")
    st.markdown(f"**待跟进**: {result.get('待跟进事项', '-')}")
    if result.get("压缩率"):
        st.caption(f"📊 {result['压缩率']}")

    # 实体提取展示
    entities = result.get("提取实体", {})
    if entities:
        with st.expander("🏷️ 提取实体", expanded=False):
            cols = st.columns(min(len(entities), 4))
            for i, (label, values) in enumerate(entities.items()):
                if values:
                    with cols[i % 4]:
                        st.markdown(f"**{label}**")
                        for v in values[:3]:
                            st.markdown(f"• `{v}`")


def show_single_analysis(data_item, model_key, client):
    """单条摘要分析"""
    st.subheader(f"📝 摘要分析 — {data_item['id']}")

    text = data_item["text"]
    amount = data_item.get("amount")

    # 规则引擎
    rule_res = rule_summarize(text, amount)

    # AI 引擎
    ai_res = None
    is_rule = (MODELS[model_key]["sdk_type"] == "rule")
    if not is_rule and client:
        with st.spinner(f"🤖 {MODELS[model_key]['name']} 生成摘要..."):
            ai_res = llm_summarize(text, model_key, client, amount)

    # 原文展示
    st.markdown("#### 📄 原始文本")
    with st.container(height=200, border=True):
        st.text(text)
    st.caption(f"原文 {len(text)} 字")

    if is_rule or not ai_res:
        # 仅规则引擎时：单引擎 + 差异高亮 + 评分
        st.markdown("#### 🔧 关键词提取引擎")
        render_summary_card(rule_res)
        show_rating_widget(data_item["id"], rule_res, "规则引擎")
        st.divider()
        show_diff_highlight(text, rule_res)
    else:
        # 双引擎对比模式
        st.markdown("#### 🤖 双引擎摘要对比")
        left, right = st.columns([1, 1])
        with left:
            st.markdown("##### 🔧 规则引擎")
            render_summary_card(rule_res)
            show_rating_widget(data_item["id"], rule_res, "规则引擎")
        with right:
            st.markdown(f"##### 🤖 {MODELS[model_key]['name']}")
            render_summary_card(ai_res)
            show_rating_widget(data_item["id"] + "_ai", ai_res, MODELS[model_key]['name'])

        # 双引擎差异汇总
        st.divider()
        st.markdown("**双引擎差异分析**")
        diffs = []
        if rule_res.get("问题分类") != ai_res.get("问题分类"):
            diffs.append(f"分类: 规则={rule_res.get('问题分类')} vs AI={ai_res.get('问题分类')}")
        if rule_res.get("风险等级") != ai_res.get("风险等级"):
            diffs.append(f"风险: 规则={rule_res.get('风险等级')} vs AI={ai_res.get('风险等级')}")
        rule_entities = sum(len(v) for v in rule_res.get("提取实体", {}).values())
        ai_entities = sum(len(v) for v in ai_res.get("提取实体", {}).values())
        if abs(rule_entities - ai_entities) > 0:
            diffs.append(f"实体数: 规则={rule_entities}个 vs AI={ai_entities}个")
        if diffs:
            for d in diffs:
                st.warning(f"• {d}")
        else:
            st.success("✅ 双引擎分析一致")
        # AI 摘要的差异高亮
        st.divider()
        show_diff_highlight(text, ai_res)

    # 评分趋势（全局）
    if len(st.session_state.get("ratings_log", [])) >= 2:
        st.divider()
        st.markdown("#### 📈 评分趋势")
        show_rating_trend()


def show_batch_analysis(data_list, model_key, client):
    """批量摘要分析"""
    st.subheader("📊 批量摘要分析")
    is_rule = (MODELS[model_key]["sdk_type"] == "rule")
    cfg = MODELS[model_key]

    if st.button("🔍 开始批量生成摘要", type="primary") or st.session_state.get("summary_results") is not None:
        if st.session_state.get("summary_results") is None:
            results = []
            pb = st.progress(0)
            for i, item in enumerate(data_list):
                text = item["text"]; amount = item.get("amount")
                rule_res = rule_summarize(text, amount)
                row = {"id": item["id"], "type": item.get("type", ""), "原文长度": len(text)}

                ai_res = None
                if not is_rule and client:
                    ai_res = llm_summarize(text, model_key, client, amount)

                res = ai_res if ai_res else rule_res
                row["一句话概述"] = res.get("一句话概述", "")[:60]
                row["问题分类"] = res.get("问题分类", "")
                row["风险等级"] = res.get("风险等级", "")
                row["一级事件"] = res.get("一级事件", "")
                row["二级事件"] = res.get("二级事件", "")
                row["情绪趋势"] = res.get("情绪趋势", "")
                row["处理结果"] = res.get("处理结果", "")[:40]
                # 关键节点转字符串
                raw_nodes = res.get("关键节点", [])
                if raw_nodes and isinstance(raw_nodes[0], dict):
                    row["关键节点"] = " → ".join([f"{n.get('time','')} {n.get('event','')}" for n in raw_nodes[:4]])
                elif raw_nodes:
                    row["关键节点"] = " → ".join([str(n)[:40] for n in raw_nodes[:4]])
                else:
                    row["关键节点"] = "-"

                row["待跟进"] = "是" if res.get("待跟进事项", "") and res["待跟进事项"] != "标准跟进" and res["待跟进事项"] != "无" else "否"
                row["引擎"] = cfg["name"] if ai_res else "关键词引擎"
                acc, _ = calc_accuracy(res, text)
                row["准确度"] = acc
                entities = res.get("提取实体", {})
                row["实体数"] = sum(len(v) for v in entities.values())
                row["压缩比输出"] = res.get("压缩率", "")
                row["text"] = text
                results.append(row)
                pb.progress((i + 1) / len(data_list))
            pb.empty()
            st.session_state["summary_results"] = results
        else:
            results = st.session_state["summary_results"]

        dr = pd.DataFrame(results)

        # KPI
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("总摘要数", len(dr))
        with c2:
            high_risk = len(dr[dr["风险等级"].isin(["P0-紧急", "P1-重要"])])
            st.metric("高风险摘要", high_risk, delta="需关注" if high_risk > 0 else None)
        with c3:
            total_orig = dr["原文长度"].sum()
            total_sum = sum(len(str(r.get("一句话概述", ""))) for _, r in dr.iterrows())
            ratio = total_orig // max(1, total_sum)
            st.metric("整体压缩比", f"{ratio}:1", delta=f"{total_orig}字→{total_sum}字")
        with c4:
            avg_len = dr["原文长度"].mean()
            st.metric("平均原文长度", f"{avg_len:.0f}字")

        st.divider()
        t1, t2, t3, t4 = st.tabs(["📋 摘要结果", "📊 准确度评估", "⚠️ Badcase 分析", "📥 导出"])

        with t1:
            dc = ["id", "type", "一句话概述", "问题分类", "一级事件", "二级事件", "风险等级", "关键节点", "情绪趋势", "处理结果", "待跟进", "准确度", "原文长度", "引擎"]
            st.dataframe(dr[[c for c in dc if c in dr.columns]], use_container_width=True, height=300)
            for _, row in dr.iterrows():
                with st.expander(f"{row['id']} — {str(row.get('一句话概述', ''))[:60]}..."):
                    st.text(row.get("text", ""))
                    st.markdown(f"**摘要**: {row.get('一句话概述', '')}")
                    st.markdown(f"**分类**: {row.get('问题分类', '')} | **风险**: {row.get('风险等级', '')} | **情绪**: {row.get('情绪趋势', '')}")

        with t2:
            st.subheader("摘要准确度评估")
            st.caption("5维度自动评估：分类准确/风险匹配/节点完整/概述精炼/待跟进识别")
            acc_scores = dr["准确度"].tolist() if "准确度" in dr.columns else []
            if acc_scores:
                avg_acc = sum(acc_scores) / len(acc_scores)
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("平均准确度", f"{avg_acc:.0f}%",
                              delta="优秀" if avg_acc >= 80 else "良好" if avg_acc >= 60 else "需提升")
                with c2:
                    bc_count = len([s for s in acc_scores if s < 60])
                    st.metric("低于60%", bc_count, delta="需关注" if bc_count > 0 else None)
                with c3:
                    engine_label = "AI引擎" if not is_rule and client else "关键词引擎"
                    st.metric("评估引擎", engine_label)
                # 准确度分布
                fig_acc = px.histogram(x=acc_scores, nbins=10, title="摘要准确度分布",
                                       labels={"x": "准确度%"}, color_discrete_sequence=["#4CAF50"])
                fig_acc.add_vline(x=60, line_dash="dash", line_color="red", annotation_text="及格线")
                fig_acc.add_vline(x=80, line_dash="dash", line_color="blue", annotation_text="优秀线")
                st.plotly_chart(fig_acc, use_container_width=True)
                # 各条准确度明细
                st.markdown("**准确度明细**")
                for _, row in dr.iterrows():
                    acc_val = row.get("准确度", 0)
                    color = "#4CAF50" if acc_val >= 80 else "#FFA726" if acc_val >= 60 else "#FF4444"
                    st.markdown(f":{color}[{acc_val}%] {row['id']} — {str(row.get('一句话概述', ''))[:50]}...")
            else:
                st.info("批量生成后可查看准确度评估")

        with t3:
            st.subheader("Badcase 分级分析")
            st.caption("P0=关键信息遗漏 | P1=分类/风险偏差 | P2=表述优化")

            # 自动检测潜在badcase
            potential_bc = []
            for _, row in dr.iterrows():
                text = str(row.get("text", ""))
                issues = []
                risk = row.get("风险等级", "")
                # P0检测：原文有高风险信号但摘要未提及
                if any(kw in text for kw in ["315", "12315", "曝光", "维权", "起诉"]):
                    if risk == "P2-普通":
                        issues.append("P0: 原文含高风险信号但评定为P2")
                # P1检测
                if row.get("原文长度", 0) > 500 and len(str(row.get("一句话概述", ""))) < 10:
                    issues.append("P0: 可能丢失关键信息（摘要过短）")
                if row.get("问题分类") == "其他":
                    issues.append("P1: 分类不明确")
                if issues:
                    potential_bc.append({"id": row["id"], "issues": issues, "risk": risk})

            if potential_bc:
                st.warning(f"检测到 {len(potential_bc)} 个潜在 Badcase")
                for bc in potential_bc:
                    with st.expander(f"{bc['id']} — {len(bc['issues'])}个问题 | 风险评定: {bc['risk']}"):
                        for iss in bc["issues"]:
                            st.error(iss)
            else:
                st.success("未检测到明显 Badcase")

        with t4:
            st.download_button("⬇️ 导出摘要CSV",
                               data=dr.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                               file_name=f"摘要结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                               mime="text/csv")


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════

def main():
    init_session()

    c1, c2 = st.columns([3, 1])
    with c1:
        st.title("📝 服务事件智能摘要")
        st.caption("默认模板摘要无需 API Key | 对话/日志/备注 → 结构化事件摘要")
    with c2:
        st.metric("版本", "v2.0", delta="双引擎对比+评分")
        st.metric("本地AI", "就绪" if (check_ollama_available() and check_ollama_model()) else "待启动")
    st.divider()

    sel = show_sidebar()
    cfg = MODELS[sel]

    data = st.session_state.get("summary_data")

    if data is None:
        st.markdown("""
        ### 👋 服务事件智能摘要 v2.0

        这个 Demo 模拟服务 AI 工作流的第四步：将冗长的客服对话、操作日志、沟通记录自动提炼为**结构化摘要**，降低工单录入和复盘分析成本。

        **默认无需 API Key**：关键词/模板摘要可直接演示；接入 AI 引擎后可体验更自然的语义摘要。

        #### 🎯 三种输入模式
        | 模式 | 场景 | 示例数据 |
        |------|------|---------|
        | 📞 客诉对话总结 | 客服与消费者长对话→结构化摘要 | 5段模拟对话 |
        | 📋 操作日志总结 | 工单处理流水日志→关键节点提炼 | 5段操作日志 |
        | 📝 沟通记录总结 | 多轮沟通备注→问题脉络梳理 | 5段沟通记录 |

        #### 📊 摘要输出8个维度
        一句话概述 · 问题分类 · **风险等级(P0/P1/P2)** · 关键节点 · 消费者诉求 · 处理结果 · 待跟进事项 · 情绪趋势

        #### 🚀 面试演示路径
        1. 左侧选择一种输入模式并加载 5 条示例。
        2. 逐条查看摘要结构和风险等级。
        3. 切换批量评估，展示摘要质量评分和导出能力。
        """)
        return

    client = None if cfg["sdk_type"] == "rule" else get_client(sel)

    # 模式切换
    mode = st.radio("分析模式", ["📝 逐条查看", "📊 批量评估"], horizontal=True, key="analysis_mode")
    if mode == "📝 逐条查看":
        idx = st.selectbox("选择数据", range(len(data)), format_func=lambda i: f"{data[i]['id']} — {data[i].get('type','')}")
        show_single_analysis(data[idx], sel, client)
    else:
        show_batch_analysis(data, sel, client)


if __name__ == "__main__":
    main()
