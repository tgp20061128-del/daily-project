# RAG 项目 1：企业级 RAG 知识库（带评测）

状态：骨架 + BM25 + Chroma 向量检索 + RRF 混合检索 + bge-reranker 重排已建成（2026-11-26）。语料 7 篇自写文档 / 76 个 chunk，评测集 100 条（5 类 × 20 条）。RAGAS 检索侧评测已接入（2026-11-27，不用 LLM）；faithfulness / answer relevancy 和 LLM 生成仍为下一步接入点，见“下一步”。

## 栈
- API：FastAPI + Uvicorn
- 切分：按标题 / 段落切分，500 字符 / 50 字符重叠（`app/chunk.py`）
- 检索：BM25（jieba 分词 + 去停用词 + 拒答门）+ Chroma 向量检索，RRF（k=60）融合；三种模式同一拒答门，见“实测结果”
- 向量模型：默认 Chroma 自带 all-MiniLM-L6-v2（ONNX，英文）；中文改用 BAAI/bge-small-zh-v1.5（sentence-transformers + CPU 版 torch，`--embedding bge-small-zh-v1.5`），见“实测结果”
- 重排：BAAI/bge-reranker-base（CrossEncoder，278M 参数，~1.1GB），混合检索先取 Top-10 再重排到 Top-3；`/ask` 用 `mode=hybrid_rerank`，评测用 `evaluate.py --rerank`
- 评测：自建评测集 + 自研命中率脚本（`app/evaluate.py`）；RAGAS 检索侧指标已接入（`app/ragas.py` + `app/ragas_eval.py`，不用 LLM），faithfulness / answer relevancy 待 LLM 生成答案后补
- 部署：`docker-compose.yml`（本机无 Docker，未验证 compose 启动；本地先用 `uvicorn` 跑）

## 目录
```
app/
  main.py        # FastAPI：/health /ask /ingest；/ask 支持 mode=bm25/vector/hybrid/hybrid_rerank（默认 bm25）
  chunk.py       # Markdown 切分
  retrieval.py   # BM25 + Chroma 向量库 + RRF 混合 + 拒答门 + bge-reranker（Reranker 类，懒加载）
  ingest.py      # 读 data/docs/*.md，建索引到 data/index.json
  evaluate.py    # 跑评测集，四组对比（--qa 选文件，--embedding 换模型，--out 选输出，--rerank 加重排组）
data/docs/       # 样例语料：7 篇自写 RAG 主题文档（01 基础、02 评测、03 混合检索、04 embedding 与重排、05 切分进阶、06 部署与成本、07 幻觉控制与拒答），共 76 个 chunk
data/chroma/     # Chroma 持久化库（自动生成，可删后重建；已在 .gitignore）
eval/
  qa_seed_20.jsonl  # 20 条种子题：5 类 × 4 条（事实 / 跨文档综合 / 多轮追问 / 拒答 / 格式与引用）
  qa_100.jsonl      # 100 条评测集：5 类 × 20 条，含原 20 条原样不动 + 新增 80 条
  results.json      # 20 条集在新语料上的默认 embedding 三组对比实测
  results_bge.json  # 20 条集在新语料上的 bge 三组对比实测
  results_100.json  # 100 条集 bge 三组对比实测（主结果）
  results_100_default.json  # 100 条集默认 embedding 三组对比实测
  results_100_rerank.json   # 100 条集 bge 四组对比实测（含混合+重排）
  results_100_rewrite.json  # 100 条集 bge 改写前后三组对比实测（--rewrite）
tests/
  test_retrieval.py # BM25 + 切分 + 拒答门单测（17 个，含新门规则单测）
  test_hybrid.py    # RRF 融合 + 向量召回 + 混合拒答单测
  test_rerank.py    # 重排排序 + 混合+重排 + 门在重排前生效（5 个，用替身模型，不加载真实模型）
  test_rewrite.py   # 上下文改写：补词、去重、代词不补、无 history 原样、改写翻转排序、/ask history 字段（9 个）
```

## 实测结果（2026-11-26，样例语料 7 篇文档 / 76 个 chunk，Top-3 命中）

三组同一规则、同一拒答门，对比的是排序本身。100 条集是主评测（20 条集是 4-chunk 时代的种子集，在新语料上只作回归对照）。

### 100 条集，中文 embedding：BAAI/bge-small-zh-v1.5（512 维）——主结果

| 模式 | 总分 | Top-1 | fact | cross_doc | multi_turn | no_answer | format |
|---|---|---|---|---|---|---|---|
| 纯 BM25 | **98/100** | 87/100 | 20/20 | 20/20 | 18/20 | 20/20 | 20/20 |
| 纯向量（bge） | 86/100 | 80/100 | 18/20 | 16/20 | 14/20 | 20/20 | 18/20 |
| 混合 RRF | 95/100 | 87/100 | 20/20 | 17/20 | 18/20 | 20/20 | 20/20 |

结论：**混合还是没有超过纯 BM25**（95 对 98），但差距从 4-chunk 时的 19/20 对 20/20 缩小到只差 3 条，且 Top-1 打平 87/100。混合错的 5 条里 3 条是 cross-01/02/03——期望文档是只有 2 个 chunk 的薄文档 02，新文档把同一概念讲得更深，02 被挤出 Top-3；1 条是 multi-02（同下）；1 条是 multi-11，融合后的 Top-1 被拒答门拦下（RRF+门交互问题）。向量单用 86/100，说明 bge 本身合格，但 Top-1 80/100 仍明显弱于 BM25 的 87/100。

### 100 条集，混合 + bge-reranker 重排（BAAI/bge-reranker-base，278M 参数，~1.1GB）

混合先取 Top-10 候选，交叉编码器逐对打分后重排到 Top-3；拒答门仍作用在融合 Top-1、重排之前（重排只能调顺序，不能复活拒答）。见 `eval/results_100_rerank.json`。

| 模式 | 总分 | Top-1 | fact | cross_doc | multi_turn | no_answer | format |
|---|---|---|---|---|---|---|---|
| 纯 BM25 | **98/100** | 87/100 | 20/20 | 20/20 | 18/20 | 20/20 | 20/20 |
| 混合 RRF | 95/100 | 87/100 | 20/20 | 17/20 | 18/20 | 20/20 | 20/20 |
| 混合 + 重排 | 94/100 | **89/100** | 20/20 | 17/20 | 17/20 | 20/20 | 20/20 |

结论：**重排也没有超过 BM25，Top-3 还比混合少 1 条（94 对 95），但 Top-1 从 87 提到 89，是四组里最高的。** 重排把 4 条的 Top-1 纠正了（fact-04、fact-07、format-04、multi-13），也把 2 条的 Top-1 带偏了（format-02、multi-16）；Top-3 上 cross-03 被重排捞回、cross-07 和 multi-16 被挤出，净 -1。拒答类没被重排带坏：no_answer 仍是 20/20。重排的价值在“第一条最准”而不是“前三条全中”——对只要首条答案的场景有用，对要求 Top-3 召全的场景没收益。成本不低：本机 CPU 上 100 题 × 10 候选逐对打分约 21 分钟，模型常驻内存约 1.3GB；生产上应只对 Top-10 重排、不要全库打分。

### 100 条集，多轮追问上下文改写（规则式，不用 LLM）

多轮追问是省略句（“那它和单纯向量检索比有什么优势？”“刚才说的重叠，具体是多少字符？”），缺了前文的主语和主题词。改写方法（`app/retrieval.py` 的 `rewrite_query`，评测用 `evaluate.py --rewrite`）：取上一轮问题（`follow_up_to` 链，18 条多轮题有上文），把其中当前问题没有的内容词（jieba 分词 + 去停用词，长度 ≥ 2，纯数字除外）去重后追加到问题末尾，原问题措辞不动；代词它/这/那是停用词，永不补入；没有上文的问题原样检索。`/ask` 加可选 `history` 字段（list[str]，最老在前），不传时行为与原来完全一致。见 `eval/results_100_rewrite.json`。

| 模式 | 改写前总分 | 改写后总分 | 改写前 multi | 改写后 multi | 改写前 Top-1 | 改写后 Top-1 |
|---|---|---|---|---|---|---|
| 纯 BM25 | 98/100 | **100/100** | 18/20 | **20/20** | 87/100 | 88/100 |
| 纯向量（bge） | 86/100 | 91/100 | 14/20 | 19/20 | 80/100 | 84/100 |
| 混合 RRF | 95/100 | 97/100 | 18/20 | **20/20** | 87/100 | **90/100** |
| 混合 + 重排 | 94/100 | 97/100 | 17/20 | **20/20** | 89/100 | **92/100** |

结论：**改写把多轮类修满了。** BM25 从 98 冲到 **100/100**——这是本项目第一次满分；multi-01（补“混合 单一”）和 multi-02（补“rag 文档 切分 一般 多大”）都被修回来。混合 Top-1 提到 90/100，超过了重排组的 89/100，而且不用 1.1GB 模型、不用 21 分钟。重排组补跑后（`eval/results_100_rewrite_rerank.json`，CPU 上约 22 分钟）：改写 + 重排 97/100，Top-1 **92/100，是全部模式里的最高**，多轮也满了 20/20——重排在改写后不再丢 Top-3（94→97 追平混合），只继续做它最擅长的事：把第一条排准（相对混合纠正 fact-04、multi-01、format-04，带偏 multi-02、format-02、fact-07，净 +2）。其他四类零回归（只有 multi_turn 一栏变了），拒答仍是 20/20——拒答题没有 `follow_up_to`，改写路径根本不会触发。向量组 multi 从 14 提到 19，只剩 1 条未修（multi-11，RRF+门交互的老问题，改写补词后仍被门拦下）。局限：规则式改写只会补词，不能做真正的指代消解（“那它呢”且上文无主题词时无能为力）；复杂省略仍需 LLM 改写。

### RAGAS 检索侧评测（不用 LLM，见 `eval/results_ragas.json`）

RAGAS 里不需要 LLM 的两个检索指标，用评测集的 ground truth 直接算，不调任何 API、不需要 key。跑法：`.venv/bin/python -m app.ragas_eval`（实现在 `app/ragas.py` + `app/ragas_eval.py`，公式单测见 `tests/test_ragas.py`）。

相关性判定：评测集每条题只标了一个期望文档（`expected_doc`），没有 chunk 级标注，所以**相关 chunk = 期望文档的全部 chunk**。这是一个诚实的近似，由此带来两个必读的说明：

- Context Recall 的分母是期望文档的总 chunk 数（2~16 个），Top-3 最多取回 3 个，所以它的上限不是 1.0——全量 80 条的上限均值只有 0.366（2-chunk 文档上限 1.0，16-chunk 文档上限 3/16 = 0.1875）。表中 CR 要对照“CR 上限”读：BM25+改写 0.240 是上限的 65.5%。
- Recall 低于 1 不代表答案没找到：答案通常只在 1 个 chunk 里，CR 回答的是“整篇文档找回来多少”，找没找到要看 Hit@3 / MRR。

指标定义（Top-3 内，n = 80 条可答题；20 条拒答题不计入分母，见下）：

- Context Precision（RAGAS 原公式）：$\text{CP} = \frac{\sum_{k=1}^{3} P@k \cdot v_k}{\sum_{k=1}^{3} v_k}$，$v_k=1$ 当且仅当第 k 个 chunk 相关，$P@k$ 是前 k 个里的相关比例。相关块排得越靠前分越高；Top-3 里一个相关块都没有时 CP = 0。
- Context Recall：Top-3 里取回的相关 chunk 数 / 期望文档总 chunk 数。
- MRR：第一个相关 chunk 的排名倒数，Top-3 里没有则为 0。
- Hit@1 / Hit@3：Top-1 / Top-3 里出现期望文档的题占比。

| 模式 | Context Precision | Context Recall（上限 0.366） | MRR | Hit@1 | Hit@3 | 拒答 |
|---|---|---|---|---|---|---|
| 纯 BM25 | 0.879 | 0.223 | 0.898 | 0.838 | 0.975 | 20/20 |
| BM25 + 改写 | 0.900 | 0.240 | 0.915 | 0.850 | **1.000** | 20/20 |
| 纯向量（bge） | 0.770 | 0.184 | 0.781 | 0.750 | 0.825 | 20/20 |
| 向量 + 改写 | 0.824 | 0.200 | 0.835 | 0.800 | 0.888 | 20/20 |
| 混合 RRF | 0.876 | 0.217 | 0.881 | 0.838 | 0.938 | 20/20 |
| 混合 + 改写 | 0.907 | 0.230 | 0.913 | 0.875 | 0.963 | 20/20 |
| 混合 + 重排 | 0.879 | 0.236 | 0.892 | 0.863 | 0.925 | 20/20 |
| 混合 + 重排 + 改写 | **0.917** | **0.250** | **0.929** | **0.900** | 0.963 | 20/20 |

结论：**综合检索质量最好的是混合 + 重排 + 改写**（CP 0.917、CR 0.250、MRR 0.929、Hit@1 0.900 四项全场最高；CR 是上限 0.366 的 68.2%）。重排的收益和之前一致：只管把第一条排准（Hit@1 0.875→0.900），不管找全——不改写时重排甚至把 Hit@3 从 0.938 拉低到 0.925，改写后才追回 0.963。**“找全”最稳的仍是 BM25 + 改写**（Hit@3 1.000，全场唯一满分；CR 0.240 第二）。改写对所有模式都有提升，CP +0.02~0.05、Hit@3 +0.025~0.063，向量受益最大（CP 0.770→0.824，Hit@3 0.825→0.888）。分类上看：重排 + 改写的短板仍是 cross_doc（CP 0.767，Hit@3 0.85——期望文档 02 只有 2 个 chunk，常被讲得更深的新文档挤出），其余三类 CP 都是 0.967、Hit@1 0.95；BM25 + 改写四类很均匀，Hit@3 全是 1.000。拒答 20 条在所有模式下都是 20/20——它是门控行为不是检索行为，所以不进 precision/recall/MRR 的分母，单独列在这里。重排成本：两组（改写前/后）CPU 上共约 42 分钟。

还没做完的 RAGAS 指标：faithfulness（答案是否忠于上下文）和 answer relevancy（答案是否切题）都需要 LLM：前者要 LLM 逐条判定答案里的每个断言有没有上下文支撑，后者要 LLM 按答案反推问题再算相似度。而当前 `/ask` 只返回引用片段、`answer` 为 null，根本没有答案可打分。所以这两项暂缓，等接了 LLM 生成答案后再补——不是漏了，是前置条件没到。

### 100 条集，默认 embedding：all-MiniLM-L6-v2（Chroma 自带，ONNX，384 维）

| 模式 | 总分 | Top-1 | fact | cross_doc | multi_turn | no_answer | format |
|---|---|---|---|---|---|---|---|
| 纯 BM25 | **98/100** | 87/100 | 20/20 | 20/20 | 18/20 | 20/20 | 20/20 |
| 纯向量 | 31/100 | 31/100 | 5/20 | 4/20 | 0/20 | 20/20 | 2/20 |
| 混合 RRF | 81/100 | 73/100 | 15/20 | 16/20 | 14/20 | 20/20 | 16/20 |

结论：英文 MiniLM 在中文语料上彻底拉胯（多轮 0/20），还把混合从 98 拖到 81。中文库不要用默认 embedding；拒答门倒是立了功——向量排错时大量返回空而不是硬答，所以 no_answer 在向量模式下也是 20/20。

### 20 条集在新语料上的回归（对照用，不是主结果）

| embedding | 纯 BM25 | 纯向量 | 混合 RRF |
|---|---|---|---|
| 默认 | 18/20（multi 2/4） | 4/20 | 10/20 |
| bge | 18/20（multi 2/4） | 13/20 | 16/20 |

BM25 从 20/20 掉到 18/20，不是门坏了——拒答仍是 4/4，掉的是 multi-01、multi-02 两条最含糊的多轮追问（“那它和单纯向量检索比有什么优势？”“刚才说的重叠，具体是多少字符？”），期望文档是只有 2 个 chunk 的 01，新文档 03/05 把同一话题讲得更深，01 被挤出 Top-3。旧门在新语料上是 16/20（另错 refuse-01、refuse-04），新门把它修回 18/20。20 条集是 4-chunk 时代的种子集，100 条集才是现在的基准。

### 拒答门修好了（2026-11-26 晚）

旧门只有一条规则：Top-1 片段须与问题共享至少 2 个内容词。新语料下 refuse-01（“2026 年最新发布的大模型”）靠“发布+模型”、refuse-04（“Chroma 默认端口”）靠“Chroma+向量”各凑够 2 个词蒙混过关；bge 混合下的 refuse-03（“如何用 LoRA 微调”）靠“模型+一个”过关。

新门（`app/retrieval.py` 的 `gate_refuses`，三模式共用）：
1. 共享词数规则不变（≥2 个，或问题本身内容词不足 2 个时以全部为准）。
2. 新增第二规则：如果问题里 ≥2 个内容词在整个语料词表里根本不存在，且 Top-1 片段已经覆盖了问题里所有词表内的内容词，说明问题的核心在库外——拒答。
3. 停用词补了两类词：通用量词（一个、一次、这个、那些等）和格式指令词（注明、出处、分点、每点、列出），它们既不能算匹配，也不能算库外证据；纯数字（200、0.7）不算内容词，数字缺失不触发第二规则。

验证：20 条集门决策 20/20 全对，100 条集门决策 100/100 全对（先用探测脚本逐条核对，再跑完整评测）；22 个单测全过；FastAPI 冒烟：事实题在 hybrid_rerank 下返回正确引用，天气 / LoRA 微调 2 条拒答题正确拒答，bm25 默认行为未动。门仍是词法启发式，不是语义判断；重排已接入但只用于调顺序，尚未用作拒答的二次判定——那是下一步的方向之一。

### 安装踩坑（实测）

- Chroma 默认模型下载时 httpx 会被本机的 `NO_PROXY`（含裸 IPv6 地址）搞崩，报 `InvalidURL`。代码里 `app/retrieval.py` 的 `_sanitize_no_proxy_for_httpx()` 会自动清洗，模型已缓存后不再下载。
- `pip install sentence-transformers` 默认会拉 CUDA 版 torch（数 GB），本机 `/tmp` 只有 512MB 直接爆掉。正确做法：先 `TMPDIR=~/.cache/pip-tmp pip install torch --index-url https://download.pytorch.org/whl/cpu`（CPU 版约 190MB，下载源偶发超时，重试并加 `--timeout 120`），再装 sentence-transformers。`.venv` 现约 1.8GB，大头是 torch。
- 内存：本机可用内存不足 1GB 时，torch 加载仍能跑通（bge 评测一次约 27 秒），但不要在评测同时跑别的重活。重排模型 1.1GB 常驻内存，100 条集加重排的评测在 CPU 上约 21 分钟（100 题 × 10 候选逐对打分），评测时别并行跑别的。

## 本地跑通（已验证，见 eval/results_100.json）
```bash
.venv/bin/python -m app.ingest          # 建索引
.venv/bin/python -m app.evaluate --qa eval/qa_100.jsonl --embedding bge-small-zh-v1.5 --out eval/results_100.json   # 100 条集，bge，三组对比
.venv/bin/python -m app.evaluate --qa eval/qa_100.jsonl --embedding bge-small-zh-v1.5 --out eval/results_100_rerank.json --rerank   # 100 条集，bge，四组对比（含重排，CPU 上约 21 分钟）
.venv/bin/python -m app.evaluate --qa eval/qa_100.jsonl --embedding bge-small-zh-v1.5 --out eval/results_100_rewrite.json --rewrite   # 100 条集，bge，改写前后三组对比（18 条多轮题带上文）
.venv/bin/python -m app.evaluate --qa eval/qa_100.jsonl --embedding bge-small-zh-v1.5 --out eval/results_100_rewrite_rerank.json --rewrite --rerank   # 100 条集，bge，改写 + 重排四组对比（CPU 上约 22 分钟）
.venv/bin/python -m app.ragas_eval --out eval/results_ragas.json   # RAGAS 检索侧指标（CP/CR/MRR/Hit@k），不用 LLM；加 --rerank 补重排组（CPU 上约 40 分钟）
.venv/bin/python -m app.evaluate --qa eval/qa_seed_20.jsonl --out eval/results.json   # 20 条集回归
.venv/bin/python -m unittest discover -s tests -v   # 45 个单测
# 起服务：
.venv/bin/uvicorn app.main:app --reload
curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "什么是 RAG？", "mode": "hybrid"}'
```

## 评测集说明
- 100 条评测集 `eval/qa_100.jsonl`：5 类 × 20 条（事实 / 跨文档综合 / 多轮追问 / 拒答 / 格式与引用），每条含 `id / category / question / expected_doc / expected_answer / follow_up_to`。前 20 条与 `eval/qa_seed_20.jsonl` 原样一致，新增 80 条全部按新文档内容出题，可答题的期望答案都能在期望文档里找到依据。
- 拒答类（`no_answer`）的期望答案是明确拒答，`expected_doc` 为空；命中任何文档都算错。其中 4 条是近失拒答（LoRA 微调、Chroma 默认端口、Milvus 默认端口、OpenAI 价格等库外问题），其余是远主题问题（天气、菜谱、股票等）。
- 多轮类（`multi_turn`）的 `follow_up_to` 指向它追问的上文题号。
- 指标留空 = 未测，不编造。`eval/results_100.json` 里的数字是本次在样例语料上实测的。

## 下一步（按 84 天计划，第 9–12 周 RAG 阶段）
1. 接入真实语料：把自己的 PDF / 网页放进 `data/docs/` 替换自写样例，评测题按真实文档重写期望答案。样例语料扩到 7 篇 / 76 chunk、评测集扩到 100 条已完成，但文档仍是自写的，数字是样例值。
2. ~~接 bge-reranker，对比 Top-5 重排前后的命中率~~——已完成（见“实测结果”）：Top-1 87→89，Top-3 95→94，拒答 20/20 未变。下一步可试：把重排分也用作拒答的二次判定，或只对 Top-5 重排降成本。
3. ~~接 RAGAS，测 faithfulness / answer_relevancy / context_recall，写进 README~~——检索侧已完成（见“RAGAS 检索侧评测”）：Context Precision / Context Recall / MRR / Hit@k 全是 ground truth 直算，不用 LLM。faithfulness / answer relevancy 仍待 LLM：要等 `/ask` 生成答案后才能打分。
4. 接 LLM 生成答案（`.env` 配 `LLM_API_KEY`，见 `.env.example`），当前 `/ask` 只返回检索到的引用片段，不生成答案。
5. ~~多轮追问的上下文改写：multi-01、multi-02 这类省略式追问在语料放大后会漂到别的文档，下一步把对话历史拼进检索查询再测~~——已完成（见“实测结果”）：规则式补词，BM25 98→100/100，多轮 18→20/20，混合 Top-1 87→90/100，其他类零回归。

## 注意事项
- 样例语料是我写的通用 RAG 基础知识，不是权威论文；正式评测集要用你自己的文档重写期望答案。
- 不要把真实 API Key 写进代码或聊天，只放 `.env`（已在 `.gitignore` 建议中）。
