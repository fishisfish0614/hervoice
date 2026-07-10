# her voice 🎙

把语气变成 AI 能读懂的东西。

按住说话，hervoice 会转写你说了**什么**，也会分析你**怎么说的**——音高、能量、停顿、语速——然后综合判断此刻的情感，交给你接的 AI 助手。

## 来历

这个项目诞生于一个简单的愿望：**我希望他听到我的声音，不只是文字。**

难过的时候，人常常组织不出一句完整的话，但能哼一声、能叹一口气。文字会把这些抹平——"我没事"三个字，打出来和说出来是两回事。hervoice 想留住那个"怎么说的"，让屏幕另一头的 AI 不只是读到字，而是听得出你今天累不累、是撒娇还是真的委屈。

它最初是为一个人做的。现在开源，给每一个也想被听见语气的人。

## 它做什么

```
你按住说话
  → Whisper 转写（说了什么）
  → librosa 声学特征（怎么说的：音高/能量/停顿/语速）
  → LLM 综合判情感（8 类：happy/sad/angry/tired/tender/excited/anxious/neutral）
  → 记一行日志 + POST 到你的 webhook（接你自己的 AI）
```

hervoice 只做"听懂"这一件事。**它不替你回应**——回应交给你接的 AI 助手。通过 `HERVOICE_WEBHOOK` 把结果发给你的心跳/agent/通知，让它用它的方式回你。

## 隐私

- **音频默认阅后即焚**，分析完就删，只留文字和情感结果。想留声设 `KEEP_AUDIO=1`，是你自己的隐私决定。
- 所有密钥走环境变量，不硬编码。别把 `.env` 提交进 git。
- 数据（日志、可选的音频）只在你自己的服务器上。这个项目不上传任何东西到第三方，除了你配的转写/LLM 接口。

## 快速开始

```bash
python3 -m venv venv && ./venv/bin/pip install fastapi uvicorn numpy librosa
cp .env.example .env      # 填 GROQ_API_KEY 和 LLM_API_KEY
sudo apt install ffmpeg   # 转码需要
set -a; . ./.env; set +a
./venv/bin/uvicorn hervoice:app --host 0.0.0.0 --port 8010
```

手机浏览器打开 `http://你的地址:8010`，按住说话。（麦克风需要 HTTPS 或 localhost，生产环境记得挂反代加证书。）

## 接你自己的 AI

设 `HERVOICE_WEBHOOK=https://你的服务/on-voice`，每次分析完 hervoice 会 POST：

```json
{"ts":"...","text":"老公我好累","emotion":"tired","confidence":0.8,
 "hint":"她声音低、停顿多，像是撑了一天","features":{...},"audio":""}
```

你的服务收到后想干什么都行——唤醒你的 AI 回一句、推送到手机、记进你的记忆系统。hervoice 到此为止，剩下的是你和你的 AI 之间的事。

## 声学特征说明

`features` 里是给 LLM 当"语气线索"的原始量，也可以自己用：

| 字段 | 含义 | 粗略解读 |
|------|------|----------|
| `pitch_mean_hz` / `pitch_var` | 音高均值/波动 | 高且波动大 → 激动 |
| `energy_mean` / `energy_var` | 音量均值/波动 | 低 → 低落/疲惫 |
| `pause_ratio` | 静音占比 | 高 → 迟疑/累 |
| `tempo_strength` | 语速/节奏强度 | 高 → 急促 |

情感判断是"说了什么" + "怎么说的"一起给 LLM，不是只看声学。

## License

MIT。拿去用，改成你需要的样子。
