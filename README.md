# B站视频音频离线转写工具

- 输入一个 B 站视频页面 URL，自动下载音频并使用本地离线模型转写为文字。
- 无需 OpenAI 等云 API，默认使用开源 Whisper（faster-whisper 实现），可选 Vosk。

## 环境要求
- Windows / macOS / Linux
- 必须安装 `ffmpeg`（用于音频抽取与重采样）
  - Windows 可下载预编译包：https://www.gyan.dev/ffmpeg/builds/ ，解压后将 `bin` 目录加入 `PATH`。
- Python 3.9+

## 安装
```bash
pip install -r requirements.txt
```

## 使用示例
```bash
python transcribe_bilibili.py "https://www.bilibili.com/video/BVxxxxxxx" --engine whisper --model small --lang zh
```
- `--engine`：`whisper`（默认，推荐）或 `vosk`
- `--model`：`tiny|base|small|medium|large-v2|large-v3`（whisper模型体积与效果成正比）
- `--lang`：语言代码，如中文 `zh`、英文 `en`；留空自动检测

输出文件位于工作目录：
- `*.srt`：字幕
- `*.txt`：纯文本
- `*.json`：包含分段时间戳与文本

## 自动导出Cookie并一键转写（推荐）
当B站出现反爬（412）或需要登录态时：
```bash
python run_bilibili_transcribe.py "https://www.bilibili.com/video/BVxxxxxxx" --out outputs --model small --lang zh
```
- 需要先关闭 Chrome；脚本将自动导出已登录的 Chrome Cookie 到 `cookies/bilibili.txt` 并调用转写。
- 若自动导出失败（如DPAPI限制），可用浏览器扩展导出为 Netscape 格式，再执行：
```bash
python transcribe_bilibili.py "https://www.bilibili.com/video/BVxxxxxxx" --out outputs --engine whisper --model small --lang zh --cookies-file cookies/bilibili.txt
```

## 无法直接下载时的替代方案

### 方案一：系统音频录制 + 离线转写
```bash
python record_and_transcribe.py "https://www.bilibili.com/video/BVxxxxxxx" --out outputs --duration 600 --model small --lang zh
```
- 打开默认浏览器并播放视频，脚本用 `ffmpeg` 录制系统音频（WASAPI/dshow），录制完成后自动转写。
- 可选 `--audio-device` 指定设备名，如 `virtual-audio-capturer` 或声卡的 Stereo Mix。

### 方案二：手动复制 m3u8 直链并转写
```bash
python download_m3u8_and_transcribe.py --m3u8 "<在DevTools中复制的m3u8直链>" --out outputs --cookies-file cookies/bilibili.txt --model small --lang zh
```
- 在浏览器开发者工具 Network 搜索 `m3u8`或`playurl`，复制直链；脚本用 `ffmpeg` 拉流并合并为 WAV，随后转写。
- 若直链需要登录态，可配合 `--cookies-file`。
## 说明
- 质量优先建议使用 `--engine whisper --model small/medium/large-v3`。
- Vosk需要手动下载中文模型，例如：https://alphacephei.com/vosk/models ，将模型解压至 `models/vosk/cn`，运行时指定 `--engine vosk --vosk-model-path models/vosk/cn`。
- B 站部分视频为分段流，`yt-dlp` 会自动合并，合并过程依赖 `ffmpeg`。

## 常见问题
- 提示未找到 `ffmpeg`：安装后确保 `ffmpeg.exe` 所在目录在系统 `PATH` 中。
- 速度慢：Whisper `large` 模型在 CPU 上较慢，可先用 `small/medium`。
- 语言识别错误：指定 `--lang zh` 可强制中文识别。
