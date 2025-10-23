# Live Conversation AI Assistant

A lightweight desktop application for macOS that listens to live conversations, transcribes them in real time, and surfaces AI-powered insights using OpenAI's APIs. The interface presents two synchronized panes: a live transcript of the conversation and contextual responses from the assistant.

## Features
- 🎙️ Continuous microphone capture with adjustable sample rate and chunk duration
- ✍️ Real-time speech-to-text transcription powered by OpenAI Whisper
- 💡 Conversational responses generated with OpenAI GPT models
- 🪟 Intuitive split-pane UI built with PyQt6
- 🔐 Configurable with environment variables or a `.env` file

## Getting Started

### Prerequisites
- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/)
- macOS (the UI also runs on Linux/Windows but audio routing may differ)

### Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the example environment file and add your credentials:
```bash
cp .env.example .env
# then edit .env to add your OpenAI API key
```

### Running the App
```bash
python -m src.main
```

When you press **Start Listening**, the assistant will begin buffering audio from the default system microphone. Every few seconds a new transcript chunk will appear in the left pane, followed by the assistant's response on the right. Use **Stop Listening** to pause capture.

### Configuration
| Variable | Description | Default |
| --- | --- | --- |
| `OPENAI_API_KEY` | Your OpenAI API key | **required** |
| `OPENAI_MODEL` | Chat completion model used for responses | `gpt-4o-mini` |
| `OPENAI_TRANSCRIPTION_MODEL` | Whisper model used for transcription | `whisper-1` |
| `AUDIO_SAMPLE_RATE` | Sample rate for microphone capture | `16000` |
| `AUDIO_CHUNK_DURATION` | Length of each audio segment in seconds | `5.0` |

You can tweak chunk length or sample rate to balance latency and accuracy.

### Packaging for macOS
To build a standalone macOS app bundle, you can use [`pyinstaller`](https://pyinstaller.org/) after installing the dependencies:
```bash
pip install pyinstaller
pyinstaller --windowed --name "LiveConversationAssistant" src/main.py
```

The resulting `.app` bundle will appear in the `dist/` folder. Ensure you include the `.env` file (or set environment variables) when distributing the application.

### Creating a downloadable zip bundle
To share the assistant without building a PyInstaller bundle, generate a source distribution zip that includes the application code, requirements, and configuration template:

```bash
python scripts/create_distribution_zip.py
```

The archive is saved to `dist/live-conversation-assistant.zip`. Pass `--include-pyinstaller` if you have already built the macOS `.app` bundle and want it included alongside the source files.

### Troubleshooting
- If the microphone isn't detected, verify macOS privacy permissions for the terminal or bundled app.
- For best results, use headphones to avoid echo and adjust the chunk duration in shorter conversations.
- Network connectivity is required to reach OpenAI services; failures will display in-app error dialogs.

## License
This project is provided as-is for demonstration purposes.
