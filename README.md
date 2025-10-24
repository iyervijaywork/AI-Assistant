# Live Conversation AI Assistant

A lightweight desktop application for macOS that listens to live conversations, transcribes them in real time, and surfaces AI-powered insights using OpenAI's APIs. The interface presents a multi-chat workspace with a live transcript pane, a scrollable assistant response pane, and controls for curating a custom knowledge base to ground the model in your own interview prep material.

## Features
- 🎙️ Continuous microphone capture with adjustable sample rate and chunk duration
- ✍️ Real-time speech-to-text transcription powered by OpenAI Whisper
- 💡 Conversational responses generated with OpenAI GPT models
- 🗂️ Manage multiple concurrent chats, each with its own transcript and assistant history
- 🧠 Upload custom reference documents so responses stay anchored to your material
- 🔁 Sync existing ChatGPT conversations so both panes mirror your account history
- 🪟 Intuitive split-pane UI built with PyQt6 and responsive scrolling panes
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

When you press **Start Listening**, the assistant begins buffering audio from the default system microphone. Short transcript snippets (about every 1–2 seconds) stream into the left pane, and once a complete thought is detected the assistant responds on the right. Use **Stop Listening** to pause capture, create additional chats for new scenarios, and reload custom reference material at any time.

### Connecting your ChatGPT account

To mirror the conversations that already live in your ChatGPT account:

1. Sign in to [chat.openai.com](https://chat.openai.com/) in your browser and open the developer tools application tab.
2. Locate the `__Secure-next-auth.session-token` cookie, copy its value, and add it to your `.env` file as `CHATGPT_ACCESS_TOKEN`. (This value must stay fresh—re-copy it whenever the web session expires.)
3. Restart the desktop app and click **Sync ChatGPT** in the sidebar. The most recent conversations appear in the chat list, and selecting one loads the transcript and responses exactly as they exist online.

If you see an "Enable JavaScript and cookies to continue" error during sync, the session cookie is no longer valid. Repeat the steps above to grab a current `__Secure-next-auth.session-token` from an active browser session and restart the app.

Live transcriptions and answers now incorporate that imported context so replies stay grounded in the specific work you have already done inside ChatGPT.

### Configuration
| Variable | Description | Default |
| --- | --- | --- |
| `OPENAI_API_KEY` | Your OpenAI API key | **required** |
| `OPENAI_MODEL` | Chat completion model used for responses | `gpt-4o-mini` |
| `OPENAI_TRANSCRIPTION_MODEL` | Whisper model used for transcription | `whisper-1` |
| `OPENAI_EMBEDDING_MODEL` | Embedding model for the knowledge base | `text-embedding-3-small` |
| `AUDIO_SAMPLE_RATE` | Sample rate for microphone capture | `16000` |
| `AUDIO_CHUNK_DURATION` | Length of each audio segment in seconds | `1.5` |
| `CHATGPT_ACCESS_TOKEN` | ChatGPT web access token used to import conversations | *(optional)* |
| `CHATGPT_BASE_URL` | Base URL for the ChatGPT web API | `https://chat.openai.com/backend-api` |
| `CHATGPT_SYNC_LIMIT` | Number of recent ChatGPT conversations to mirror | `12` |

You can tweak chunk length or sample rate to balance latency and accuracy.

### Providing your own reference material

1. Click **Load reference files…** in the sidebar and select any combination of `.txt`, `.md`, or `.pdf` documents that contain your interview notes, resumes, or study guides.
2. The assistant indexes each document into reusable snippets using OpenAI embeddings. The knowledge base status indicator shows how many sources are active.
3. During a conversation, the assistant automatically retrieves the most relevant snippets and cites that context in its answers, helping you stay grounded in your own material rather than generic responses.

The uploaded knowledge base lives in memory for the current app session. Load different sets of files to tailor the assistant for each mock interview.

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
