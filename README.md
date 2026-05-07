# Vani-1092: AI-Assisted Voice Helpline

Vani-1092 is a real-time voice-to-voice assistance system for the **1092 Helpline**. It ensures accurate understanding of citizen issues in **Kannada, Hindi, and English** before routing to human agents.

## Key Features
- **Voice-to-Voice Interaction**: Natural spoken dialogue using Sarvam AI and LiveKit.
- **Verification Loop**: Explicitly confirms understanding with the citizen before taking action.
- **Urgency & Emotion Detection**: Detects distress and critical situations to bypass verification when every second counts.
- **Operator Dashboard**: Provides a structured summary and emotional context to agents.
- **Feedback Loop**: Captures operator corrections for future model improvement.

## Setup & Running

### 1. Requirements
- Python 3.10+
- Sarvam AI API Key
- LiveKit Cloud (or self-hosted) Credentials

### 2. Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory:
```env
SARVAM_API_KEY=your_key
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret
DATABASE_URL=sqlite:///./vani1092.db
```

### 4. Running the App
**Terminal 1 (Server & Dashboard):**
```bash
source .venv/bin/activate
python server.py
```
*Accessible at: http://localhost:8001/dashboard*

**Terminal 2 (AI Voice Agent):**
```bash
source .venv/bin/activate
python agent.py dev
```

## Testing
Run the logic test harness to verify intent and urgency detection:
```bash
python test_harness.py
```
