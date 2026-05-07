# Vani-1092: AI-Assisted Voice Helpline

Vani-1092 is a real-time voice-to-voice assistance system for the **1092 Helpline**. It ensures accurate understanding of citizen issues in **Kannada, Hindi, and English** before routing to human agents.

## Project Structure
- `agent.py`: Main AI Voice Agent (LiveKit).
- `server.py`: FastAPI Backend & Operator Dashboard.
- `vani1092/`: Core logic package (STT, TTS, Intent Analysis, Database).
- `static/`: Frontend assets (Citizen UI & Operator Console).
- `docs/`: Presentation materials and documentation.
- `test_harness.py`: Automated logic verification suite.

## Setup & Running

### 1. Requirements
- Python 3.10+
- Sarvam AI API Key
- LiveKit Cloud Credentials

### 2. Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root:
```env
SARVAM_API_KEY=your_key
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret
```

### 4. Running the App
**Terminal 1 (Server & Dashboard):**
```bash
source .venv/bin/activate
python3 server.py
```
*Accessible at: http://localhost:8001/dashboard*

**Terminal 2 (AI Voice Agent):**
```bash
source .venv/bin/activate
python3 agent.py dev
```

## Testing
Run the logic test harness to verify 5/5 test cases:
```bash
python3 test_harness.py
```

## Database
The system uses SQLite. The database (`vani1092.db`) is automatically initialized on the first run of `server.py` or `agent.py`. No manual setup is required.
