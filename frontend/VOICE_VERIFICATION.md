# Frontend Voice UX Verification & Technical Architecture (QA Corrective Pass)

**Date**: 2026-09-10  
**Scope**: `/home/daino/progetti/alveare/frontend` ONLY  
**Test Suite**: 15 test files passed (116 tests passed, 0 failed)  
**Code Quality**: `npm run lint` (`oxlint`) passes with 0 errors; `git diff --check .` passes with 0 whitespace errors  
**Production Build**: `dist/` built cleanly via Vite (`vite build`) in ~740ms  
**Live Endpoint**: `https://127.0.0.1:8443` (serving latest `dist/` assets)  

---

## 1. Executive Summary & Defect Resolutions

This QA corrective pass addressed blocking defects identified in the voice orchestrator interaction, audio playback pipeline, and mobile usability:

1. **Elimination of WebSocket Interrupt Feedback Loop**:
   - *Problem*: Previously, receiving `interrupted` or `vad_speech_start` from the backend WebSocket triggered `handleInterrupt()`, which emitted `{ type: 'interrupt' }` back to the server. The server acknowledged with `interrupted`, creating an infinite message ping-pong and breaking conversation state.
   - *Fix*: Split interruption into `handleLocalInterrupt()` (stops local playback queue, cancels SpeechSynthesis, clears leftover audio tokens without touching the socket) and `handleUserInterrupt()` (stops local playback AND dispatches `{ type: 'interrupt' }` to backend). Receiving server `interrupted` or `vad_speech_start` events now strictly invokes `handleLocalInterrupt()`, preventing redundant sends. Regression tested with fake WebSocket mock calls.

2. **Strict Handshake & Dynamic Configuration Protocol**:
   - *Problem*: WebSocket `onopen` previously assumed immediate connection before the backend `live_connected` handshake was received, ignoring handshake metadata and snapshotting stale React configuration state.
   - *Fix*: `ws.onopen` transitions state to `'connecting'`. Only upon receiving `{ event: 'live_connected' }` does the frontend set `'connected'` and transmit the authoritative `{ type: 'config', settings: { ... } }` payload derived from `liveSettingsRef.current`. Stale socket message drops are guarded with `wsRef.current !== ws`.

3. **Backend Wire Field Alignment for Tool Cards**:
   - *Problem*: Backend `live_orchestrator.py` dispatches `tool_call_start` (`tool`, `tool_name`, `tool_args`, `turn_id`) and `tool_call_done` (`tool`, `tool_name`, `status`, `summary`, `result_summary`), never sending a `result` field.
   - *Fix*: `LiveStudio.jsx` correlates incoming tool events using both `tool_name` and `tool` as fallback, and extracts the payload using `data.result_summary || data.summary || data.result`. Tool execution cards correctly transition from `running` to `done` with input arguments and output summary displayed in the conversation transcript.

4. **Multi-Turn Assistant Flow & Physical Audio Queue Decoupling**:
   - *Problem*: Rapid turns or multiple sentences could cause race conditions in `AudioPlaybackQueue` if `decodeAudioData` completed after a new turn began, or if native client TTS signaled turn completion prematurely.
   - *Fix*: Added generation-owned `sessionToken` tracking in `audioQueue.js`. In-flight decode jobs for superseded turns are discarded without setting `isPlaying = false`. Added sentence buffer chunking on punctuation for client TTS, only signaling turn end when all queued utterances finish and `turn_complete` has been received.

5. **Mobile 360x800 Viewport Usability & Ergonomics**:
   - *Problem*: On 360x800 mobile viewports, hardware routing cards (GPU, NPU, CPU), the Ultra-Fast banner, and wakeword notice pushed the central voice action button off-screen (`y > 800px`), requiring users to scroll down just to talk.
   - *Fix*: Reorganized layout: Central Voice Console (Mic action button, status label, canvas waveform, and telemetry HUD) is placed at the top immediately below the mode switcher. Hardware cards and wakeword notices are tucked into a collapsible section toggled by a `"Dettagli Hardware & Routing"` button. Headless Chrome DOM measurements verify that on a 360x800 screen, the primary mic button sits at `top: 629.9px`, `bottom: 713.9px`, 100% in viewport without any scrolling.

6. **Responsive Chat History Drawer on Mobile**:
   - *Problem*: On viewports `< 768px`, the history sidebar consumed horizontal screen real estate or clipped the chat interface.
   - *Fix*: Added responsive drawer mode in `SidebarHistory.jsx`. When collapsed on mobile, the sidebar renders `null` (zero width consumed). When opened, it renders as a sliding overlay drawer (`.mobile-drawer-overlay`) with a backdrop, auto-closing on conversation selection.

7. **ESLint & Whitespace Hygiene**:
   - *Problem*: `DocumentConfigModal.jsx` had 6 conditional React hook ESLint errors (`rules-of-hooks`), and trailing blank lines were present at EOF.
   - *Fix*: Refactored `DocumentConfigModal.jsx` into an outer guard component and inner `DocumentConfigModalContent` component with `key={attachment.id}`. `npm run lint` now passes with 0 errors. Fixed EOF blank lines; `git diff --check .` passes with 0 whitespace errors.

---

## 2. Headless Chrome DOM Verification & Measurements

Headless Chrome (`Google Chrome 151.0.7922.173`) was launched against `https://127.0.0.1:8443` via Chrome DevTools Protocol (CDP) to measure exact DOM bounding rects and capture screenshots.

### Mobile Viewport (360 x 800 px) - Voice Studio (`/#live`)
```json
{
  "mic": {
    "tagName": "BUTTON",
    "ariaLabel": "Tocca per parlare",
    "x": 138,
    "y": 629.9,
    "width": 84,
    "height": 84,
    "top": 629.9,
    "bottom": 713.9,
    "left": 138,
    "right": 222,
    "inViewport": true
  },
  "toggle": {
    "top": 535.5,
    "bottom": 573.5
  },
  "window": {
    "innerWidth": 360,
    "innerHeight": 800,
    "scrollY": 0
  }
}
```
- **Mic Button Visibility**: `top = 629.9px`, `bottom = 713.9px` $\implies$ fully visible on the 800px viewport with a ~30px safety margin above the 60px fixed bottom navigation bar.
- **Scroll Offset**: `scrollY = 0` $\implies$ zero scrolling needed to initiate voice interaction.

### Mobile Viewport (360 x 800 px) - Chat View (`/`)
```json
{
  "sidebarPresent": false,
  "drawerOverlayPresent": false,
  "drawerBtnPresent": false,
  "windowWidth": 360
}
```
- **Sidebar Footprint**: Zero horizontal width consumed; full 360px allocated to conversation messages and text input.

### Screenshot Artifacts
- `screenshots/voice_mobile_360x800.png`: Mobile portrait layout showing header, mode tabs, privacy badge, collapsible hardware toggle, and central glowing mic button.
- `screenshots/voice_desktop_1440x900.png`: Desktop layout showing full tri-hardware telemetry, audio visualizer canvas, and conversation transcript.
- `screenshots/chat_mobile_360x800.png`: Mobile chat layout with full-width conversation view.
- `screenshots/chat_desktop_1440x900.png`: Desktop chat layout with history sidebar and document tools.

---

## 3. Automated Test Suite Matrix

```
 RUN  v4.1.10 /home/daino/progetti/alveare/frontend

 Test Files  15 passed (15)
      Tests  116 passed (116)
   Start at  16:37:52
   Duration  7.21s
```

### Key Test Coverage Highlights:
1. `src/utils/audioQueue.test.js` (9 tests):
   - Keeps turn active when `turn_complete` arrives before audio playback finishes.
   - Discards stale in-flight decode when interrupted.
   - Bounds the audio queue to `maxQueueSize` and drops oldest on overflow with warning callback.
   - Does not corrupt `isPlaying` when an old decode resolves while a new playback session is active.
   - Handles back-to-back turns with exactly one `onTurnEnd` per turn.
   - Multi-sentence native TTS does not signal turn end before all sentences finish.
   - Does not stall the queue when `decodeAudioData` fails.
   - Reports audible latency separately from server TTFA.
   - Supports client TTS mode when configured and available.

2. `src/components/LiveStudio.test.jsx` (24 tests):
   - Renders correctly with slots and status indicators.
   - Renders Live badge and toggle.
   - Handles microphone permission error gracefully with visible alert.
   - Visibly separates Assistente tap-to-talk and Live Sperimentale modes.
   - Never requests microphone on initial mount and shows privacy status.
   - Supports explicit mute control that disables capture.
   - Renders tool execution cards when `tool_call_start` and `tool_call_done` arrive with `result_summary`.
   - Avoids text duplicated between `llm_chunk` and `turn_complete.full_text`.
   - Acknowledges memory clearing and resets context.
   - Displays distinct labels for Server TTFA and Audible Latency.
   - Cleans up all media tracks and audio contexts on unmount.
   - **Regression**: Does not enter interrupt feedback loop on server `interrupted` or `vad_speech_start`.
   - **Regression**: Sends initial config settings strictly upon `live_connected` handshake.
   - **Regression**: Handles two consecutive assistant turns cleanly in transcript.
   - **Regression**: Toggles collapsible hardware and routing section.
   - Audio resampling & downsampling ($48\text{kHz} \to 16\text{kHz}$, PCM16 clamping, Base64 decoders).

3. `src/components/ServerControl.test.jsx` (7 tests):
   - Master controller and 3 slot cards (LLM, STT, TTS).
   - Slot start actions and master live start triggers.
   - Model option deduplication (Kokoro 82M, Gemma 4, Whisper Base).
   - **Regression**: Opens and dismisses `buildTargetModal` using the `X` button without runtime error.

4. `src/components/LogsViewer.test.jsx` (4 tests):
   - NPU health diagnostics and initial logs rendering.
   - Suppresses background log polling when `isActive === false`.

5. `src/App.test.jsx` (2 tests):
   - Tab switching, hash navigation (`#live`), and background polling pause on `document.visibilityState === 'hidden'`.

---

## 4. Honest Scope & Physical Limitations

To adhere to strict engineering integrity and avoid false deployment or hardware claims:

- **UI & Protocol Verification vs Physical Audio**:
  - The screenshots, DOM tests, and unit tests verify the visual presentation, responsive layout, Web Audio queue behavior, and WebSocket protocol messaging.
  - They do **NOT** test physical acoustic speaker pressure or microphone diaphragm vibrations on actual client hardware. Physical acoustic verification requires running the browser on physical target devices with speakers and microphones.
- **Client TTS vs Server Kokoro/Audio8**:
  - Browser Client TTS uses the local operating system's `window.speechSynthesis` (Web Speech API). Quality and availability depend entirely on the host browser and OS voices.
  - Server TTS streams uncompressed $24\text{kHz}$ PCM WAV chunks generated by Kokoro-82M or Audio8 on the backend.
  - In `LiveStudio.jsx`, the user can explicitly toggle between Server and Client TTS in the Live Settings drawer.
- **Zero Modifications Outside Frontend**:
  - All modifications are strictly contained within `/home/daino/progetti/alveare/frontend`.
  - Backend files (`runtime/py/control_server.py`, `runtime/py/live_orchestrator.py`, kernels, services) were **NOT modified or restarted**.
  - The running Uvicorn control server process on port 8443 continues serving the latest rebuilt `dist/` web assets without downtime.
