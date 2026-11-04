import {
  DrawingUtils,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
import {
  classifyFace,
  createFaceLandmarker,
  drawFaceMesh,
} from "./face-tracking.js";
import {
  classifyGesture,
  createHandLandmarker,
  drawHands,
} from "./hand-tracking.js";

// Configure model limits and the tracking update rate.
const MAX_FACES = 5;
const MAX_HANDS = 4;
const INFERENCE_INTERVAL_MS = 1000 / 15;
const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";

// Cache the page controls and display elements used by the tracking workflow.
const startButton = document.querySelector("#start-button");
const stopButton = document.querySelector("#stop-button");
const statusElement = document.querySelector("#status");
const faceCountElement = document.querySelector("#face-count");
const handCountElement = document.querySelector("#hand-count");
const video = document.querySelector("#camera");
const meshCanvas = document.querySelector("#tracking-overlay");
const meshContext = meshCanvas.getContext("2d");
const scoreControls = [
  { selector: "#weight-fist", output: "#weight-fist-value" },
  { selector: "#weight-raised-fist", output: "#weight-raised-fist-value" },
  { selector: "#weight-middle-finger", output: "#weight-middle-finger-value" },
  { selector: "#weight-open-mouth", output: "#weight-open-mouth-value" },
  { selector: "#weight-lowered-brows", output: "#weight-lowered-brows-value" },
];
const operatorBias = document.querySelector("#operator-bias");
const operatorBiasValue = document.querySelector("#operator-bias-value");
const liveScore = document.querySelector("#live-score");
const trackedHandList = document.querySelector("#tracked-hand-list");
const trackedFaceList = document.querySelector("#tracked-face-list");

// Keep reusable models, camera state, and the latest in-memory cue results.
let faceLandmarker;
let handLandmarker;
const drawingUtils = new DrawingUtils(meshContext);
let cameraStream;
let animationFrameId;
let lastInferenceAt = 0;
let trackedGestures = [];
let trackedFaceCues = [];

// Remove samples saved by older versions; current tracking results stay in memory.
try {
  localStorage.removeItem("mediapipe-face-detections");
} catch (error) {
  statusElement.textContent = `Could not remove previously saved detections: ${error.message}`;
}

// Apply configured cue weights and, for the overall live score, the operator bias.
function getCueScore(cues, includeOperatorBias = true) {
  const weightedCues = cues.reduce(
    (total, cue) => total + Number(document.querySelector(cue.weight).value),
    0,
  );
  const bias = includeOperatorBias ? Number(operatorBias.value) : 0;
  return Math.min(1, bias + weightedCues);
}

// Refresh the live face/hand cue lists and show the highest current score.
function updateLiveScores() {
  trackedHandList.replaceChildren();
  trackedFaceList.replaceChildren();
  const scores = [];

  for (const [index, gesture] of trackedGestures.entries()) {
    if (!gesture) continue;

    const score = getCueScore([gesture]);
    scores.push(score);

    const item = document.createElement("li");
    item.textContent = `Hand ${index + 1}: ${gesture.label} — ${score.toFixed(2)}`;
    trackedHandList.append(item);
  }

  for (const [index, cues] of trackedFaceCues.entries()) {
    const cueScore = getCueScore(cues, false);
    if (cues.length > 0) {
      scores.push(getCueScore(cues));
    }

    const item = document.createElement("li");
    const cueLabels = cues.map(({ label }) => label);
    const cueSummary = cueLabels.length > 0 ? cueLabels.join(", ") : "No configured cues";
    item.textContent = `Face ${index + 1}: ${cueSummary} — ${cueScore.toFixed(2)} / 1`;
    trackedFaceList.append(item);
  }

  liveScore.value = scores.length > 0
    ? Math.max(...scores).toFixed(2)
    : trackedGestures.length > 0 || trackedFaceCues.length > 0
      ? "No weighted cues"
      : "No faces or hands detected";
}

// Mirror slider values in their outputs and recalculate live scores.
function updateScoreControls() {
  for (const { selector, output } of scoreControls) {
    document.querySelector(output).value = Number(document.querySelector(selector).value).toFixed(2);
  }
  operatorBiasValue.value = Number(operatorBias.value).toFixed(2);
  updateLiveScores();
}

// Display a message in the page's accessible status area.
function setStatus(message) {
  statusElement.textContent = message;
}

// Run face and hand inference and update the live overlay, counts, and scores.
function detectFrame() {
  if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    animationFrameId = requestAnimationFrame(detectFrame);
    return;
  }

  const now = performance.now();
  if (now - lastInferenceAt < INFERENCE_INTERVAL_MS) {
    animationFrameId = requestAnimationFrame(detectFrame);
    return;
  }
  lastInferenceAt = now;

  try {
    const faceResult = faceLandmarker.detectForVideo(video, now);
    const handResult = handLandmarker.detectForVideo(video, now);
    faceCountElement.value = String(faceResult.faceLandmarks.length);
    handCountElement.value = String(handResult.landmarks.length);
    trackedGestures = handResult.landmarks.map(classifyGesture);
    trackedFaceCues = faceResult.faceLandmarks.map((_, index) => {
      const blendshapes = faceResult.faceBlendshapes[index];
      return blendshapes ? classifyFace(blendshapes) : [];
    });
    updateLiveScores();
    drawFaceMesh(faceResult, video, meshCanvas, meshContext, drawingUtils);
    drawHands(handResult, drawingUtils);

    if (!cameraStream) return;
  } catch (error) {
    setStatus(`Face or hand tracking failed: ${error.message}`);
    stopCamera();
    return;
  }

  animationFrameId = requestAnimationFrame(detectFrame);
}

// Request camera access, initialize any missing models, and start frame processing.
async function startCamera() {
  startButton.disabled = true;
  setStatus("Requesting camera access…");

  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error(
        "Camera access requires HTTPS (or localhost) and a supported browser. On iPhone, open this page in Safari and allow camera access.",
      );
    }

    cameraStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: "user" },
        width: { ideal: 640 },
        height: { ideal: 480 },
        frameRate: { ideal: 24, max: 30 },
      },
    });
    video.srcObject = cameraStream;
    await video.play();

    if (!faceLandmarker || !handLandmarker) {
      setStatus("Loading face and hand trackers…");
      const vision = await FilesetResolver.forVisionTasks(WASM_URL);
      if (!faceLandmarker) {
        faceLandmarker = await createFaceLandmarker(vision, MAX_FACES);
      }
      if (!handLandmarker) {
        handLandmarker = await createHandLandmarker(vision, MAX_HANDS);
      }
    }
    stopButton.disabled = false;
    stopButton.disabled = false;
    lastInferenceAt = 0;
    setStatus("Camera running. Face and hand tracking are live; results are not saved.");
    animationFrameId = requestAnimationFrame(detectFrame);
  } catch (error) {
    stopCamera();
    setStatus(`Could not start camera or trackers: ${error.message}`);
  } finally {
    startButton.disabled = false;
  }
}

// Stop frame processing and camera tracks, then reset live page state.
function stopCamera() {
  if (animationFrameId !== undefined) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = undefined;
  }
  cameraStream?.getTracks().forEach((track) => track.stop());
  cameraStream = undefined;
  video.srcObject = null;
  meshContext.clearRect(0, 0, meshCanvas.width, meshCanvas.height);
  faceCountElement.value = "0";
  handCountElement.value = "0";
  trackedGestures = [];
  trackedFaceCues = [];
  updateLiveScores();
  stopButton.disabled = true;
  startButton.disabled = false;
  if (statusElement.textContent === "Camera running. Face and hand tracking are live; results are not saved.") {
    setStatus("Camera stopped.");
  }
}

// Connect the page buttons and weight sliders to their actions.
startButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", stopCamera);

for (const { selector } of scoreControls) {
  document.querySelector(selector).addEventListener("input", updateScoreControls);
}
operatorBias.addEventListener("input", updateScoreControls);

// Initialize score outputs when the page loads.
updateScoreControls();
